//! VLM "Auto" concepts: ask a vision-language model for the matte-relevant
//! object CLASSES in the frame, then feed them to SAM3 as text concepts.
//!
//! Logic adapted from an earlier Python VLM subject picker
//! (`_build_class_list_prompt` / `_parse_class_list`): the model returns a
//! short comma list ("person, dog, ball"), never coordinates.
//!
//! Transport: any OpenAI-compatible `/v1/chat/completions` endpoint (Ollama,
//! vLLM, llama.cpp server, LM Studio ...). The default is LOCAL
//! (Ollama on 127.0.0.1) because the frame is uploaded to whatever the URL
//! points at. Results are cached per (frame, settings), so a frame is sent
//! at most once.

use crate::imgutil::{self, Rgb8};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Read;
use std::sync::Mutex;
use std::time::Duration;

pub const DEFAULT_URL: &str = "http://127.0.0.1:11434/v1";
/// Default = the small instruct model: it fits next to Flame on a 24 GB card
/// (~3.3 GB weights). If the endpoint does not have the configured model,
/// [`resolve_model`] picks the best installed vision model instead.
pub const DEFAULT_MODEL: &str = "qwen3-vl:4b-instruct";
/// A cold Ollama load (model not in the page cache) can take well over 20 s
/// next to a busy Flame; warm it is ~4 s load + ~2 s answer (qwen3-vl:4b, 3090).
pub const MAX_TIMEOUT_S: f64 = 60.0;

#[derive(Clone, Debug)]
pub struct VlmOpts {
    pub url: String,
    pub model: String,
    pub api_key: String,
    /// 0 minimal, 1 standard, 2 inclusive, 3 exhaustive
    pub detail: i32,
    /// true = strict EXCLUDE list (no furniture / fixtures / backdrops)
    pub restrict: bool,
    pub timeout_s: f64,
    /// Ollama only: unload the model right after each request (keep_alive 0)
    /// so it does not sit in VRAM next to Flame.
    pub unload: bool,
}

struct Level {
    max: usize,
    floor: f32,
    blurb: &'static str,
}
const LEVELS: [Level; 4] = [
    Level {
        max: 4,
        floor: 0.6,
        blurb: "MINIMAL detail mode — only the 1-4 hero subjects. Drop anything peripheral, partial, or \
                background. Skip groups entirely (the artist will key them separately if needed).",
    },
    Level { max: 8, floor: 0.4, blurb: "STANDARD detail mode — heroes + clearly visible secondary subjects. Cap at 8." },
    Level {
        max: 16,
        floor: 0.3,
        blurb: "INCLUSIVE detail mode — heroes + secondary + peripheral subjects + focal-environmental \
                elements (hero buildings / trees). Cap at 16. Better for shots with many small trackable elements.",
    },
    Level {
        max: 24,
        floor: 0.2,
        blurb: "EXHAUSTIVE detail mode — every plausible matte subject. Cap at 24. Use for crowded scenes \
                where the artist will cherry-pick from the full list. Expect more false positives — confidence \
                floor relaxed to 0.2.",
    },
];

const EXCLUDE_STRICT: &str = "EXCLUDE classes for: sky, clouds, atmosphere, smoke, walls, floors,
ceilings, ground, water, road surfaces, distant terrain, distant
trees, background buildings. Reflections, shadows, motion-blur
trails. BUILT-IN fixtures (fireplaces, mantels, columns, panelling,
doors, doorways, windows, shutters, curtains, beams, pipes — they
are part of the SET, not subjects). FURNITURE (chairs, tables,
sofas, beds, lamps, shelves, cabinets — set dressing). Backdrop
and greenscreen / bluescreen panels.

The test: does this thing MOVE on its own or get manipulated?
Yes → include. No → exclude.";

const EXCLUDE_LOOSE: &str = "EXCLUDE only the universal non-subjects: sky, clouds, atmosphere,
walls (as flat planes), floors, ceilings, ground, road / water
surfaces. Reflections, shadows, motion-blur trails of an already-
listed class.

EVERYTHING ELSE IS FAIR GAME if the artist might want a matte for
it: built-in fixtures (fireplaces, columns, panelling, curtains,
shutters), furniture (chairs, tables, sofas, lamps), set dressing
(books, vases, paintings), backdrops (greenscreen, bluescreen,
projection screens). VFX artists DO need these for relight, swap,
period-fix, or comp-replace work — emit them when present.";

const PROMPT_BASE: &str = "You are a VFX matte-prep assistant.

GOAL — given a film plate frame, return a SHORT comma-separated list
of matte-relevant object CLASSES the shot contains. The list is fed
straight into SAM3 text mode, which then finds and segments every
instance of each class in every frame automatically.

You are NOT locating, naming, or counting individual objects. You
are ONLY listing the KINDS of things present that a VFX artist
would matte.

INCLUDE classes for: people, vehicles (cars/bikes/etc), animals
(specific species), hand-held / worn props (umbrellas, bags, signs,
weapons, instruments), discrete moving objects (balls, debris,
drones), and FOCAL environmental elements only when they are the
clear hero of the shot.

__EXCLUDE_BLOCK__

Use SHORT CLASS-LEVEL terms SAM3 understands:
  • \"person\" not \"man in white suit\"
  • \"car\" not \"silver SUV\"
  • \"dog\" not \"black labrador\"
  • \"umbrella\" not \"red umbrella held by woman\"

Stay generic. SAM3 will find every instance of \"person\" in every
frame; you don't need to enumerate \"person_left\", \"person_right\".

DETAIL LEVEL
============
__DETAIL_BLURB__

OUTPUT
======
A single line: comma-separated class names. NO preamble, NO bullets,
NO JSON, NO markdown. Just the classes.

Examples:
  person, dog, ball
  person, car, bicycle
  person, umbrella
  (empty line if no matte-relevant subjects)

Cap at __MAX__ classes. Drop classes you're below confidence
__FLOOR__ on.
";

pub fn system_prompt(detail: i32, restrict: bool) -> String {
    let l = &LEVELS[detail.clamp(0, 3) as usize];
    PROMPT_BASE
        .replace("__EXCLUDE_BLOCK__", if restrict { EXCLUDE_STRICT } else { EXCLUDE_LOOSE })
        .replace("__DETAIL_BLURB__", l.blurb)
        .replace("__MAX__", &l.max.to_string())
        .replace("__FLOOR__", &format!("{}", l.floor))
}

/// Port of `_parse_class_list`: tolerant of fences, "Classes:" preambles,
/// bullets, articles, duplicates; also strips Qwen-style <think> blocks.
pub fn parse_class_list(text: &str, max: usize) -> Vec<String> {
    let mut t = text.to_string();
    while let (Some(a), Some(b)) = (t.find("<think>"), t.find("</think>")) {
        if b < a {
            break;
        }
        t.replace_range(a..b + "</think>".len(), "");
    }
    let mut cleaned = t.trim().to_string();
    if cleaned.starts_with("```") {
        let mut lines: Vec<&str> = cleaned.lines().collect();
        if lines.first().map(|l| l.starts_with("```")).unwrap_or(false) {
            lines.remove(0);
        }
        if lines.last().map(|l| l.starts_with("```")).unwrap_or(false) {
            lines.pop();
        }
        cleaned = lines.join("\n").trim().to_string();
    }
    let first = cleaned.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let first = match first.split_once(':') {
        Some((_, rest)) => rest,
        None => first,
    };
    let mut out: Vec<String> = Vec::new();
    for raw in first.split([',', '+']) {
        let mut c = raw.trim().trim_matches('.').trim_matches('-').trim_matches('•').trim().to_lowercase();
        if c.is_empty() || c.chars().count() > 32 {
            continue;
        }
        for pre in ["a ", "an ", "the "] {
            if let Some(r) = c.strip_prefix(pre) {
                c = r.to_string();
            }
        }
        if c.is_empty() || out.contains(&c) {
            continue;
        }
        out.push(c);
        if out.len() >= max {
            break;
        }
    }
    out
}

fn endpoint(url: &str) -> String {
    let u = url.trim().trim_end_matches('/');
    if u.ends_with("/chat/completions") {
        u.to_string()
    } else if u.ends_with("/v1") {
        format!("{u}/chat/completions")
    } else {
        format!("{u}/v1/chat/completions")
    }
}

/// Server root without the OpenAI `/v1` (or `/v1/chat/completions`) suffix,
/// where Ollama's native API (`/api/tags`, `/api/generate`) lives.
fn root(url: &str) -> String {
    let u = url.trim().trim_end_matches('/');
    let u = u.strip_suffix("/chat/completions").unwrap_or(u);
    u.strip_suffix("/v1").unwrap_or(u).to_string()
}

/// One installed model as the endpoint lists it.
#[derive(Clone, Debug)]
pub struct Installed {
    pub name: String,
    /// Ollama `details.families` (empty for OpenAI-compatible servers).
    pub families: Vec<String>,
    pub size: u64,
}

/// Is this a vision(-language) model? Ollama families first, then the name.
pub fn is_vision(m: &Installed) -> bool {
    let n = m.name.to_lowercase();
    let fam = m.families.iter().any(|f| {
        let f = f.to_lowercase();
        f.contains("vl") || f == "clip" || f == "mllama" || f == "gemma3" || f.contains("llava")
    });
    if n.starts_with("gemma3:1b") || n.contains("embed") {
        return false;
    }
    fam || n.contains("-vl") || n.contains("vl:") || n.contains("2.5vl") || n.contains("vision") || n.contains("llava")
        || n.contains("minicpm-v") || n.contains("moondream") || n.starts_with("gemma3")
}

/// Parameter count in billions from the tag ("qwen3-vl:4b-instruct" -> 4).
fn billions(name: &str) -> Option<f64> {
    let tag = name.rsplit(':').next().unwrap_or(name).to_lowercase();
    let mut num = String::new();
    for c in tag.chars() {
        if c.is_ascii_digit() || c == '.' {
            num.push(c);
        } else if c == 'b' && !num.is_empty() {
            return num.parse().ok();
        } else {
            num.clear();
        }
    }
    None
}

/// Auto-pick rule (documented in README): the best vision model is a qwen3-vl
/// INSTRUCT model (no thinking preamble), the official tag before re-uploads
/// ("abliterated" etc.), and the SMALLEST of those - cwSegment runs next to
/// Flame, which holds 10-15 GB of a 24 GB card, and the class list needs no
/// big model. Then qwen2.5-vl, then any other vision model, smallest first.
pub fn pick_vision(models: &[Installed]) -> Option<String> {
    let rank = |m: &Installed| {
        let n = m.name.to_lowercase();
        let family = if n.contains("qwen3-vl") || m.families.iter().any(|f| f == "qwen3vl") {
            0
        } else if n.contains("qwen2.5vl") || n.contains("qwen2.5-vl") {
            1
        } else {
            2
        };
        let thinking = n.contains("thinking") as u8;
        let instruct = (!n.contains("instruct")) as u8;
        let reupload = n.contains('/') as u8;
        let size = billions(&m.name).map(|b| (b * 100.0) as u64).unwrap_or(m.size >> 20);
        (family, thinking, instruct, reupload, size, n)
    };
    models.iter().filter(|m| is_vision(m)).min_by_key(|m| rank(m)).map(|m| m.name.clone())
}

/// Installed models: Ollama `/api/tags`, else OpenAI `/v1/models`.
/// Returns (is_ollama, models). Cached per URL for 60 s.
fn installed(url: &str) -> Result<(bool, Vec<Installed>), String> {
    static TAGS: Mutex<Vec<(String, std::time::Instant, bool, Vec<Installed>)>> = Mutex::new(Vec::new());
    let r = root(url);
    if let Some(e) = TAGS.lock().ok().and_then(|g| g.iter().find(|e| e.0 == r && e.1.elapsed().as_secs() < 60).cloned()) {
        return Ok((e.2, e.3));
    }
    let agent = cwtools_core::http::shared_agent();
    let get = |u: &str| -> Result<Value, String> {
        let resp = agent.get(u).timeout(Duration::from_secs(3)).call().map_err(cwtools_core::http::err_str)?;
        let mut s = String::new();
        resp.into_reader().take(4 << 20).read_to_string(&mut s).map_err(|e| format!("read: {e}"))?;
        serde_json::from_str(&s).map_err(|e| format!("bad JSON: {e}"))
    };
    let (ollama, list) = match get(&format!("{r}/api/tags")) {
        Ok(v) if v["models"].is_array() => {
            let l: Vec<Installed> = v["models"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| Installed {
                    name: m["name"].as_str().or(m["model"].as_str()).unwrap_or("").to_string(),
                    families: m["details"]["families"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|f| f.as_str().map(String::from)).collect())
                        .unwrap_or_default(),
                    size: m["size"].as_u64().unwrap_or(0),
                })
                .collect();
            (true, l)
        }
        ollama_err => {
            let v = get(&format!("{r}/v1/models")).map_err(|e| {
                let oe = match ollama_err {
                    Err(x) => x,
                    Ok(_) => "no models list".into(),
                };
                format!("cannot list models at {r} (/api/tags: {oe}; /v1/models: {e})")
            })?;
            let l: Vec<Installed> = v["data"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|m| m["id"].as_str())
                        .map(|n| Installed { name: n.to_string(), families: vec![], size: 0 })
                        .collect()
                })
                .unwrap_or_default();
            (false, l)
        }
    };
    if let Ok(mut g) = TAGS.lock() {
        g.retain(|e| e.0 != r);
        g.push((r, std::time::Instant::now(), ollama, list.clone()));
    }
    Ok((ollama, list))
}

/// The model to ask, plus a note when it is not the configured one.
/// - configured model installed (exact, or Ollama's implicit ":latest") -> it
/// - otherwise -> [`pick_vision`] over the installed models (note says so)
/// - model list not available -> the configured name as-is (the request
///   itself then reports the error)
pub fn resolve_model(o: &VlmOpts) -> Result<(String, bool, Option<String>), String> {
    let want = o.model.trim().to_string();
    let (ollama, list) = match installed(&o.url) {
        Ok(x) => x,
        Err(e) => {
            crate::log::log(&format!("vlm: {e} - using model '{want}' as configured"));
            return Ok((want, false, None));
        }
    };
    let auto = want.is_empty() || want.eq_ignore_ascii_case("auto");
    if !auto && list.iter().any(|m| m.name == want || m.name == format!("{want}:latest")) {
        return Ok((want, ollama, None));
    }
    let names: Vec<&str> = list.iter().map(|m| m.name.as_str()).collect();
    match pick_vision(&list) {
        Some(p) => {
            let note = if auto {
                format!("VLM model auto: {p}")
            } else {
                format!("VLM model '{want}' not installed - using {p}")
            };
            crate::log::log(&format!("vlm: {note} (installed: {names:?})"));
            Ok((p, ollama, Some(note)))
        }
        None => Err(format!("no vision model at {} ('{want}' missing; installed: {})", root(&o.url), names.join(", "))),
    }
}

/// Context window asked of Ollama. The request is ~2k tokens (system prompt
/// + ~870 image tokens at 1280 px + a short answer); Ollama's own default
/// (32k on a 24 GB card) made qwen3-vl:4b take 9.4 GB of VRAM instead of 5.0.
pub const OLLAMA_NUM_CTX: u32 = 4096;

/// Ollama native `/api/chat` body: same prompt as [`request_body`], plus
/// `num_ctx` and `keep_alive` (0 = unload right after the reply), which the
/// OpenAI-compatible endpoint cannot carry.
pub fn ollama_body(img_jpeg: &[u8], o: &VlmOpts) -> Value {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(img_jpeg);
    let mut b = json!({
        "model": o.model,
        "stream": false,
        "options": {"temperature": 0, "seed": 42, "num_predict": 256, "num_ctx": OLLAMA_NUM_CTX},
        "messages": [
            {"role": "system", "content": system_prompt(o.detail, o.restrict)},
            {"role": "user", "content": "List the matte-relevant object classes.", "images": [b64]}
        ]
    });
    if o.unload {
        b["keep_alive"] = json!(0);
    }
    b
}

pub fn request_body(img_jpeg: &[u8], o: &VlmOpts) -> Value {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(img_jpeg);
    json!({
        "model": o.model,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 256,
        "stream": false,
        "messages": [
            {"role": "system", "content": system_prompt(o.detail, o.restrict)},
            {"role": "user", "content": [
                {"type": "text", "text": "List the matte-relevant object classes."},
                {"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b64}")}}
            ]}
        ]
    })
}

static CACHE: Mutex<Vec<(String, Vec<String>)>> = Mutex::new(Vec::new());
const CACHE_CAP: usize = 64;
/// Last VLM result per node (effect handle) — for "Copy to Text".
static LAST: Mutex<Option<HashMap<usize, Vec<String>>>> = Mutex::new(None);

pub fn clear_cache() {
    if let Ok(mut g) = CACHE.lock() {
        g.clear();
    }
}
pub fn set_last(node: usize, c: &[String]) {
    if let Ok(mut g) = LAST.lock() {
        g.get_or_insert_with(HashMap::new).insert(node, c.to_vec());
    }
}
pub fn last(node: usize) -> Option<Vec<String>> {
    LAST.lock().ok()?.as_ref()?.get(&node).cloned()
}

/// One VLM answer: the classes plus a note for the node message (model
/// substitution), if any.
pub struct Detected {
    pub classes: Vec<String>,
    pub note: Option<String>,
}

/// Classes for one frame (cached on frame key + settings). Every request,
/// its time and any error go to the log.
pub fn detect_classes(frame_key: &str, img: &Rgb8, o: &VlmOpts) -> Result<Detected, String> {
    let ck = format!("{frame_key}|{}|{}|{}|{}", endpoint(&o.url), o.model, o.detail, o.restrict);
    if let Some(v) = CACHE.lock().ok().and_then(|g| g.iter().find(|e| e.0 == ck).map(|e| e.1.clone())) {
        return Ok(Detected { classes: v, note: None });
    }
    let (model, ollama, note) = resolve_model(o).map_err(|e| {
        crate::log::log(&format!("vlm: ERROR {e}"));
        e
    })?;
    let jpg = imgutil::encode_jpeg(img, 88)?;
    let mut ro = o.clone();
    ro.model = model.clone();
    // Ollama: native /api/chat (num_ctx + keep_alive); others: OpenAI /v1.
    let (url, body) = if ollama {
        (format!("{}/api/chat", root(&o.url)), ollama_body(&jpg, &ro))
    } else {
        (endpoint(&o.url), request_body(&jpg, &ro))
    };
    let tmo = o.timeout_s.clamp(1.0, MAX_TIMEOUT_S);
    crate::log::log(&format!(
        "vlm: request -> {url} model={model} image={}x{} jpeg={} KB timeout={tmo:.0}s unload={}",
        img.w,
        img.h,
        jpg.len() / 1024,
        ollama && o.unload
    ));
    let t0 = std::time::Instant::now();
    let r = (|| -> Result<Vec<String>, String> {
        let mut req = cwtools_core::http::shared_agent()
            .post(&url)
            .timeout(Duration::from_secs_f64(tmo))
            .set("Content-Type", "application/json");
        if !o.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", o.api_key));
        }
        let resp = req.send_string(&body.to_string()).map_err(cwtools_core::http::err_str)?;
        let mut s = String::new();
        resp.into_reader().take(4 << 20).read_to_string(&mut s).map_err(|e| format!("read: {e}"))?;
        let v: Value = serde_json::from_str(&s).map_err(|e| format!("bad JSON from VLM: {e}"))?;
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .or(v["message"]["content"].as_str())
            .ok_or_else(|| format!("VLM reply has no message content: {}", s.chars().take(160).collect::<String>()))?;
        if ollama {
            let sec = |k: &str| v[k].as_f64().unwrap_or(0.0) / 1e9;
            crate::log::log(&format!(
                "vlm: ollama load {:.1}s prompt {:.1}s answer {:.1}s",
                sec("load_duration"),
                sec("prompt_eval_duration"),
                sec("eval_duration")
            ));
        }
        crate::log::log(&format!("vlm: reply {:?}", content.chars().take(300).collect::<String>()));
        let max = LEVELS[o.detail.clamp(0, 3) as usize].max.min(16);
        Ok(parse_class_list(content, max))
    })();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let classes = match r {
        Ok(c) => c,
        Err(e) => {
            crate::log::log(&format!("vlm: ERROR after {ms:.0} ms from {url} ({model}): {e}"));
            return Err(format!("{model}: {e}"));
        }
    };
    crate::log::log(&format!("vlm: {} classes in {ms:.0} ms from {url} ({model}): {classes:?}", classes.len()));
    if let Ok(mut g) = CACHE.lock() {
        if g.len() >= CACHE_CAP {
            g.remove(0);
        }
        g.push((ck, classes.clone()));
    }
    Ok(Detected { classes, note })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_like_vision_subjects() {
        assert_eq!(parse_class_list("person, dog, ball", 8), vec!["person", "dog", "ball"]);
        assert_eq!(parse_class_list("Classes: The Person, a dog, dog.", 8), vec!["person", "dog"]);
        assert_eq!(parse_class_list("```\nperson + car\n```", 8), vec!["person", "car"]);
        assert_eq!(parse_class_list("<think>hmm, sofa?</think>\nperson, umbrella", 8), vec!["person", "umbrella"]);
        assert_eq!(parse_class_list("", 8), Vec::<String>::new());
        assert_eq!(parse_class_list("a, b, c, d, e", 3).len(), 3);
        let long = "x".repeat(40);
        assert_eq!(parse_class_list(&format!("{long}, cat"), 8), vec!["cat"]);
    }
    #[test]
    fn prompt_and_endpoint() {
        let p = system_prompt(1, true);
        assert!(p.contains("Cap at 8 classes") && p.contains("FURNITURE") && !p.contains("__"));
        assert!(system_prompt(3, false).contains("FAIR GAME"));
        assert_eq!(endpoint("http://h:1/v1/"), "http://h:1/v1/chat/completions");
        assert_eq!(endpoint("http://h:1"), "http://h:1/v1/chat/completions");
        assert_eq!(endpoint("http://h:1/v1/chat/completions"), "http://h:1/v1/chat/completions");
        let o = VlmOpts { url: DEFAULT_URL.into(), model: "m".into(), api_key: String::new(), detail: 1, restrict: true, timeout_s: 5.0, unload: true };
        let b = request_body(b"\xff\xd8", &o);
        assert_eq!(b["model"], "m");
        assert!(b["messages"][1]["content"][1]["image_url"]["url"].as_str().unwrap().starts_with("data:image/jpeg;base64,"));
        assert_eq!(root("http://h:1/v1/"), "http://h:1");
        assert_eq!(root("http://h:1/v1/chat/completions"), "http://h:1");
        assert_eq!(root("http://h:1"), "http://h:1");
        let ob = ollama_body(b"\xff\xd8", &o);
        assert_eq!(ob["keep_alive"], 0);
        assert_eq!(ob["options"]["num_ctx"], OLLAMA_NUM_CTX);
        assert!(ob["messages"][1]["images"][0].as_str().unwrap().starts_with("/9"));
        let o2 = VlmOpts { unload: false, ..o.clone() };
        assert!(ollama_body(b"x", &o2).get("keep_alive").is_none());
    }
    #[test]
    fn pick_vision_model() {
        let m = |n: &str, f: &[&str]| Installed { name: n.into(), families: f.iter().map(|s| s.to_string()).collect(), size: 1 };
        // a real Ollama list (mixed text / vision models)
        let installed = vec![
            m("qwen3.5:27b", &["qwen35"]),
            m("qwen3.5:35b-a3b", &["qwen35moe"]),
            m("gemma3:12b", &["gemma3"]),
            m("qwen3:32b", &["qwen3"]),
            m("qwen3:8b", &["qwen3"]),
            m("community/qwen3-vl-variant:4b-instruct", &["qwen3vl"]),
            m("community/qwen3-vl-variant:8b-instruct", &["qwen3vl"]),
            m("qwen3-vl:8b-instruct", &["qwen3vl"]),
            m("qwen3-vl:4b-instruct", &["qwen3vl"]),
        ];
        assert_eq!(pick_vision(&installed).as_deref(), Some("qwen3-vl:4b-instruct"));
        assert!(!is_vision(&installed[0]) && !is_vision(&installed[4]) && is_vision(&installed[2]));
        assert_eq!(pick_vision(&installed[..5]).as_deref(), Some("gemma3:12b"));
        assert_eq!(pick_vision(&installed[..2]), None);
        // thinking variants lose against instruct; OpenAI-style ids (no families)
        let v = vec![m("qwen3-vl:2b-thinking", &[]), m("qwen3-vl:8b-instruct", &[]), m("Qwen/Qwen2.5-VL-7B-Instruct", &[])];
        assert_eq!(pick_vision(&v).as_deref(), Some("qwen3-vl:8b-instruct"));
        assert_eq!(billions("qwen3-vl:4b-instruct"), Some(4.0));
        assert_eq!(billions("x:latest"), None);
    }
}
