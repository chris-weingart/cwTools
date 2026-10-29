//! Multi-object render path: concepts (typed and/or VLM Auto) → per-instance
//! masks (SAM3 server API v2 or the Local text path) → ordering → Output views:
//!   Matte / Source + Matte / Overlay: union of all objects (or best only)
//!   Object Channels: R/G/B/A of the Output clip (Flame: Result RGB + OutMatte) each carry
//!                    the selection of their slot
//!   ID Map: every object in its own colour, numbered
//!   Object IDs: object k as k / ID_SCALE in RGB (hard, unfiltered), for cwExtract
//! Geometric prompts without concepts (points / box / matte, Local or Remote)
//! give ONE object ("prompt"), so the object views work in Local mode too.
//!
//! Failure policy as in render.rs: never hang, never crash; on any error the
//! plate passes through with empty mattes + a node message.

use crate::imgutil::{self, Gray};
use crate::objects::{self, Instance, SEL_OFF};
use crate::ofx_api::*;
use crate::params::{self, Values};
use crate::prompt::Prompts;
use crate::render::{write_output, Geo};
use cwtools_core::image::{downscale_ofx, Image};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

static INST_CACHE: Mutex<Vec<(String, Arc<Vec<Instance>>)>> = Mutex::new(Vec::new());
const INST_CACHE_CAP: usize = 16;
static LAST_LIST: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);
/// Last detections per node (frame, geometry, numbered objects): Pick Objects
/// turns a viewer click into an object number with them.
type LastDet = (i64, Geo, Arc<Vec<Instance>>);
static LAST_DET: Mutex<Option<HashMap<usize, LastDet>>> = Mutex::new(None);

/// (frame, object number as in the ID Map) under canonical point (x, y) of the
/// detections node `node` rendered last = what the viewer shows. The frame comes
/// from that render: Flame passes no usable time with viewer clicks (picks
/// arrived as frame 0, 2026-10-27).
pub fn object_at(node: usize, x: f64, y: f64) -> Option<(i64, Option<usize>)> {
    let g = LAST_DET.lock().ok()?;
    let (f, geo, insts) = g.as_ref()?.get(&node)?;
    let (mx, my) = geo.canon_to_model(x, y);
    Some((*f, objects::instance_at(insts, mx, my).map(|k| k + 1)))
}

pub fn clear_cache() {
    if let Ok(mut g) = INST_CACHE.lock() {
        g.clear();
    }
}
fn cache_get(k: &str) -> Option<Arc<Vec<Instance>>> {
    let g = INST_CACHE.lock().ok()?;
    g.iter().find(|e| e.0 == k).map(|e| e.1.clone())
}
fn cache_put(k: String, v: Arc<Vec<Instance>>) {
    if let Ok(mut g) = INST_CACHE.lock() {
        g.retain(|e| e.0 != k);
        if g.len() >= INST_CACHE_CAP {
            g.remove(0);
        }
        g.push((k, v));
    }
}
/// Last object list (multi-line) rendered by node `node`.
pub fn last_list(node: usize) -> Option<String> {
    LAST_LIST.lock().ok()?.as_ref()?.get(&node).cloned()
}
fn set_last_list(node: usize, s: String) {
    if let Ok(mut g) = LAST_LIST.lock() {
        g.get_or_insert_with(HashMap::new).insert(node, s);
    }
}

fn mask_bbox(m: &Gray) -> [f32; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (m.w, m.h, 0, 0);
    for y in 0..m.h {
        for x in 0..m.w {
            if m.data[y * m.w + x] >= 0.5 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x1 == 0 {
        [0.0; 4]
    } else {
        [x0 as f32, y0 as f32, x1 as f32, y1 as f32]
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(n).collect::<String>())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_multi(
    eh: OfxImageEffectHandle,
    src: &Image,
    out: &Image,
    win: &OfxRectI,
    v: &Values,
    g: &Geo,
    prompts: &Prompts,
    typed: Vec<String>,
    time: f64,
) -> OfxStatus {
    let node = eh as usize;
    let (mw, mh) = (g.mw, g.mh);
    let rgb = downscale_ofx(src, &[0, 1, 2], mw, mh);
    let img8 = imgutil::rgbf_to_display8(&imgutil::RgbF { w: mw, h: mh, data: rgb }, v.colorspace);
    let fk = format!("cwseg_{}_{}x{}", imgutil::frame_key(&img8), mw, mh);
    let aborted = || unsafe { (crate::effect().abort)(eh) } != 0;

    // ── concepts ──────────────────────────────────────────────────────
    let mut concepts = typed.clone();
    let mut notes: Vec<String> = Vec::new();
    if v.vlm_auto != 0 {
        if aborted() {
            return kOfxStatOK;
        }
        match crate::vlm::detect_classes(&fk, &img8, &v.vlm) {
            Ok(d) => {
                if let Some(n) = d.note {
                    notes.push(n);
                }
                let c = d.classes;
                crate::vlm::set_last(node, &c);
                concepts = if v.vlm_auto == 1 { objects::parse_concepts(&c.join(",")) } else { objects::merge_concepts(&typed, &c) };
                if c.is_empty() {
                    notes.push("VLM found no classes".into());
                }
            }
            Err(e) => notes.push(format!("VLM failed ({}) - using Text Prompt", truncate(&e, 200))),
        }
    }
    let has_geo = !prompts.points.is_empty() || prompts.bbox.is_some();
    // Tracking: only the anchor frame detects (its objects start the tracker).
    let fr = time.round() as i64;
    let tracking = v.track != crate::track::DIR_OFF && !v.track_list.is_empty();
    if tracking && crate::log::verbose() {
        crate::log::log(&format!("track: render time {time} -> frame {fr} (anchor {}, objects {})", v.track_anchor, crate::track::format_list(&v.track_list)));
    }
    let local_track = tracking && v.mode == 1;

    // ── objects ───────────────────────────────────────────────────────
    let ropts = crate::remote::RemoteOpts {
        server: v.server.clone(),
        timeout_s: v.timeout,
        best_only: v.best_only,
        score_threshold: v.score_threshold,
        jpeg_quality: 88,
    };
    let t0 = std::time::Instant::now();
    // Concepts: Remote (also with an exemplar box) or Local text-only (Local has
    // no exemplar boxes: points / box there go the single-object tracker path).
    let result: Result<Arc<Vec<Instance>>, String> = if local_track && fr != v.track_anchor {
        Ok(Arc::new(Vec::new()))
    } else if !concepts.is_empty() && (v.mode == 0 || !has_geo) {
        let key = format!(
            "inst|{}|{}|{fk}|{:?}|{:.2}|{}|{:?}",
            v.mode, if v.mode == 1 { &v.model_dir } else { &v.server }, concepts, v.score_threshold, v.max_objects, prompts.bbox
        );
        match if v.cache { cache_get(&key) } else { None } {
            Some(c) => Ok(c),
            None => {
                if aborted() {
                    return kOfxStatOK;
                }
                let r = if v.mode == 1 {
                    crate::local::segment_instances(&fk, &img8, &concepts, v, v.max_objects)
                } else {
                    crate::remote::segment_instances(&fk, &img8, &concepts, prompts.bbox, &ropts, v.max_objects)
                };
                r.map(|mut insts| {
                    for i in insts.iter_mut() {
                        if i.w != mw || i.h != mh {
                            let gm = imgutil::bilinear_gray(&i.to_gray(), mw, mh);
                            let (sx, sy) = (mw as f32 / i.w as f32, mh as f32 / i.h as f32);
                            i.bbox = [i.bbox[0] * sx, i.bbox[1] * sy, i.bbox[2] * sx, i.bbox[3] * sy];
                            i.mask = gm.data.iter().map(|p| (p.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
                            i.w = mw;
                            i.h = mh;
                        }
                    }
                    let a = Arc::new(insts);
                    if v.cache {
                        cache_put(key, a.clone());
                    }
                    a
                })
            }
        }
    } else if has_geo {
        let mut p = prompts.clone();
        p.text.clear();
        let key = format!("single|{}|{}|{fk}|{}|{}", v.mode, v.server, p.key(), v.model_dir);
        match if v.cache { cache_get(&key) } else { None } {
            Some(c) => Ok(c),
            None => {
                if aborted() {
                    return kOfxStatOK;
                }
                let r = if v.mode == 1 {
                    crate::local::segment(&fk, &img8, &p, v)
                } else {
                    crate::remote::segment(&fk, &img8, &p, &ropts).map(|r| r.mask)
                };
                r.map(|m| {
                    let m = if m.w != mw || m.h != mh { imgutil::bilinear_gray(&m, mw, mh) } else { m };
                    let inst = Instance {
                        concept: "prompt".into(),
                        concept_index: 0,
                        score: 1.0,
                        bbox: mask_bbox(&m),
                        w: mw,
                        h: mh,
                        mask: m.data.iter().map(|p| (p.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect(),
                    };
                    let a = Arc::new(vec![inst]);
                    if v.cache {
                        cache_put(key, a.clone());
                    }
                    a
                })
            }
        }
    } else {
        Err(String::new()) // no prompt at all
    };

    let mut insts: Vec<Instance> = Vec::new();
    // VLM notes (model substitution, errors) lead every message below.
    let head = if notes.is_empty() { String::new() } else { format!("{} | ", notes.join("; ")) };
    if let Ok(a) = &result {
        insts = (**a).clone();
        objects::order(&mut insts, v.order);
        if !(local_track && fr != v.track_anchor) {
            if let Ok(mut last) = LAST_DET.lock() {
                last.get_or_insert_with(HashMap::new).insert(node, (fr, g.clone(), Arc::new(insts.clone())));
            }
        }
    }
    // concepts the Object Channels slots count (Concept #): the picked objects once there are some
    let mut slot_concepts = concepts.clone();
    // Picked objects without tracking (Track Off, or Remote): on the anchor frame the
    // output carries just them, in list order, as tracking would.
    if !v.track_list.is_empty() && !local_track && !v.picking && fr == v.track_anchor {
        let picked: Vec<Instance> = v
            .track_list
            .iter()
            .enumerate()
            .filter_map(|(k, n)| insts.get(n - 1).map(|i| Instance { concept: format!("pick {n}"), concept_index: k, ..i.clone() }))
            .collect();
        insts = picked;
        slot_concepts = insts.iter().map(|i| i.concept.clone()).collect();
    }
    let mut note: Option<String> = None; // viewer text on a frame not tracked yet
    if local_track {
        let key = track_key(v, &concepts, mw, mh);
        let c0 = cwtools_core::image::clip_frame_range(eh, kOfxImageEffectSimpleSourceClipName).map_or(0, |r| r.0.round() as i64);
        let num = |f: i64| f - c0 + v.track_base; // Flame's frame number
        // A job of an older selection (objects / prompt changed) stops first.
        let job = crate::track::job(node).filter(|j| j.running);
        if let Some(j) = &job {
            if j.key != key {
                crate::track::cancel_job(node);
            }
        }
        let mut batch_msg = None;
        if crate::track::take_run(node) {
            batch_msg = Some(match &job {
                Some(_) => {
                    crate::track::cancel_job(node);
                    "cwSegment track: stopping the background tracking (press Track Range again to continue)".to_string()
                }
                None => start_batch(eh, v, g, &concepts, node, &key, num),
            });
        }
        let busy = crate::track::job(node).map_or(false, |j| j.running);
        let anchor_insts = insts.clone();
        let req = crate::track::Request {
            node, frame: fr, anchor: v.track_anchor, numbers: &v.track_list, dir: v.track, key: key.clone(),
            anchor_insts: (fr == v.track_anchor).then_some(&anchor_insts[..]), mw, mh,
        };
        // While a job tracks in the background, renders only read its results.
        let res = if busy {
            Ok(crate::track::cached(&req).unwrap_or(crate::track::Frame::NotYet(fr)))
        } else {
            let mut be = crate::local::TrackBackend { v, fk: &fk, img: &img8 };
            crate::track::frame(&req, &mut be)
        };
        let progress = crate::track::job(node).filter(|j| j.key == key);
        let range_txt = || match crate::track::status(node) {
            Some((_, a, b, _)) if a == b => format!("tracked frame {}", num(a)),
            Some((_, a, b, _)) => format!("tracked frames {}-{}", num(a), num(b)),
            None => "nothing tracked yet".to_string(),
        };
        let msg = match res {
            Ok(crate::track::Frame::Objects(objs)) => {
                insts = objs
                    .into_iter()
                    .enumerate()
                    .map(|(k, (n, m, score))| Instance {
                        concept: format!("track {n}"),
                        concept_index: k,
                        score: 1.0 / (1.0 + (-score).exp()),
                        bbox: mask_bbox(&m),
                        w: mw,
                        h: mh,
                        mask: m.data.iter().map(|p| (p.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect(),
                    })
                    .collect();
                slot_concepts = insts.iter().map(|i| i.concept.clone()).collect();
                set_last_list(node, objects::list_lines(&insts, &slot_concepts));
                let lost: Vec<String> = insts.iter().filter(|i| i.score < 0.5).map(|i| i.concept.clone()).collect();
                if busy {
                    progress.as_ref().map(|j| format!("{head}cwSegment track: tracking in the background, {} of {} frames done - {}", j.done, j.total, range_txt()))
                } else if !lost.is_empty() {
                    Some(format!("{head}cwSegment track: not visible on frame {}: {}", num(fr), lost.join(", ")))
                } else if v.list_in_message || !notes.is_empty() {
                    Some(truncate(&format!("{head}cwSegment track: {}", range_txt()), 480))
                } else {
                    None
                }
            }
            Ok(crate::track::Frame::NotYet(_)) => {
                insts.clear();
                note = Some(match &progress {
                    Some(j) if busy => format!("tracking {} / {}", j.done, j.total),
                    _ => "not tracked yet - press Track Range".to_string(),
                });
                Some(match (busy, &progress) {
                    (true, Some(j)) => format!(
                        "{head}cwSegment track: tracking in the background, {} of {} frames done - {} (this one is not done yet)",
                        j.done, j.total, range_txt()
                    ),
                    _ => format!("{head}cwSegment track: frame {} is not tracked yet - press Track Range ({})", num(fr), range_txt()),
                })
            }
            Err(e) => {
                insts.clear();
                Some(format!("{head}cwSegment track: {e}"))
            }
        };
        // A finished job's error or summary is shown once on the next render.
        let done_msg = progress.filter(|j| !j.running && (j.error.is_some() || j.total > 0)).map(|j| {
            crate::track::job_update(node, |x| *x = crate::track::Job::default());
            match j.error {
                Some(e) => format!("{head}cwSegment track: stopped: {e} ({})", range_txt()),
                None if j.cancel => format!("{head}cwSegment track: stopped after {} of {} frames ({})", j.done, j.total, range_txt()),
                None => format!("{head}cwSegment track: done, {} frames ({})", j.done, range_txt()),
            }
        });
        let shown = batch_msg.map(|m| format!("{head}{m}")).or(done_msg).or(msg);
        crate::set_node_message(eh, shown.as_deref());
    }
    match &result {
        _ if local_track => {}
        Ok(_) => {
            set_last_list(node, objects::list_lines(&insts, &concepts));
            if tracking {
                crate::set_node_message(eh, Some(&format!("{head}cwSegment: Track needs Device = Local (tracking runs on this workstation)")));
            } else if insts.is_empty() {
                crate::set_node_message(eh, Some(&format!("{head}cwSegment: no objects found for: {}", concepts.join(", "))));
            } else if v.list_in_message || !notes.is_empty() {
                crate::set_node_message(eh, Some(&truncate(&format!("{head}cwSegment objects: {}", objects::list_text(&insts)), 480)));
            } else {
                crate::set_node_message(eh, None);
            }
        }
        Err(e) if e.is_empty() => {
            crate::set_node_message(eh, Some(&format!("{head}cwSegment: no prompt - set Text, Points, Box, Use Prompt Mask or Auto Concepts")));
        }
        Err(e) => {
            let mode = if v.mode == 1 { "Local" } else { "Remote" };
            crate::set_node_message(eh, Some(&format!("{head}cwSegment {mode} failed: {e}")));
        }
    }
    if crate::log::verbose() {
        crate::log::log(&format!(
            "multi: output={} concepts={:?} objects={} {:.0}ms list=[{}]",
            v.output,
            concepts,
            insts.len(),
            t0.elapsed().as_secs_f64() * 1000.0,
            objects::list_text(&insts)
        ));
    }

    let (fw, fh) = (g.fw, g.fh);
    let to_full = |m: &Gray| -> Gray {
        let mut f = imgutil::bilinear_gray(m, fw, fh);
        if !v.soft {
            for x in f.data.iter_mut() {
                *x = if *x >= 0.5 { 1.0 } else { 0.0 };
            }
        }
        f
    };

    match v.output {
        // Object Channels (the default) before anything is detected: the plate,
        // matte 0, rather than four black channels.
        params::OUT_CHANNELS if insts.is_empty() => {
            let plate = Values { output: params::OUT_SRC_MATTE, invert: false, ..v.clone() };
            write_output(src, out, win, None, &plate, g, prompts);
        }
        params::OUT_CHANNELS => {
            // Picked slots: the instance under the slot's viewer pick point on this frame.
            let sels: Vec<objects::Sel> = v
                .sels
                .iter()
                .zip(&v.picks)
                .map(|(s, p)| match (s.kind, p) {
                    (objects::SEL_PICK, Some((px, py))) => {
                        let (mx, my) = g.canon_to_model(*px, *py);
                        match objects::instance_at(&insts, mx, my) {
                            Some(k) => objects::Sel { kind: objects::SEL_INSTANCE, index: k as i32 + 1 },
                            None => objects::Sel { kind: SEL_OFF, index: 0 },
                        }
                    }
                    _ => *s,
                })
                .collect();
            let mut chans: Vec<Option<Gray>> = sels.iter().map(|s| objects::select(&insts, *s).map(|m| to_full(&m))).collect();
            // Matte (A) on a Concept # beyond the concepts entered (fewer than 4
            // typed): the full matte, all objects. Decided by the prompt, never by
            // what was found on the frame, so an object leaving the frame keeps its
            // channel black (user 2026-10-22).
            if sels[3].kind == objects::SEL_CONCEPT && sels[3].index as usize > slot_concepts.len() {
                chans[3] = objects::select(&insts, objects::Sel { kind: objects::SEL_ALL, index: 0 }).map(|m| to_full(&m));
            }
            write_channels(src, out, win, &chans[..4], &sels[..4], v.invert);
        }
        params::OUT_IDMAP => write_idmap(src, out, win, &insts, v, g, &to_full),
        params::OUT_OBJIDS => {
            if insts.len() > params::ID_MAX {
                crate::set_node_message(eh, Some(&format!(
                    "cwSegment: {} objects, Object IDs holds the first {}", insts.len(), params::ID_MAX)));
            }
            let lab = full_labels(&insts, fw, fh);
            write_objids(src, out, win, &lab, fw, v.invert);
        }
        _ => {
            let full = objects::union_all(&insts, v.best_only).map(|m| to_full(&m));
            write_output(src, out, win, full.as_ref(), v, g, prompts);
        }
    }
    if let Some(t) = &note {
        draw_note(src, out, win, g, t);
    }
    kOfxStatOK
}

/// Text in the middle of the Result (RGB only, white on a black outline).
fn draw_note(src: &Image, out: &Image, win: &OfxRectI, g: &Geo, txt: &str) {
    let sb = src.bounds;
    let (x1, x2) = (win.x1.max(out.bounds.x1), win.x2.min(out.bounds.x2));
    let (y1, y2) = (win.y1.max(out.bounds.y1), win.y2.min(out.bounds.y2));
    let cells = txt.len() as i32 * 4 + 2;
    let s = ((g.fh as i32) / 120).min(g.fw as i32 * 9 / 10 / cells).max(2);
    for (px, py, ink) in objects::text_pixels(txt, g.fw as i32 / 2, g.fh as i32 / 2, s) {
        let (ox, oy) = (sb.x1 + px, sb.y2 - 1 - py);
        if ox < x1 || ox >= x2 || oy < y1 || oy >= y2 {
            continue;
        }
        let c = if ink { 1.0 } else { 0.0 };
        let op = unsafe { out.px(ox, oy) };
        unsafe {
            *op = c;
            *op.add(1) = c;
            *op.add(2) = c;
        }
    }
}

/// Session key: anything that changes the anchor objects starts the track over.
fn track_key(v: &Values, concepts: &[String], mw: usize, mh: usize) -> String {
    format!(
        "{}|{:?}|{:?}|{:.3}|{}|{}|{}|{}|{}|{mw}x{mh}",
        v.track_anchor, v.track_list, concepts, v.score_threshold, v.order, v.max_objects, v.model_dir, v.use_gpu, v.colorspace
    )
}

/// Track Range: fetch the Source frames from the tracked range outwards (anchor
/// first) inside this render, then track them on a background thread so Flame stays
/// usable (Flame shows no OFX progress dialog; tracking inside the render froze it).
/// Renders read the finished frames meanwhile. Returns the node message.
#[allow(clippy::too_many_arguments)]
fn start_batch(
    eh: OfxImageEffectHandle, v: &Values, g: &Geo, concepts: &[String], node: usize, key: &str, num: impl Fn(i64) -> i64,
) -> String {
    use crate::track::{self, DIR_BACK, DIR_BOTH, DIR_FWD, MAX_BATCH};
    let anchor = v.track_anchor;
    let (c0, c1) = cwtools_core::image::clip_frame_range(eh, kOfxImageEffectSimpleSourceClipName)
        .map(|(a, b)| (a.round() as i64, b.round() as i64))
        .unwrap_or((anchor, anchor));
    // Track From / To are Batch numbers; 0 and 0 = the whole clip
    let whole = v.track_from == 0 && v.track_to == 0;
    let clip = |n: i64| (n - v.track_base + c0).clamp(c0, c1);
    let to = if whole { c1 } else { clip(v.track_to) };
    let from = if whole { c0 } else { clip(v.track_from).min(to) };
    let done = track::range(node, key);
    let mut todo: Vec<i64> = Vec::new();
    if done.is_none() {
        todo.push(anchor);
    }
    if v.track == DIR_FWD || v.track == DIR_BOTH {
        todo.extend(done.map_or(anchor + 1, |r| r.1 + 1)..=to);
    }
    if v.track == DIR_BACK || v.track == DIR_BOTH {
        todo.extend((from..=done.map_or(anchor - 1, |r| r.0 - 1)).rev());
    }
    if todo.is_empty() {
        return format!("cwSegment track: frames {}-{} are tracked already", num(from), num(to));
    }
    let more = todo.len() > MAX_BATCH;
    todo.truncate(MAX_BATCH);
    // fetch + downscale in this render (fast); the models run in the background
    let t0 = std::time::Instant::now();
    let (mw, mh) = (g.mw, g.mh);
    let mut frames: Vec<(i64, imgutil::Rgb8, String)> = Vec::with_capacity(todo.len());
    // Flame 2027.2 offers the progress suite but showed no bar on macOS; harmless if it stays hidden.
    let bar = cwtools_core::suites::Progress::start(eh, "cwSegment: reading frames for Track Range");
    for (k, &f) in todo.iter().enumerate() {
        if !bar.update(k as f64 / todo.len() as f64) {
            return "cwSegment track: cancelled".to_string();
        }
        let Some(src) = cwtools_core::image::fetch_image(eh, kOfxImageEffectSimpleSourceClipName, f as f64) else {
            return format!("cwSegment track: Flame gave the plugin no image of frame {} - step through the frames instead", num(f));
        };
        let rgb = downscale_ofx(&src, &[0, 1, 2], mw, mh);
        cwtools_core::image::release(&src);
        let img8 = imgutil::rgbf_to_display8(&imgutil::RgbF { w: mw, h: mh, data: rgb }, v.colorspace);
        let fk = format!("cwseg_{}_{}x{}", imgutil::frame_key(&img8), mw, mh);
        frames.push((f, img8, fk));
    }
    let bar_shown = bar.shown();
    drop(bar);
    crate::track::set_fetch_ms((t0.elapsed().as_millis() as u64 / frames.len().max(1) as u64).max(1));
    crate::log::log(&format!(
        "track range: fetched {} frames in {:.1} s (clip {c0}-{c1}, anchor {anchor}, progress bar {}), tracking in the background",
        frames.len(), t0.elapsed().as_secs_f64(), if bar_shown { "started" } else { "not offered" }
    ));
    let total = frames.len();
    track::job_update(node, |j| *j = track::Job { key: key.to_string(), done: 0, total, running: true, cancel: false, error: None });
    let (v, concepts, key) = (v.clone(), concepts.to_vec(), key.to_string());
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        for (i, (f, img, fk)) in frames.iter().enumerate() {
            if track::job(node).map_or(true, |j| j.cancel || j.key != key) {
                break;
            }
            let r = (|| -> Result<(), String> {
                let anchor_insts = if *f == v.track_anchor && track::range(node, &key).is_none() {
                    let mut d = crate::local::segment_instances(fk, img, &concepts, &v, v.max_objects)?;
                    objects::order(&mut d, v.order);
                    Some(d)
                } else {
                    None
                };
                let mut be = crate::local::TrackBackend { v: &v, fk, img };
                let req = track::Request {
                    node, frame: *f, anchor: v.track_anchor, numbers: &v.track_list, dir: v.track, key: key.clone(),
                    anchor_insts: anchor_insts.as_deref(), mw: img.w, mh: img.h,
                };
                match track::frame(&req, &mut be)? {
                    track::Frame::Objects(_) => Ok(()),
                    track::Frame::NotYet(n) => Err(format!("frame {f} is not next to the tracked range (next is {n})")),
                }
            })();
            match r {
                Ok(()) => track::job_update(node, |j| j.done = i + 1),
                Err(e) => {
                    crate::log::log(&format!("track range: clip time {f}: {e}"));
                    track::job_update(node, |j| j.error = Some(e));
                    break;
                }
            }
        }
        track::job_update(node, |j| j.running = false);
        crate::log::log(&format!("track range: background job ended after {:.0} s", t0.elapsed().as_secs_f64()));
    });
    format!(
        "cwSegment track: tracking {total} frames in the background (from frame {}){} - scrub to see the tracked frames",
        num(todo[0]),
        if more { format!("; the first {MAX_BATCH}, press Track Range again for more") } else { String::new() }
    )
}

/// Write four channel masks (full res, top-down, None = black) into an
/// RGBA / RGB output image.
fn write_channels(src: &Image, out: &Image, win: &OfxRectI, ch: &[Option<Gray>], sels: &[objects::Sel], invert: bool) {
    let sb = src.bounds;
    let (x1, x2) = (win.x1.max(out.bounds.x1), win.x2.min(out.bounds.x2));
    let (y1, y2) = (win.y1.max(out.bounds.y1), win.y2.min(out.bounds.y2));
    let nc = (out.n_comps.max(1) as usize).min(4);
    for y in y1..y2 {
        for x in x1..x2 {
            let inside = x >= sb.x1 && x < sb.x2 && y >= sb.y1 && y < sb.y2;
            let op = unsafe { out.px(x, y) };
            for k in 0..4 {
                // Alpha-only output: its single channel carries slot A.
                let dst = if nc == 1 {
                    if k != 3 {
                        continue;
                    }
                    0
                } else if k >= nc {
                    continue;
                } else {
                    k
                };
                let mut m = match (&ch[k], inside) {
                    (Some(mk), true) => mk.data[(sb.y2 - 1 - y) as usize * mk.w + (x - sb.x1) as usize],
                    _ => 0.0,
                };
                if invert && sels[k].kind != SEL_OFF {
                    m = 1.0 - m;
                }
                unsafe { *op.add(dst) = m };
            }
        }
    }
}

/// Full-resolution hard labels (top-down, 0 = background, k+1 = insts[k]) for the
/// first ID_MAX objects. Same overlap rule as `objects::id_labels`: biggest first,
/// smaller objects overwrite. One object upscaled at a time, so memory stays at one
/// float frame plus one byte per pixel whatever the object count.
fn full_labels(insts: &[Instance], fw: usize, fh: usize) -> Vec<u8> {
    let mut lab = vec![0u8; fw * fh];
    let n = insts.len().min(params::ID_MAX);
    let mut by_area: Vec<(usize, usize)> = insts[..n].iter().enumerate().map(|(k, i)| (i.area(), k)).collect();
    by_area.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, k) in by_area {
        let m = imgutil::bilinear_gray(&insts[k].to_gray(), fw, fh);
        for (l, x) in lab.iter_mut().zip(&m.data) {
            if *x >= 0.5 {
                *l = (k + 1) as u8;
            }
        }
    }
    lab
}

/// Object IDs view: k / ID_SCALE in R = G = B, A = any object (inverted with Invert;
/// the IDs never are).
fn write_objids(src: &Image, out: &Image, win: &OfxRectI, lab: &[u8], fw: usize, invert: bool) {
    let sb = src.bounds;
    let (x1, x2) = (win.x1.max(out.bounds.x1), win.x2.min(out.bounds.x2));
    let (y1, y2) = (win.y1.max(out.bounds.y1), win.y2.min(out.bounds.y2));
    let nc = (out.n_comps.max(1) as usize).min(4);
    for y in y1..y2 {
        for x in x1..x2 {
            let inside = x >= sb.x1 && x < sb.x2 && y >= sb.y1 && y < sb.y2;
            let id = if inside { lab[(sb.y2 - 1 - y) as usize * fw + (x - sb.x1) as usize] } else { 0 };
            let val = id as f32 / params::ID_SCALE;
            let mut a = if id > 0 { 1.0 } else { 0.0 };
            if invert {
                a = 1.0 - a;
            }
            let op = unsafe { out.px(x, y) };
            unsafe {
                if nc == 1 {
                    *op = a;
                    continue;
                }
                for c in 0..nc.min(3) {
                    *op.add(c) = val;
                }
                if nc == 4 {
                    *op.add(3) = a;
                }
            }
        }
    }
}

fn write_idmap(src: &Image, out: &Image, win: &OfxRectI, insts: &[Instance], v: &Values, g: &Geo, to_full: &dyn Fn(&Gray) -> Gray) {
    let sb = src.bounds;
    let (fw, fh) = (g.fw, g.fh);
    let labels = objects::id_labels(insts);
    // Picking objects to track: picked objects in colour (alpha = them), the others grey.
    let picked: Vec<usize> = if v.picking { v.track_list.clone() } else { Vec::new() };
    let alpha = if picked.is_empty() {
        objects::union_all(insts, false).map(|m| to_full(&m))
    } else {
        let sel: Vec<&Instance> = picked.iter().filter_map(|n| insts.get(n - 1)).collect();
        objects::union_of(&sel).map(|m| to_full(&m))
    };
    let cols: Vec<[f32; 3]> = (0..=insts.len())
        .map(|k| if k == 0 || picked.is_empty() || picked.contains(&k) { objects::palette(k) } else { [0.32, 0.32, 0.32] })
        .collect();
    let (x1, x2) = (win.x1.max(out.bounds.x1), win.x2.min(out.bounds.x2));
    let (y1, y2) = (win.y1.max(out.bounds.y1), win.y2.min(out.bounds.y2));
    for y in y1..y2 {
        for x in x1..x2 {
            let inside = x >= sb.x1 && x < sb.x2 && y >= sb.y1 && y < sb.y2;
            let (mut r, mut gg, mut b, mut a) = (0.0, 0.0, 0.0, 0.0);
            if inside {
                let sp = unsafe { src.px(x, y) };
                let (sr, sg, sbb) = unsafe { (*sp, *sp.add(1), *sp.add(2)) };
                let (col, row) = ((x - sb.x1) as usize, (sb.y2 - 1 - y) as usize);
                let lab = labels.as_ref().map(|(lw, lh, l)| {
                    let mx = (col * lw / fw).min(lw - 1);
                    let my = (row * lh / fh).min(lh - 1);
                    l[my * lw + mx]
                });
                match lab {
                    Some(k) if k > 0 => {
                        let c = cols[k as usize];
                        (r, gg, b) = (c[0], c[1], c[2]);
                    }
                    _ => (r, gg, b) = (sr * 0.25, sg * 0.25, sbb * 0.25),
                }
                if let Some(am) = &alpha {
                    a = am.data[row * am.w + col];
                    if v.invert {
                        a = 1.0 - a;
                    }
                }
            }
            let op = unsafe { out.px(x, y) };
            unsafe {
                *op = r;
                *op.add(1) = gg;
                *op.add(2) = b;
                if out.n_comps == 4 {
                    *op.add(3) = a;
                }
            }
        }
    }
    if !v.id_labels {
        return;
    }
    let s = ((fh as i32) / 180).max(2);
    for (k, inst) in insts.iter().enumerate() {
        let (cx, cy) = inst.centroid();
        let fx = (cx * fw as f32 / inst.w.max(1) as f32) as i32;
        let fy = (cy * fh as f32 / inst.h.max(1) as f32) as i32;
        for (px, py, ink) in objects::glyph_pixels(k + 1, fx, fy, s) {
            let (ox, oy) = (sb.x1 + px, sb.y2 - 1 - py);
            if ox < x1 || ox >= x2 || oy < y1 || oy >= y2 {
                continue;
            }
            let c = if ink { 1.0 } else { 0.0 };
            let op = unsafe { out.px(ox, oy) };
            unsafe {
                *op = c;
                *op.add(1) = c;
                *op.add(2) = c;
            }
        }
    }
}
