//! Parameter definitions + readback.
//!
//! Flame constraints (flame_ofx_host_facts): no Pages (use Groups), no
//! parametric params, String params never animate, hints truncate at 255
//! chars, Double labels clip at ~15 chars (guard tests below).

use crate::ofx_api::*;
use crate::{prop_set_double, prop_set_int};
use cwtools_core::params::{boolp, button, caption, choice, define, hint, int, label, parent, string, xy};

pub const MAX_POINTS: usize = 6;
/// Point / box / prompt-mask prompting in the panel (hidden for now,
/// text prompts only). The params stay described (secret), so the code paths keep
/// working; viewer clicks default to Off so nothing adds hidden points.
pub const PROMPT_UI: bool = false;

pub const P_MODE: &[u8] = b"mode\0"; // 0 remote, 1 local
pub const P_TEXT: &[u8] = b"textPrompt\0";
pub const P_USE_MATTE: &[u8] = b"usePromptMatte\0";
pub const P_MATTE_AS: &[u8] = b"matteAs\0"; // 0 box+points, 1 box, 2 points
pub const P_INVERT: &[u8] = b"invert\0";
pub const P_OUTPUT: &[u8] = b"outputMode\0"; // 0 matte, 1 rgb+matte, 2 overlay
pub const P_COLORSPACE: &[u8] = b"inputColorSpace\0";
pub const P_CLICK_MODE: &[u8] = b"clickMode\0"; // 0 off, 1 fg, 2 bg, 3 box
pub const P_CLEAR_POINTS: &[u8] = b"clearPoints\0";
pub const P_USE_BOX: &[u8] = b"useBox\0";
pub const P_BOX_A: &[u8] = b"boxCornerA\0";
pub const P_BOX_B: &[u8] = b"boxCornerB\0";
pub const P_SERVER: &[u8] = b"serverUrl\0";
pub const P_SEND_SIZE: &[u8] = b"sendSize\0";
pub const P_TIMEOUT: &[u8] = b"timeoutSec\0";
pub const P_MODEL_DIR: &[u8] = b"modelDir\0";
pub const P_ORT_LIB: &[u8] = b"ortLibrary\0";
pub const P_USE_GPU: &[u8] = b"useGpu\0";
pub const P_FREE_VRAM: &[u8] = b"freeVramPerFrame\0";
pub const P_RELEASE: &[u8] = b"releaseGpu\0";
pub const P_CACHE: &[u8] = b"cache\0";
pub const P_CLEAR_CACHE: &[u8] = b"clearCache\0";
pub const P_TEST_SERVER: &[u8] = b"testServer\0";
pub const P_BEST_ONLY: &[u8] = b"bestOnly\0";
pub const P_SCORE: &[u8] = b"scoreThreshold\0";
pub const P_SOFT: &[u8] = b"softMask\0";
// Multi-object
pub const P_ORDER: &[u8] = b"objOrder\0";
pub const P_MAX_OBJ: &[u8] = b"maxObjects\0";
pub const P_ID_LABELS: &[u8] = b"idLabels\0";
pub const P_LIST_MSG: &[u8] = b"listInMessage\0";
pub const P_OBJ_LIST: &[u8] = b"objectList\0";
pub const P_REFRESH_LIST: &[u8] = b"refreshList\0";
pub const P_VLM_AUTO: &[u8] = b"vlmAuto\0"; // 0 off, 1 replace text, 2 add to text
pub const P_VLM_URL: &[u8] = b"vlmUrl\0";
pub const P_VLM_MODEL: &[u8] = b"vlmModel\0";
pub const P_VLM_KEY: &[u8] = b"vlmApiKey\0";
pub const P_VLM_DETAIL: &[u8] = b"vlmDetail\0";
pub const P_VLM_RESTRICT: &[u8] = b"vlmRestrict\0";
pub const P_VLM_TIMEOUT: &[u8] = b"vlmTimeout\0";
pub const P_VLM_COPY: &[u8] = b"vlmCopy\0";
pub const P_VLM_UNLOAD: &[u8] = b"vlmUnload\0";
// Tracking (crate::track)
pub const P_TRACK: &[u8] = b"track\0"; // 0 off, 1 forward, 2 backward, 3 both
pub const P_TRACK_PICK: &[u8] = b"trackPick\0";
pub const P_TRACK_CLEAR: &[u8] = b"trackClear\0";
pub const P_TRACK_RESET: &[u8] = b"trackReset\0";
pub const P_TRACK_LIST: &[u8] = b"trackList\0";
pub const P_TRACK_ANCHOR: &[u8] = b"trackAnchor\0";
pub const P_TRACK_HERE: &[u8] = b"trackHere\0";
pub const P_TRACK_RUN: &[u8] = b"trackRun\0";
pub const P_TRACK_FROM: &[u8] = b"trackFrom\0";
pub const P_TRACK_TO: &[u8] = b"trackTo\0";
/// Visible status line written by the Track buttons: a real change, so Flame renders
/// (hidden params / buttons alone do not trigger a render).
pub const P_TRACK_INFO: &[u8] = b"trackInfo\0";
/// The number Flame shows for the clip's first frame: the plugin sees clip times from
/// the clip start, Flame shows its own numbering (frame 1, 1001 ...).
pub const P_TRACK_BASE: &[u8] = b"trackFirstFrame\0";

/// Output modes (P_OUTPUT). 0..2 are the original single-matte views.
#[allow(dead_code)]
pub const OUT_MATTE: i32 = 0;
#[allow(dead_code)]
pub const OUT_SRC_MATTE: i32 = 1;
#[allow(dead_code)]
pub const OUT_OVERLAY: i32 = 2;
pub const OUT_CHANNELS: i32 = 3;
pub const OUT_IDMAP: i32 = 4;
/// Object IDs: object k (1..=ID_MAX) as k / ID_SCALE in R = G = B, background 0, hard
/// (one object per pixel), all objects in A. Decoded by the cwExtract Matchbox, whose
/// ID_SCALE must stay equal (test `objids_scale_matches_cwextract`).
pub const OUT_OBJIDS: i32 = 5;
pub const ID_SCALE: f32 = 50.0;
pub const ID_MAX: usize = 50;

/// Output channel slots = the Output clip's R/G/B/A (Flame: Result RGB +
/// OutMatte socket from A). An OFX node has exactly one output clip; every
/// other clip is an input, so 4 objects is the maximum per node.
pub const SLOTS: [&str; 4] = ["R", "G", "B", "A"];
pub fn slot_sel_name(k: usize) -> Vec<u8> {
    format!("sel{}\0", SLOTS[k]).into_bytes()
}
/// Viewer pick for slot k (canonical coords), used when the slot is "Picked".
pub fn slot_pick_name(k: usize) -> Vec<u8> {
    format!("pick{}\0", SLOTS[k]).into_bytes()
}
/// Push button that arms a viewer pick for slot k.
pub fn slot_pick_btn_name(k: usize) -> Vec<u8> {
    format!("pickBtn{}\0", SLOTS[k]).into_bytes()
}
/// Viewer Clicks values 4..7 = pick for slot R/G/B/A (appended: saved setups keep 0..3).
pub const CLICK_PICK0: i32 = 4;
pub fn slot_idx_name(k: usize) -> Vec<u8> {
    format!("idx{}\0", SLOTS[k]).into_bytes()
}

/// Viewer Clicks value 8: every click adds the object under it to Tracked Objects
/// (stays armed until Pick Objects is pressed again).
pub const CLICK_TRACK: i32 = 8;
pub fn armed_track(ps: OfxParamSetHandle, t: f64) -> bool {
    get_int(ps, P_CLICK_MODE, t) == CLICK_TRACK
}

/// The slot (0..4) whose viewer pick is armed (Pick button pressed, no click yet).
pub fn armed_pick(ps: OfxParamSetHandle, t: f64) -> Option<usize> {
    let m = get_int(ps, P_CLICK_MODE, t);
    (CLICK_PICK0..CLICK_PICK0 + SLOTS.len() as i32).contains(&m).then(|| (m - CLICK_PICK0) as usize)
}

pub fn n_slots() -> usize {
    SLOTS.len()
}

pub fn point_name(i: usize) -> Vec<u8> {
    format!("point{}\0", i + 1).into_bytes()
}
pub fn point_on_name(i: usize) -> Vec<u8> {
    format!("point{}On\0", i + 1).into_bytes()
}
pub fn point_bg_name(i: usize) -> Vec<u8> {
    format!("point{}Bg\0", i + 1).into_bytes()
}

/// Remote with no server anywhere: the node message.
pub const NO_SERVER_MSG: &str = "No Remote server configured: set segment_server in cwtools.conf or the Server URL field";

/// Empty Server URL / Model Folder / ONNX Runtime of an existing node (older
/// setup) get the values in effect, as a new node shows them.
pub fn fill_setup_fields(ps: OfxParamSetHandle) {
    let cfg = cwtools_core::config::get();
    cwtools_core::params::fill_empty_strings(ps, &[
        (P_SERVER, server_url("", &cfg.segment_server)),
        (P_MODEL_DIR, cfg.segment_models.clone()),
        (P_ORT_LIB, cwtools_core::config::shown_ort_lib()),
    ]);
}

/// Server base URL: the Server URL text, or `configured` (cwtools.conf
/// segment_server, "" = none configured) when it is empty.
pub fn server_url(url: &str, configured: &str) -> String {
    let u = url.trim();
    if u.is_empty() { configured.trim().to_string() } else { u.to_string() }
}

const G_MAIN: &[u8] = b"grpMain\0";
const G_OUT: &[u8] = b"grpOutput\0";
const G_OPTIONS: &[u8] = b"grpOptions\0";
const G_LIST: &[u8] = b"grpObjectList\0";
const G_RUNTIME: &[u8] = b"grpRuntime\0";
/// One group (= one panel column) per output slot.
fn slot_group_name(k: usize) -> Vec<u8> {
    format!("grpSlot{}\0", SLOTS[k]).into_bytes()
}
const G_PROMPTS: &[u8] = b"grpPrompts\0";
const G_REMOTE: &[u8] = b"grpRemote\0";
const G_LOCAL: &[u8] = b"grpLocal\0";
const G_OBJECTS: &[u8] = b"grpObjects\0";
const G_VLM: &[u8] = b"grpVlm\0";
const G_TRACK: &[u8] = b"grpTrack\0";
const G_TRACK_OBJ: &[u8] = b"grpTrackObjects\0";
const G_TRACK_INFO: &[u8] = b"grpTrackStatus\0";

pub fn define_params(ps: OfxParamSetHandle) {
    // setup fields show what an empty field resolves to (cwtools.conf); clearing one still falls back at render
    let cfg = cwtools_core::config::get();
    let (shown_server, shown_ort) = (server_url("", &cfg.segment_server), cwtools_core::config::shown_ort_lib());
    // Flame panel: 5-row columns, a Choice = 2 rows, every group starts a column,
    // 6 columns a page (cwtools_core::params::grid); String labels are not shown,
    // so each String has a caption row. Page 1: Main / Output / R / G / B / Matte.
    let g = define(ps, kOfxParamTypeGroup, G_MAIN);
    label(g, b"Main\0");
    caption(ps, b"capText\0", b"Text Prompt\0", G_MAIN);
    string(ps, P_TEXT, b"Text Prompt\0",
        b"What to segment, e.g. 'person' or several concepts 'person, sofa, dog'; each \
          instance becomes an object. Local needs the SAM3 text models, Remote a SAM3 server \
          with API v2. Empty = no mask unless Auto Concepts is on.\0",
        "", G_MAIN);
    choice(ps, P_MODE, b"Device\0",
        b"Remote = SAM3 server (Remote group, or segment_server in cwtools.conf). Local = SAM3 \
          ONNX on this workstation via Flame's onnxruntime (Local group; text prompts need the \
          SAM3 text models in the Model Folder).\0",
        &[b"Remote (SAM3 server)\0", b"Local (ONNX)\0"], 0, G_MAIN);
    button(ps, P_RELEASE, b"Release GPU Memory\0",
        b"Unload the SAM encoder/decoder sessions shared by all cwSegment nodes and free their VRAM. The \
          next Local render reloads them (~3-5 s). Remote mode holds no local model.\0",
        G_MAIN);
    let g = define(ps, kOfxParamTypeGroup, G_OUT);
    label(g, b"Output\0");
    choice(ps, P_OUTPUT, b"Output\0",
        b"Object Channels (default): R/G/B/Matte = the slots below (the plate until an object \
          is found). Mask, Source + Mask, Overlay: all objects as one mask. ID Map: objects in \
          colour, numbered. Object IDs: object n as n x 0.02 (keep unfiltered).\0",
        &[b"Mask\0", b"Source + Mask\0", b"Overlay\0", b"Object Channels\0", b"ID Map\0", b"Object IDs\0"], OUT_CHANNELS, G_OUT);
    choice(ps, P_COLORSPACE, b"Input Colour\0",
        b"Encoding of the Source RGB. SAM expects display-referred pixels: ACEScg, ACEScct and \
          linear Rec.709 go through the ACES 2.0 SDR Rec.709 view (as Flame's View Transform) before \
          upload/inference; Video / sRGB passes as-is.\0",
        &[b"ACEScg (linear)\0", b"Linear Rec.709\0", b"Video / sRGB\0", b"ACEScct (log)\0"], 0, G_OUT);
    boolp(ps, P_INVERT, b"Invert\0", b"Invert the output mask.\0", false, G_OUT);

    // ── R / G / B / Matte: one column each (Object 2 + Pick + Pick Point + Number) ──
    for k in 0..n_slots() {
        let ch = SLOTS[k];
        let g = slot_group_name(k);
        let gl = format!("{}\0", if ch == "A" { "Matte" } else { ch });
        let gd = define(ps, kOfxParamTypeGroup, &g);
        label(gd, gl.as_bytes());
        let out_name = "Output";
        let lbl = format!("{ch} Object\0");
        let hnt = format!(
            "What channel {ch} of {out_name} carries in Output = Object Channels: one object (#), \
             all objects of one concept (concept number as typed), every object, nothing, or the \
             object picked in the viewer.\0"
        );
        choice(ps, &slot_sel_name(k), lbl.as_bytes(), hnt.as_bytes(),
            &[b"Off\0", b"Object #\0", b"Concept #\0", b"All Objects\0", b"Picked\0"], 2, &g);
        let name = if ch == "A" { "Matte" } else { ch };
        let lbl = format!("Pick {name}\0");
        let hnt = format!(
            "Then click an object in the viewer: it goes to {name} (the slot becomes Picked, Output \
             becomes Object Channels). The object under the click point is used on every frame.\0"
        );
        button(ps, &slot_pick_btn_name(k), lbl.as_bytes(), hnt.as_bytes(), &g);
        let lbl = format!("{ch} Pick Point\0");
        xy(ps, &slot_pick_name(k), lbl.as_bytes(), b"Where the object for this slot was picked (set by the Pick button + a viewer click).\0", 0.5, 0.5, &g);
        let lbl = format!("{ch} Number\0");
        int(ps, &slot_idx_name(k), lbl.as_bytes(),
            b"Used when the slot is Object # (this object number, #1 = first in the list) or Concept # \
              (every object of this concept, 1 = first concept typed). Ignored for Off / All / Picked.\0",
            k as i32 + 1, 1, 64, &g);
    }

    // ── Page 2: Track / Objects to Track / Track Status / Objects / Options / Object List ──
    let g = define(ps, kOfxParamTypeGroup, G_TRACK);
    label(g, b"Track\0");
    choice(ps, P_TRACK, b"Track\0",
        b"Follow the picked objects from the Anchor Frame (SAM3 video memory, Local). Frames are \
          tracked as they render in order from the anchor; results stay cached. Off = detect on \
          every frame.\0",
        &[b"Off\0", b"Forward\0", b"Backward\0", b"Both\0"], 0, G_TRACK);
    button(ps, P_TRACK_PICK, b"Pick Objects\0",
        b"Then click objects in the viewer (shown numbered): each click adds that object to Tracked \
          Objects. The first pick sets the Anchor Frame. Press again to stop picking.\0",
        G_TRACK);
    button(ps, P_TRACK_CLEAR, b"Clear Picks\0", b"Empty Tracked Objects and forget the tracked frames.\0", G_TRACK);
    button(ps, P_TRACK_RESET, b"Reset Track\0",
        b"Forget the tracked frames of this node and track again from the Anchor Frame.\0", G_TRACK);
    let g = define(ps, kOfxParamTypeGroup, G_TRACK_OBJ);
    label(g, b"Objects to Track\0");
    caption(ps, b"capTrackList\0", b"Tracked Objects\0", G_TRACK_OBJ);
    string(ps, P_TRACK_LIST, b"Tracked Objects\0",
        b"Object numbers on the anchor frame (as in the ID Map), e.g. 2, 5. Filled by Pick Objects, \
          or type them. Tracked object 1, 2 ... = R, G ... with Object Channels.\0",
        "", G_TRACK_OBJ);
    int(ps, P_TRACK_ANCHOR, b"Anchor Frame\0",
        b"The frame the objects were picked on (set by Pick Objects / Anchor Here).\0", 0, -1_000_000, 10_000_000, G_TRACK_OBJ);
    button(ps, P_TRACK_HERE, b"Anchor Here\0",
        b"Make the current frame the anchor frame (for typed object numbers).\0", G_TRACK_OBJ);
    int(ps, P_TRACK_BASE, b"Batch Start\0",
        b"Batch's start frame (the number Flame shows for the clip's first frame, often 1001). Frame \
          numbers in this node (Track From / To, messages) use it. Changing it resets Track From / To to \
          the whole clip.\0", 1001, -1_000_000, 10_000_000, G_TRACK_OBJ);
    let g = define(ps, kOfxParamTypeGroup, G_TRACK_INFO);
    label(g, b"Track Range\0");
    button(ps, P_TRACK_RUN, b"Track Range\0",
        b"Track from the anchor frame to Track To (Forward / Both) and back to Track From (Backward / Both) \
          in the background. Press again to stop; once more continues.\0",
        G_TRACK_INFO);
    int(ps, P_TRACK_FROM, b"Track From\0",
        b"First frame Track Range tracks (Batch numbers). Filled with the clip start; 0 and 0 = whole clip.\0",
        0, -1_000_000, 10_000_000, G_TRACK_INFO);
    int(ps, P_TRACK_TO, b"Track To\0",
        b"Last frame Track Range tracks (Batch numbers). Filled with the clip end; 0 and 0 = whole clip.\0",
        0, -1_000_000, 10_000_000, G_TRACK_INFO);
    caption(ps, b"capTrackInfo\0", b"Status\0", G_TRACK_INFO);
    string(ps, P_TRACK_INFO, b"Status\0",
        b"Last Track action of this node (the node message has the result).\0", "", G_TRACK_INFO);

    let g = define(ps, kOfxParamTypeGroup, G_OBJECTS);
    label(g, b"Objects\0");
    choice(ps, P_ORDER, b"Number By\0",
        b"How objects are numbered (#1, #2 ...). Concept + Score = concept order as typed, then \
          score. No tracking: numbers can change between frames when objects move or scores \
          swap.\0",
        &[b"Concept + Score\0", b"Left to Right\0", b"Top to Bottom\0", b"Size (largest)\0", b"Score\0"],
        0, G_OBJECTS);
    int(ps, P_MAX_OBJ, b"Max Objects\0",
        b"Keep at most this many objects (highest scores win). Each object is one more \
          mask to compute (Remote: to download).\0",
        32, 1, 128, G_OBJECTS);
    boolp(ps, P_ID_LABELS, b"ID Map Numbers\0", b"Stamp each object's number into the ID Map view.\0", true, G_OBJECTS);
    boolp(ps, P_LIST_MSG, b"List in Message\0",
        b"Show the object list (number, concept, score) as the node message after each \
          multi-object render.\0",
        true, G_OBJECTS);
    let g = define(ps, kOfxParamTypeGroup, G_OPTIONS);
    label(g, b"Options\0");
    boolp(ps, P_SOFT, b"Soft Edges\0",
        b"ON = soft probability mask (anti-aliased edge). OFF = binary 0/1 mask \
          (thresholded at 0.5 after upscaling to full resolution).\0",
        false, G_OPTIONS);
    boolp(ps, P_BEST_ONLY, b"Best Instance Only\0",
        b"Text prompts find ALL matching instances (union). ON = keep only the \
          highest-scoring one.\0",
        false, G_OPTIONS);
    let p = define(ps, kOfxParamTypeDouble, P_SCORE);
    label(p, b"Detect Thresh\0");
    hint(p, b"Text-prompt detection score threshold (default 0.4). Lower finds more \
               (weaker) instances.\0");
    prop_set_double(p, kOfxParamPropDefault, 0, 0.4);
    prop_set_double(p, kOfxParamPropMin, 0, 0.0);
    prop_set_double(p, kOfxParamPropMax, 0, 1.0);
    prop_set_double(p, kOfxParamPropDisplayMin, 0, 0.0);
    prop_set_double(p, kOfxParamPropDisplayMax, 0, 1.0);
    prop_set_int(p, kOfxParamPropDigits, 0, 2);
    prop_set_double(p, kOfxParamPropIncrement, 0, cwtools_core::params::increment_for(2));
    parent(p, G_OPTIONS);
    boolp(ps, P_CACHE, b"Cache\0",
        b"Keep masks in memory keyed on (downscaled pixels + prompts). Re-rendering an \
          unchanged frame is then instant. Local mode also caches the image embedding per \
          frame.\0",
        true, G_OPTIONS);
    button(ps, P_CLEAR_CACHE, b"Clear Cache\0", b"Forget all cached masks/embeddings.\0", G_OPTIONS);
    let g = define(ps, kOfxParamTypeGroup, G_LIST);
    label(g, b"Object List\0");
    button(ps, P_REFRESH_LIST, b"Show Object List\0",
        b"Copy the last rendered object list of this node into the Object List field.\0", G_LIST);
    cwtools_core::params::string_multiline(ps, P_OBJ_LIST, b"Object List\0",
        b"Filled by Show Object List: one line per object (number, concept, score, box). \
          Display only.\0",
        "", G_LIST);
    let g = define(ps, kOfxParamTypeGroup, G_REMOTE);
    label(g, b"Remote\0");
    caption(ps, b"capServer\0", b"Server URL\0", G_REMOTE);
    string(ps, P_SERVER, b"Server URL\0", b"SAM3 server base URL, e.g. http://host:8199. Empty = segment_server from cwtools.conf.\0", &shown_server, G_REMOTE);
    let p = define(ps, kOfxParamTypeInteger, P_SEND_SIZE);
    label(p, b"Send Size\0");
    hint(p, b"Longest edge (pixels) of the JPEG uploaded to the server. The mask comes back \
               at this size and is upscaled. Smaller uploads faster on slow \
               networks; 1024-1280 is a good range.\0");
    prop_set_int(p, kOfxParamPropDefault, 0, 1280);
    prop_set_int(p, kOfxParamPropMin, 0, 256);
    prop_set_int(p, kOfxParamPropMax, 0, 4096);
    prop_set_int(p, kOfxParamPropDisplayMin, 0, 256);
    prop_set_int(p, kOfxParamPropDisplayMax, 0, 2048);
    parent(p, G_REMOTE);
    let p = define(ps, kOfxParamTypeDouble, P_TIMEOUT);
    label(p, b"Timeout (s)\0");
    hint(p, b"Give up on the server after this many seconds (max 15) and pass the plate \
               through with an empty mask + a node message.\0");
    prop_set_double(p, kOfxParamPropDefault, 0, 12.0);
    prop_set_double(p, kOfxParamPropMin, 0, 1.0);
    prop_set_double(p, kOfxParamPropMax, 0, 15.0);
    prop_set_double(p, kOfxParamPropDisplayMin, 0, 1.0);
    prop_set_double(p, kOfxParamPropDisplayMax, 0, 15.0);
    parent(p, G_REMOTE);
    button(ps, P_TEST_SERVER, b"Test Server\0",
        b"Ping the server's health endpoint and report the result as a node message / in \
          /tmp/cwSegment.log.\0",
        G_REMOTE);
    let g = define(ps, kOfxParamTypeGroup, G_LOCAL);
    label(g, b"Local\0");
    boolp(ps, P_USE_GPU, b"Use GPU (CUDA)\0",
        b"Run local inference with the CUDA execution provider (falls back to CPU if it \
          cannot be created).\0",
        true, G_LOCAL);
    boolp(ps, P_FREE_VRAM, b"Free VRAM/Frame\0",
        b"Return the SAM encoder/decoder working memory to the GPU after every run (keeps the weights \
          loaded). Leaves VRAM for Flame at a small per-frame cost. Off = keep the arena (fastest).\0",
        true, G_LOCAL);
    caption(ps, b"capModelDir\0", b"Model Folder\0", G_LOCAL);
    string(ps, P_MODEL_DIR, b"Model Folder\0",
        b"Folder with the SAM encoder/decoder ONNX files (see README). Empty = the installed models \
          folder (cwtools.conf).\0",
        &cfg.segment_models, G_LOCAL);
    cwtools_core::version::define_param(ps, G_LOCAL);
    let g = define(ps, kOfxParamTypeGroup, G_RUNTIME);
    label(g, b"ONNX Runtime\0");
    caption(ps, b"capOrtLib\0", b"ONNX Runtime Library\0", G_RUNTIME);
    string(ps, P_ORT_LIB, b"ONNX Runtime\0",
        b"Path to libonnxruntime (1.22). Empty = ort_lib from cwtools.conf, else auto (Flame's own \
          copy).\0",
        &shown_ort, G_RUNTIME);

    // ── Page 3: Remote / Local / ONNX Runtime (above) + Auto (VLM) ──────
    let g = define(ps, kOfxParamTypeGroup, G_VLM);
    label(g, b"Auto (VLM)\0");
    choice(ps, P_VLM_AUTO, b"Auto Concepts\0",
        b"Ask a vision-language model which object classes are in the frame and segment those. \
          The FRAME IS SENT to the VLM URL (local by default). Cached per frame.\0",
        &[b"Off\0", b"Replace Text\0", b"Add to Text\0"], 0, G_VLM);
    caption(ps, b"capVlmUrl\0", b"VLM URL\0", G_VLM);
    string(ps, P_VLM_URL, b"VLM URL\0",
        b"OpenAI-compatible base URL (/v1). Default = Ollama on this machine. Frames go to this \
          address - only point it at machines you trust with the footage.\0",
        crate::vlm::DEFAULT_URL, G_VLM);
    button(ps, P_VLM_COPY, b"Copy to Text\0",
        b"Write this node's last VLM concepts into Text Prompt and set Auto Concepts to Off \
          (keeps the list stable and stops sending frames).\0",
        G_VLM);
    caption(ps, b"capVlmModel\0", b"VLM Model\0", G_VLM);
    string(ps, P_VLM_MODEL, b"VLM Model\0",
        b"Model name as the endpoint knows it (Ollama: qwen3-vl:4b-instruct). Not installed there, or 'auto' = \
          the smallest installed qwen3-vl instruct model (else another vision model); the node message says which.\0",
        crate::vlm::DEFAULT_MODEL, G_VLM);
    caption(ps, b"capVlmKey\0", b"VLM API Key\0", G_VLM);
    string(ps, P_VLM_KEY, b"VLM API Key\0", b"Optional bearer token (empty = none).\0", "", G_VLM);
    boolp(ps, P_VLM_RESTRICT, b"Skip Set Dressing\0",
        b"ON = ignore furniture, fixtures and backdrops (people, vehicles, animals, props only). \
          OFF = list those too.\0",
        true, G_VLM);
    choice(ps, P_VLM_DETAIL, b"VLM Detail\0",
        b"How many classes the VLM lists: Minimal (1-4 heroes), Standard (8), Inclusive (16), \
          Exhaustive (24; SAM3 takes the first 16).\0",
        &[b"Minimal\0", b"Standard\0", b"Inclusive\0", b"Exhaustive\0"], 1, G_VLM);
    boolp(ps, P_VLM_UNLOAD, b"VLM Unload After\0",
        b"Ollama: unload the model right after each request (keep_alive 0), so it does not hold ~5 GB of VRAM \
          next to Flame. Every new frame then reloads it (~4 s warm). Off = Ollama keeps it ~5 min.\0",
        true, G_VLM);
    let p = define(ps, kOfxParamTypeDouble, P_VLM_TIMEOUT);
    label(p, b"VLM Timeout\0");
    hint(p, b"Seconds to wait for the VLM (max 60). The render waits this long on a new frame. Ollama: ~6 s \
               with the model file cached, much longer on a cold first load.\0");
    prop_set_double(p, kOfxParamPropDefault, 0, 30.0);
    prop_set_double(p, kOfxParamPropMin, 0, 1.0);
    prop_set_double(p, kOfxParamPropMax, 0, 60.0);
    prop_set_double(p, kOfxParamPropDisplayMin, 0, 1.0);
    prop_set_double(p, kOfxParamPropDisplayMax, 0, 60.0);
    parent(p, G_VLM);

    // ── Prompts (hidden while PROMPT_UI is false: secret params take no row) ──
    let g = define(ps, kOfxParamTypeGroup, G_PROMPTS);
    label(g, b"Prompts\0");
    boolp(ps, P_USE_MATTE, b"Use Prompt Mask\0",
        b"Derive a box + points from the mask in the Source Alpha socket (white = object). \
          A GMask or paint scribble works.\0",
        false, G_PROMPTS);
    choice(ps, P_MATTE_AS, b"Mask Gives\0",
        b"What the prompt mask becomes: its bounding box + interior points, only the box, \
          or only interior points.\0",
        &[b"Box + Points\0", b"Box Only\0", b"Points Only\0"], 0, G_PROMPTS);
    choice(ps, P_CLICK_MODE, b"Viewer Clicks\0",
        b"Viewer clicks: add a foreground / background point (next free Point slot), drag a \
          box, or pick the object under the click for R / G / B / Matte (set by the Pick \
          buttons, back to Off after the pick). Off = clicks ignored.\0",
        &[b"Off\0", b"Add FG Point\0", b"Add BG Point\0", b"Drag Box\0", b"Pick R\0", b"Pick G\0", b"Pick B\0", b"Pick Matte\0",
          b"Pick Track\0"],
        if PROMPT_UI { 1 } else { 0 }, G_PROMPTS);
    button(ps, P_CLEAR_POINTS, b"Clear Points\0", b"Disable all points and the box.\0", G_PROMPTS);
    boolp(ps, P_USE_BOX, b"Use Box\0", b"Use the box (two corners) as a prompt.\0", false, G_PROMPTS);
    xy(ps, P_BOX_A, b"Box Corner A\0", b"One corner of the prompt box.\0", 0.3, 0.3, G_PROMPTS);
    xy(ps, P_BOX_B, b"Box Corner B\0", b"Opposite corner of the prompt box.\0", 0.7, 0.7, G_PROMPTS);
    for i in 0..MAX_POINTS {
        let lbl_on = format!("Point {} On\0", i + 1);
        let lbl = format!("Point {}\0", i + 1);
        let lbl_bg = format!("Point {} BG\0", i + 1);
        boolp(ps, &point_on_name(i), lbl_on.as_bytes(), b"Enable this point prompt.\0", false, G_PROMPTS);
        xy(ps, &point_name(i), lbl.as_bytes(), b"Point position.\0", 0.5, 0.5, G_PROMPTS);
        boolp(ps, &point_bg_name(i), lbl_bg.as_bytes(),
            b"OFF = foreground point (object). ON = background point (exclude).\0", false, G_PROMPTS);
    }
    // Pick points are set by Pick + a viewer click (shown there as a ring): no panel rows,
    // so each slot column is Object + Pick + Number (4 of 5 rows).
    for k in 0..n_slots() {
        cwtools_core::params::set_secret(ps, &slot_pick_name(k));
    }
    // Flame's frame numbers are not the plugin's clip times: the anchor is set by picking, not typed.
    cwtools_core::params::set_secret(ps, P_TRACK_ANCHOR);
    if !PROMPT_UI {
        let mut hide: Vec<Vec<u8>> = [G_PROMPTS, P_USE_MATTE, P_MATTE_AS, P_CLICK_MODE, P_CLEAR_POINTS, P_USE_BOX, P_BOX_A, P_BOX_B]
            .iter().map(|n| n.to_vec()).collect();
        for i in 0..MAX_POINTS {
            hide.extend([point_on_name(i), point_name(i), point_bg_name(i)]);
        }
        for n in &hide {
            cwtools_core::params::set_secret(ps, n);
        }
    }
}

// ---------------------------------------------------------------------------
// Readback
// ---------------------------------------------------------------------------
pub use cwtools_core::params::{get_bool, get_double, get_int, get_string, get_xy, set_bool, set_int, set_string, set_xy};

#[derive(Clone, Debug)]
pub struct UserPoint {
    /// Canonical coordinates (OFX: origin bottom-left, full-res pixels).
    pub x: f64,
    pub y: f64,
    pub fg: bool,
}

#[derive(Clone, Debug)]
pub struct Values {
    pub mode: i32,
    pub text: String,
    pub use_matte: bool,
    pub matte_as: i32,
    pub invert: bool,
    pub output: i32,
    pub colorspace: i32,
    pub cache: bool,
    pub soft: bool,
    pub best_only: bool,
    pub score_threshold: f64,
    pub box_canon: Option<(f64, f64, f64, f64)>,
    pub points: Vec<UserPoint>,
    pub server: String,
    pub send_size: i32,
    pub timeout: f64,
    pub model_dir: String,
    pub ort_lib: String,
    pub use_gpu: bool,
    pub free_vram: bool,
    // multi-object
    pub order: i32,
    pub sels: Vec<crate::objects::Sel>,
    /// Viewer pick point (canonical) per slot, Some only for Picked slots.
    pub picks: Vec<Option<(f64, f64)>>,
    pub max_objects: i32,
    pub id_labels: bool,
    pub list_in_message: bool,
    pub vlm_auto: i32,
    pub vlm: crate::vlm::VlmOpts,
    pub track: i32,
    pub track_list: Vec<usize>,
    pub track_anchor: i64,
    pub track_from: i64,
    pub track_to: i64,
    pub track_base: i64,
    /// Pick Objects armed (render.rs): the ID Map with every detection, picked ones highlighted.
    pub picking: bool,
}

#[cfg(test)]
impl Values {
    /// Local-mode values for host-free tests (Remote / VLM fields empty).
    pub(crate) fn for_test(model_dir: &str, ort_lib: &str, use_gpu: bool) -> Values {
        Values {
            mode: 1, text: String::new(), use_matte: false, matte_as: 0, invert: false, output: 0, colorspace: 0,
            cache: false, soft: true, best_only: false, score_threshold: 0.4, box_canon: None, points: Vec::new(),
            server: String::new(), send_size: 1008, timeout: 30.0, model_dir: model_dir.into(), ort_lib: ort_lib.into(),
            use_gpu, free_vram: true, order: 0, sels: Vec::new(), picks: Vec::new(), max_objects: 32, id_labels: false,
            list_in_message: false, vlm_auto: 0,
            vlm: crate::vlm::VlmOpts { url: String::new(), model: String::new(), api_key: String::new(), detail: 0,
                                       restrict: false, timeout_s: 1.0, unload: true },
            track: 0, track_list: Vec::new(), track_anchor: 0, track_from: 0, track_to: 0, track_base: 1001, picking: false,
        }
    }
}

pub fn read_values(ps: OfxParamSetHandle, t: f64) -> Values {
    let box_canon = if get_bool(ps, P_USE_BOX, t) {
        let (ax, ay) = get_xy(ps, P_BOX_A, t);
        let (bx, by) = get_xy(ps, P_BOX_B, t);
        Some((ax.min(bx), ay.min(by), ax.max(bx), ay.max(by)))
    } else {
        None
    };
    let mut points = Vec::new();
    for i in 0..MAX_POINTS {
        if get_bool(ps, &point_on_name(i), t) {
            let (x, y) = get_xy(ps, &point_name(i), t);
            points.push(UserPoint { x, y, fg: !get_bool(ps, &point_bg_name(i), t) });
        }
    }
    let cfg = cwtools_core::config::get();
    let server = server_url(&get_string(ps, P_SERVER), &cfg.segment_server);
    let sels: Vec<crate::objects::Sel> = (0..n_slots())
        .map(|k| crate::objects::Sel { kind: get_int(ps, &slot_sel_name(k), t), index: get_int(ps, &slot_idx_name(k), t) })
        .collect();
    let picks = (0..n_slots())
        .map(|k| (sels[k].kind == crate::objects::SEL_PICK).then(|| get_xy(ps, &slot_pick_name(k), t)))
        .collect();
    let mut vlm_url = get_string(ps, P_VLM_URL);
    if vlm_url.is_empty() {
        vlm_url = crate::vlm::DEFAULT_URL.into();
    }
    let mut vlm_model = get_string(ps, P_VLM_MODEL);
    if vlm_model.is_empty() {
        vlm_model = crate::vlm::DEFAULT_MODEL.into();
    }
    let model_dir = cwtools_core::config::folder_or_installed(&get_string(ps, P_MODEL_DIR), &cfg.segment_models);
    let ort_lib = get_string(ps, P_ORT_LIB);
    Values {
        mode: get_int(ps, P_MODE, t),
        text: get_string(ps, P_TEXT),
        use_matte: get_bool(ps, P_USE_MATTE, t),
        matte_as: get_int(ps, P_MATTE_AS, t),
        invert: get_bool(ps, P_INVERT, t),
        output: get_int(ps, P_OUTPUT, t),
        colorspace: get_int(ps, P_COLORSPACE, t),
        cache: get_bool(ps, P_CACHE, t),
        soft: get_bool(ps, P_SOFT, t),
        best_only: get_bool(ps, P_BEST_ONLY, t),
        score_threshold: get_double(ps, P_SCORE, t).clamp(0.0, 1.0),
        box_canon,
        points,
        server,
        send_size: get_int(ps, P_SEND_SIZE, t).clamp(256, 4096),
        timeout: get_double(ps, P_TIMEOUT, t).clamp(1.0, 15.0),
        model_dir,
        ort_lib,
        use_gpu: get_bool(ps, P_USE_GPU, t),
        free_vram: get_bool(ps, P_FREE_VRAM, t),
        order: get_int(ps, P_ORDER, t),
        sels,
        picks,
        max_objects: get_int(ps, P_MAX_OBJ, t).clamp(1, 128),
        id_labels: get_bool(ps, P_ID_LABELS, t),
        list_in_message: get_bool(ps, P_LIST_MSG, t),
        vlm_auto: get_int(ps, P_VLM_AUTO, t),
        vlm: crate::vlm::VlmOpts {
            url: vlm_url,
            model: vlm_model,
            api_key: get_string(ps, P_VLM_KEY),
            detail: get_int(ps, P_VLM_DETAIL, t),
            restrict: get_bool(ps, P_VLM_RESTRICT, t),
            timeout_s: get_double(ps, P_VLM_TIMEOUT, t).clamp(1.0, crate::vlm::MAX_TIMEOUT_S),
            unload: get_bool(ps, P_VLM_UNLOAD, t),
        },
        track: get_int(ps, P_TRACK, t),
        track_list: crate::track::parse_list(&get_string(ps, P_TRACK_LIST)),
        track_anchor: get_int(ps, P_TRACK_ANCHOR, t) as i64,
        track_from: get_int(ps, P_TRACK_FROM, t) as i64,
        track_to: get_int(ps, P_TRACK_TO, t) as i64,
        track_base: get_int(ps, P_TRACK_BASE, t) as i64,
        picking: false,
    }
}

#[cfg(test)]
mod tests {
    /// cwExtract decodes Object IDs with its own copy of the scale.
    #[test]
    fn objids_scale_matches_cwextract() {
        let glsl = concat!(env!("CARGO_MANIFEST_DIR"), "/../../flame_inference/matchbox/cwExtract/cwExtract.1.glsl");
        let src = std::fs::read_to_string(glsl).expect("cwExtract shader");
        let want = format!("const float ID_SCALE = {:.1};", super::ID_SCALE);
        assert!(src.contains(&want), "cwExtract.1.glsl must define `{want}`");
        assert!(super::ID_MAX as f32 <= super::ID_SCALE && super::ID_MAX <= u8::MAX as usize);
    }

    /// Flame truncates OFX strings at 255 chars: scan every byte-string literal.
    #[test]
    fn all_literals_fit_flame_limit() {
        cwtools_core::params::check_byte_literals(include_str!("params.rs")).unwrap();
    }
    #[test]
    fn server_from_params_or_config() {
        use super::*;
        assert_eq!(server_url("", ""), "");
        assert_eq!(server_url(" ", "http://cfg:8199"), "http://cfg:8199");
        assert_eq!(server_url("http://h:1", "http://cfg:8199"), "http://h:1");
    }
}
