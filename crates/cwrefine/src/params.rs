//! cwRefine parameters (cwtools-core table DSL). Groups, not Pages (Flame);
//! every continuous knob is an animatable Double; Double labels <= 15 chars,
//! all strings <= 255 bytes (guard tests below).
//!
//! Defaults = the production hair recipe (memory vitmatte_hair_core):
//! kernel_inward 4 / kernel_outward 30 (ellipse), seed clamp 250, padding 128,
//! max crop 2048, mid-bias off, no pre-dilate.

use crate::recipe::TrimapParams;
use cwtools_core::color::CS_ACESCG;
use cwtools_core::ofx_api::OfxParamSetHandle;
use cwtools_core::ort_rt::ConvSearch;
use cwtools_core::params::{get_double, get_int, get_string, n, Kind, ParamDef};

/// Model Folder default: empty = `refine_models` from cwtools.conf.
pub const DEFAULT_MODEL_DIR: &str = "";

/// The Model Folder in effect: the param if set (and present here), else cwtools.conf `refine_models`.
pub fn model_dir(param: &str) -> String {
    cwtools_core::config::folder_or_installed(param, &cwtools_core::config::get().refine_models)
}

/// One Model menu entry: a core ONNX with the cwRefine contract
/// (`pixel_values` [1,4,h,w] -> `alphas` [1,1,h,w], h/w multiples of 32,
/// RGB (x/255 - 0.5)/0.5 + trimap/255), file name in the Model Folder.
pub struct ModelEntry {
    pub label: &'static str,
    pub file: &'static str,
    /// cuDNN conv algorithm search for the local CUDA session.
    pub conv_search: ConvSearch,
}

/// Model choice -> file in the Model Folder (index = Model choice value).
/// MEMatte (ViT-B, MIT): token-routed global attention, ~1/3 of ViTMatte's
/// VRAM at a 2048 crop (4.3 GB vs ~12.5 GB), same pre/post.
pub const MODELS: [ModelEntry; 4] = [
    ModelEntry { label: "ViTMatte", file: "vitmatte_core.onnx", conv_search: ConvSearch::Heuristic },
    // Default (implicit GEMM), not Heuristic: on an RTX 3090 the heuristic picks an
    // algorithm for /bb/residual/conv2 that asks for a 78 GB workspace at a 2048 crop
    // (the cap is ignored outside Exhaustive, and Exhaustive fails the same way).
    // Default: 3.7 GB peak, 0.77 s, max 5.2e-6 vs CPU.
    ModelEntry { label: "MEMatte", file: "mematte_vitb_core.onnx", conv_search: ConvSearch::Default },
    // Internal test models (feature `test-models`): MEMatte fine-tunes, same graph, so the
    // same conv search.
    ModelEntry { label: "MEMatte Synth", file: "mematte_synth_s1_core.onnx", conv_search: ConvSearch::Default },
    ModelEntry { label: "MEMatte Synth 2", file: "mematte_synth_s1b_core.onnx", conv_search: ConvSearch::Default },
];
/// Choice order. Custom Path stays at 2 (saved setups); new models are appended after it.
/// The Synth test fine-tunes (not validated for release) exist only with the
/// `test-models` feature; public builds end the menu at Custom Path.
#[cfg(feature = "test-models")]
pub const MODEL_LABELS: [&str; 5] = [MODELS[0].label, MODELS[1].label, "Custom Path", MODELS[2].label, MODELS[3].label];
#[cfg(not(feature = "test-models"))]
pub const MODEL_LABELS: [&str; 3] = [MODELS[0].label, MODELS[1].label, "Custom Path"];
pub const MODEL_VITMATTE: i32 = 0;
pub const MODEL_MEMATTE: i32 = 1;
pub const MODEL_CUSTOM: i32 = 2;
pub const MODEL_SYNTH: i32 = 3;
pub const MODEL_SYNTH2: i32 = 4;

/// Model choice value -> its MODELS entry (None = Custom Path or out of range).
pub fn model_entry(choice: i32) -> Option<&'static ModelEntry> {
    match choice {
        MODEL_VITMATTE => Some(&MODELS[0]),
        MODEL_MEMATTE => Some(&MODELS[1]),
        #[cfg(feature = "test-models")]
        MODEL_SYNTH => Some(&MODELS[2]),
        #[cfg(feature = "test-models")]
        MODEL_SYNTH2 => Some(&MODELS[3]),
        _ => None,
    }
}
/// New nodes start on MEMatte: same inputs/outputs as ViTMatte at about a third of the
/// VRAM, full UHD in one pass next to Flame (quality a touch above ViTMatte at 1536). The parity recipe (`Values::production`) stays ViTMatte.
pub const MODEL_DEFAULT: i32 = MODEL_MEMATTE;

pub const G_OUTPUT: &str = "grpOutput";
pub const G_MAIN: &str = "grpMain";
pub const G_TRIMAP: &str = "grpTrimap";
pub const G_MASK: &str = "grpMask";
pub const G_SOURCES: &str = "grpSources";
pub const G_LOCAL: &str = "grpLocal";
pub const G_FILES: &str = "grpFiles";
pub const G_RUNTIME: &str = "grpRuntime";
pub const G_CROP: &str = "grpCrop";
pub const G_AUX: &str = "grpOutMatte";
pub const G_SPLINE: &str = "grpSpline";
pub const G_POINTS: &str = "grpSplinePoints";
pub const G_REMOTE: &str = "grpRemote";

pub const P_VIEW: &str = "view";
pub const P_COLORSPACE: &str = "inputColorSpace";
pub const P_USE_TRIMAP: &str = "useTrimapClip";
pub const P_BAND_FROM: &str = "bandFrom";
pub const P_MID_POINT: &str = "midPoint";
pub const P_ERODE_IN: &str = "erodeIn";
pub const P_DILATE_OUT: &str = "dilateOut";
pub const P_SEED_CLAMP: &str = "seedClamp";
pub const P_MASK_BLACK: &str = "maskBlackClip";
pub const P_PRE_DILATE: &str = "preDilate";
pub const P_MID_BIAS: &str = "midBias";
pub const P_INVERT_SEED: &str = "invertSeed";
pub const P_PADDING: &str = "padding";
pub const P_MAX_CROP: &str = "maxCrop";
pub const P_MAX_CROP_CUSTOM: &str = "maxCropCustom";
pub const P_DEVICE: &str = "device";
pub const P_USE_GPU: &str = "useGpu";
pub const P_GPU_MATH: &str = "gpuMath";
pub const P_FREE_VRAM: &str = "freeVramPerFrame";
pub const P_CACHE: &str = "cacheResults";
pub const P_MODEL: &str = "model";
pub const P_MODEL_DIR: &str = "modelFolder";
/// Model = Custom Path: full path (id kept from the single Model Path field).
pub const P_MODEL_PATH: &str = "modelPath";
pub const P_ORT_LIB: &str = "ortLibrary";
pub const P_SERVER: &str = "serverUrl";
pub const P_RELEASE: &str = "releaseGpu";
pub const P_TIMEOUT: &str = "timeoutSec";
pub const P_TEST_SERVER: &str = "testServer";
// Result RGB mode + channel pickers (ids auxR/G/B kept from the old OutMatte
// clip, removed: every OFX clip except Output is an INPUT in Flame).
pub const P_MATTE_OUT: &str = "matteOut";
pub const MATTE_OUT_MATTE: i32 = 0;
pub const MATTE_OUT_INVERTED: i32 = 1;
pub const P_AUX_R: &str = "auxR";
pub const P_AUX_G: &str = "auxG";
pub const P_AUX_B: &str = "auxB";
// Spline trimap
pub const P_SPLINE_ON: &str = "splineOn";
pub const P_SPLINE_EDIT: &str = "splineEdit";
pub const P_SPLINE_SHOW: &str = "splineShow";
pub const P_BAND_GRADIENT: &str = "bandGradient";
pub const P_INNER_LEVEL: &str = "innerLevel";
pub const P_OUTER_LEVEL: &str = "outerLevel";
pub const P_GRAD_BIAS: &str = "gradientBias";
pub const P_OUTER2_ON: &str = "outer2On";
pub const P_OUTER2_WIDTH: &str = "outer2Width";
pub const P_OUTER2_LEVEL: &str = "outer2Level";
pub const P_CLEAR_POINTS: &str = "clearPoints";

/// Control-point slot param names (`i` = 0-based slot): cp<i+1>On / U / V / Out / In / Width.
pub fn cp_name(i: usize, field: &str) -> String {
    format!("cp{}{}", i + 1, field)
}

/// Result RGB channel sources (Channel Pickers mode). 0 = that channel of the plate.
pub const AUX_OFF: i32 = 0;
pub const AUX_TRIMAP: i32 = 1;
pub const AUX_SEED: i32 = 2;
pub const AUX_REFINED: i32 = 3;
pub const AUX_CROP: i32 = 4;
pub const AUX_DIFF: i32 = 5;
pub const AUX_CLASS: i32 = 6;
pub const AUX_OPTS: &[&str] = &["Plate", "Trimap", "Input Mask", "Refined", "Crop Region", "Difference", "Band Class"];
/// Name of the always-on second output clip.

/// Result RGB (Flame's Result socket); Matte Out (OutMatte) is set separately
/// and always carries the refined matte (Edit Splines preview aside).
pub const VIEW_PLATE: i32 = 0;
pub const VIEW_MATTE: i32 = 1;
pub const VIEW_PREMULT: i32 = 2;
pub const VIEW_TRIMAP: i32 = 3;
pub const VIEW_CROP: i32 = 4;
pub const VIEW_DIFF: i32 = 5;
/// R / G / B each carry what the Result Channels pickers choose.
pub const VIEW_CHANNELS: i32 = 6;
/// Internal view while Edit Splines is on: plate + band tint, no inference.
pub const VIEW_EDIT: i32 = 100;


pub const MAX_CROP_CHOICES: [usize; 3] = [1024, 1536, 2048];

const fn dbl(def: f64, min: f64, max: f64, dmin: f64, dmax: f64, digits: i32) -> Kind {
    Kind::Double { def, min, max, dmin, dmax, digits }
}

const HINT_ON: &str = "Use this control point. Points are edge-relative: they follow the subject over the shot.";
const HINT_U: &str = "Anchor across the input mask's bounding box (0 = left, 1 = right). Each frame it is snapped to the nearest point of the matte edge, so it follows the subject. Animate it where the shape changes a lot.";
const HINT_V: &str = "Anchor up the input mask's bounding box (0 = bottom, 1 = top). See U.";
const HINT_OUT: &str = "Local push of the OUTER line (px, full res) at this point: + = wider unknown band (e.g. 80 for flyaways), - = tighter. Falls off along the edge over Width.";
const HINT_IN: &str = "Local push of the INNER line (px) into the subject: + = more unknown inside (soft interior edge), - = core reaches closer to the edge.";
const HINT_W: &str = "How far along the edge (px of arc, each side) the offsets reach, with a smooth falloff.";

pub const PARAMS: &[ParamDef] = &[
    // ── Page 1 ──────────────────────────────────────────────────────────
    // Flame: 5-row columns, a Choice = 2 rows, every group starts a column,
    // 6 columns a page (see cwtools_core::params::grid, checked in the tests).
    ParamDef { name: G_MAIN, label: "Main", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_MODEL,
        label: "Model",
        hint: "ViTMatte: hair-detail model, ~12.5 GB VRAM at a 2048 crop. MEMatte (default): ViT-B with routed attention, \
               ~4.3 GB. Custom Path: any core ONNX file (Files group).",
        group: G_MAIN,
        kind: Kind::Choice { opts: &MODEL_LABELS, def: MODEL_DEFAULT },
    },
    ParamDef {
        name: P_DEVICE,
        label: "Device",
        hint: "Local = the model on this workstation (Flame's onnxruntime). Remote = the same ONNX on the cwtools-infer \
               server (see Remote group): no local VRAM, big crops fit. Same pre/post, result within 1 level.",
        group: G_MAIN,
        kind: Kind::Choice { opts: &["Local", "Remote"], def: 0 },
    },
    ParamDef {
        name: P_RELEASE,
        label: "Release GPU Memory",
        hint: "Unload the local model session(s) shared by all cwRefine nodes and free their VRAM; the next Local \
               render reloads (~1-3 s). Remote: nothing on the server is touched (it frees its own memory).",
        group: G_MAIN,
        kind: Kind::Button,
    },
    ParamDef { name: G_OUTPUT, label: "Output", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_VIEW,
        label: "Result RGB",
        hint: "Result socket only. Matte = Matte Out in RGB, Premultiplied = plate x Matte Out, Trimap = what the \
               model sees, Crop Region = the model's crop box, Mask vs Matte: red removed / green added, Channel \
               Pickers = see Result Channels.",
        group: G_OUTPUT,
        kind: Kind::Choice {
            opts: &["Plate", "Matte", "Premultiplied", "Trimap", "Crop Region", "Mask vs Matte", "Channel Pickers"],
            def: VIEW_PLATE,
        },
    },
    ParamDef {
        name: P_MATTE_OUT,
        label: "Matte Out",
        hint: "What Flame's OutMatte socket carries, whatever Result RGB shows: the refined Matte, or Inverted (1 - matte).",
        group: G_OUTPUT,
        kind: Kind::Choice { opts: &["Matte", "Inverted"], def: MATTE_OUT_MATTE },
    },
    ParamDef { name: G_SOURCES, label: "Sources", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_COLORSPACE,
        label: "Input Colour",
        hint: "Encoding of the Source RGB. ACEScg, ACEScct and Linear Rec.709 go through the ACES 2.0 SDR Rec.709 view (as Flame's View Transform); Video / sRGB passes as-is (display plates, or after a View Transform).",
        group: G_SOURCES,
        kind: Kind::Choice { opts: &["ACEScg (linear)", "Linear Rec.709", "Video / sRGB", "ACEScct (log)"], def: CS_ACESCG },
    },
    ParamDef {
        name: P_USE_TRIMAP,
        label: "Use Trimap Clip",
        hint: "If the Trimap clip is connected and not black, it replaces the generated trimap (0 BG / 0.5 unknown / 1 \
               FG; <= 0.1 = BG, > 0.9 = FG) and its non-BG region sets the crop. Matte kernels are then ignored.",
        group: G_SOURCES,
        kind: Kind::Bool { def: true },
    },
    ParamDef { name: G_MASK, label: "Mask Prep", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_INVERT_SEED,
        label: "Invert Mask",
        hint: "Use 1 - mask (for masks where the subject is black).",
        group: G_MASK,
        kind: Kind::Bool { def: false },
    },
    ParamDef {
        name: P_MASK_BLACK,
        label: "Black Clip",
        hint: "Mask levels (0-255) at or below this become 0 before the trimap (removes faint grey noise around the \
               subject). 0 = off (default, output unchanged).",
        group: G_MASK,
        kind: dbl(0.0, 0.0, 254.0, 0.0, 60.0, 0),
    },
    ParamDef {
        name: P_SEED_CLAMP,
        label: "White Clip",
        hint: "Mask levels (0-255) at or above this become solid 255 before the trimap, so near-white mask interiors \
               count as certain FG. 255 = off. Default 250.",
        group: G_MASK,
        kind: dbl(250.0, 1.0, 255.0, 200.0, 255.0, 0),
    },
    ParamDef {
        name: P_PRE_DILATE,
        label: "Pre-Dilate",
        hint: "Grow the mask by this many px (square) before bbox + trimap. Useful for tight binary masks (SAM). \
               Default 0.",
        group: G_MASK,
        kind: dbl(0.0, 0.0, 200.0, 0.0, 60.0, 0),
    },
    ParamDef { name: G_TRIMAP, label: "Trimap", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_BAND_FROM,
        label: "Band From",
        hint: "Where Erode In / Dilate Out start. Mask Edges = from the solid core in and the outermost non-black pixel \
               out (softer mask = wider band). Mid Point = from the Mid Point contour (band = Erode + Dilate).",
        group: G_TRIMAP,
        kind: Kind::Choice { opts: &["Mask Edges", "Mid Point"], def: 0 },
    },
    ParamDef {
        name: P_MID_POINT,
        label: "Mid Point",
        hint: "Band From = Mid Point: the mask level the band is centred on. Lower = the band moves out (toward the \
               background), higher = in (toward the core). 0.5 = the mask's half-way contour.",
        group: G_TRIMAP,
        kind: dbl(0.5, 0.01, 0.99, 0.01, 0.99, 2),
    },
    ParamDef {
        name: P_ERODE_IN,
        label: "Erode In",
        hint: "Ellipse kernel (px) eroding the mask into the certain-FG core. Small keeps the body solid. Default 4.",
        group: G_TRIMAP,
        kind: dbl(15.0, 1.0, 200.0, 1.0, 60.0, 0),
    },
    ParamDef {
        name: P_DILATE_OUT,
        label: "Dilate Out",
        hint: "Ellipse kernel (px) dilating the mask into the unknown band. Larger gives the model room for hair and \
               soft edges. Default 30.",
        group: G_TRIMAP,
        kind: dbl(30.0, 1.0, 400.0, 1.0, 120.0, 0),
    },
    ParamDef { name: G_CROP, label: "Model Input", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_MAX_CROP,
        label: "Max Crop",
        hint: "Crops with a longer edge are Lanczos-downscaled for the model and scaled back. VRAM grows ~ quadratically. \
               ViTMatte: 2048 default, 1536 ~ a third of the memory. MEMatte: 2048 ~4.3 GB; Custom 4096 = no downscale \
               up to UHD (~7 GB).",
        group: G_CROP,
        kind: Kind::Choice { opts: &["1024", "1536", "2048", "Custom"], def: 2 },
    },
    ParamDef {
        name: P_MAX_CROP_CUSTOM,
        label: "Custom Max Crop",
        hint: "Long-edge limit (px) when Max Crop = Custom.",
        group: G_CROP,
        kind: dbl(2048.0, 256.0, 4096.0, 512.0, 4096.0, 0),
    },
    ParamDef {
        name: P_PADDING,
        label: "Padding",
        hint: "Pixels added around the mask bbox before cropping (context for the model). Default 128.",
        group: G_CROP,
        kind: dbl(128.0, 0.0, 2048.0, 0.0, 512.0, 0),
    },
    ParamDef {
        name: P_MID_BIAS,
        label: "Mid Bias",
        hint: "The value the model is given for the unknown band (0.5 = neutral, default). > 0.5 biases toward FG (keeps thin \
               detail), < 0.5 toward BG (cleaner edges). It does not move the band itself.",
        group: G_CROP,
        kind: dbl(0.5, 0.0, 1.0, 0.0, 1.0, 3),
    },
    // ── Page 2 ──────────────────────────────────────────────────────────
    ParamDef { name: G_LOCAL, label: "Local", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_USE_GPU,
        label: "Use GPU (CUDA)",
        hint: "Local only: run on the CUDA execution provider (falls back to CPU, ~2-3 s per 800 px crop, if it cannot \
               be created). Remote always runs on the server's GPU.",
        group: G_LOCAL,
        kind: Kind::Bool { def: true },
    },
    ParamDef {
        name: P_GPU_MATH,
        label: "GPU Math",
        hint: "FP32 = full-precision GEMMs (closest to the PyTorch reference). TF32 = tensor-core GEMMs, faster on RTX 30xx+ \
               with ~1-level differences. Each choice is its own cached session (also on the server).",
        group: G_LOCAL,
        kind: Kind::Choice { opts: &["FP32 (exact)", "TF32 (fast)"], def: 0 },
    },
    ParamDef {
        name: P_FREE_VRAM,
        label: "Free VRAM/Frame",
        hint: "Return the model's working memory to the GPU after every frame (keeps the weights loaded). Leaves VRAM \
               for Flame at a small per-frame cost. Off = keep the arena (fastest).",
        group: G_LOCAL,
        kind: Kind::Bool { def: true },
    },
    ParamDef {
        name: P_CACHE,
        label: "Cache Results",
        hint: "Remember refined mattes keyed on the exact model input, so re-renders and view switches are instant.",
        group: G_LOCAL,
        kind: Kind::Bool { def: true },
    },
    ParamDef { name: G_AUX, label: "Result Channels", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_AUX_R,
        label: "Result R",
        hint: "Result RGB = Channel Pickers: what red carries. Plate = the plate's red, Trimap = what the model sees, Input Mask = the mask fed in, Crop Region = the model's crop box, Difference = |refined - mask|, Band Class = BG/outer 2/band/core 0/.33/.67/1.",
        group: G_AUX,
        kind: Kind::Choice { opts: AUX_OPTS, def: AUX_TRIMAP },
    },
    ParamDef {
        name: P_AUX_G,
        label: "Result G",
        hint: "Result RGB = Channel Pickers: what green carries (see Result R). Plate = the plate's green.",
        group: G_AUX,
        kind: Kind::Choice { opts: AUX_OPTS, def: AUX_SEED },
    },
    ParamDef {
        name: P_AUX_B,
        label: "Result B",
        hint: "Result RGB = Channel Pickers: what blue carries (see Result R). Matte Out still carries the refined matte.",
        group: G_AUX,
        kind: Kind::Choice { opts: AUX_OPTS, def: AUX_DIFF },
    },
    ParamDef { name: G_FILES, label: "Files", hint: "", group: "", kind: Kind::Group },
    ParamDef { name: "capModelDir", label: "Model Folder", hint: "", group: G_FILES, kind: Kind::Caption },
    ParamDef {
        name: P_MODEL_DIR,
        label: "Model Folder",
        hint: "Folder with vitmatte_core.onnx and mematte_vitb_core.onnx. Empty = the installed models folder (cwtools.conf).",
        group: G_FILES,
        kind: Kind::Str { def: DEFAULT_MODEL_DIR },
    },
    ParamDef { name: "capModelPath", label: "Custom Model", hint: "", group: G_FILES, kind: Kind::Caption },
    ParamDef {
        name: P_MODEL_PATH,
        label: "Custom Model",
        hint: "Model = Custom Path: full path of a core ONNX (pixel_values [1,4,h,w] -> alphas [1,1,h,w], h/w multiples \
               of 32, RGB (x/255-0.5)/0.5 + trimap/255).",
        group: G_FILES,
        kind: Kind::Str { def: "" },
    },
    ParamDef { name: "cwtoolsVersion", label: "Version", hint: "", group: G_FILES, kind: Kind::Version },
    ParamDef { name: G_RUNTIME, label: "ONNX Runtime", hint: "", group: "", kind: Kind::Group },
    ParamDef { name: "capOrtLib", label: "ONNX Runtime Library", hint: "", group: G_RUNTIME, kind: Kind::Caption },
    ParamDef {
        name: P_ORT_LIB,
        label: "ONNX Runtime",
        hint: "Path to libonnxruntime (1.22). Empty = ort_lib from cwtools.conf, else auto (Flame's own copy). Read once \
               per Flame session.",
        group: G_RUNTIME,
        kind: Kind::Str { def: "" },
    },
    ParamDef { name: G_REMOTE, label: "Remote", hint: "", group: "", kind: Kind::Group },
    ParamDef { name: "capServer", label: "Server URL", hint: "", group: G_REMOTE, kind: Kind::Caption },
    ParamDef {
        name: P_SERVER,
        label: "Server URL",
        hint: "cwtools-infer server base URL, e.g. http://host:8201. Empty = remote_server from cwtools.conf.",
        group: G_REMOTE,
        kind: Kind::Str { def: cwtools_remote::plugin::DEFAULT_SERVER },
    },
    ParamDef {
        name: P_TIMEOUT,
        label: "Timeout (s)",
        hint: "Give up on the server after this many seconds; the input mask passes through with a node message. Raise \
               it for large crops or slow networks.",
        group: G_REMOTE,
        kind: dbl(cwtools_remote::plugin::DEFAULT_TIMEOUT_S, 2.0, 120.0, 5.0, 120.0, 0),
    },
    ParamDef {
        name: P_TEST_SERVER,
        label: "Test Server",
        hint: "Ask the server for its health (VRAM, loaded models) and whether it has this node's model with the same \
               sha256. Result in the node message and /tmp/cwRefine.log.",
        group: G_REMOTE,
        kind: Kind::Button,
    },
    // ── Spline (feature `spline`, not described by default) ─────────────
    ParamDef { name: G_SPLINE, label: "Spline", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_SPLINE_ON,
        label: "Spline Trimap",
        hint: "Apply the spline edits (points, gradient, Outer 2) to the generated trimap. With no points, Flat gradient and Outer 2 off the trimap is exactly the standard one.",
        group: G_SPLINE,
        kind: Kind::Bool { def: true },
    },
    ParamDef {
        name: P_SPLINE_EDIT,
        label: "Edit Splines",
        hint: "Edit: the viewer shows plate + bands (no ViTMatte) and the overlay is interactive: drag the outer/inner line (Dilate Out / Erode In), click to add a point, drag its handle. Off = Finish: refine with it.",
        group: G_SPLINE,
        kind: Kind::Bool { def: false },
    },
    ParamDef {
        name: P_SPLINE_SHOW,
        label: "Show Splines",
        hint: "Draw the inner (green) / middle (white) / outer (yellow) / outer 2 (orange) lines and the point handles in the viewer also when not editing.",
        group: G_SPLINE,
        kind: Kind::Bool { def: false },
    },
    ParamDef {
        name: P_BAND_GRADIENT,
        label: "Band Gradient",
        hint: "Trimap value across the unknown band. Flat = 0.5 (default; Mid Bias applies). Linear / Smooth = Inner Level at the inner line to Outer Level at the outer line, shaped by Gradient Bias.",
        group: G_SPLINE,
        kind: Kind::Choice { opts: &["Flat (default)", "Linear", "Smooth"], def: 0 },
    },
    ParamDef {
        name: P_INNER_LEVEL,
        label: "Inner Level",
        hint: "Gradient only: trimap value at the inner line (> 0.5 leans toward FG).",
        group: G_SPLINE,
        kind: dbl(0.75, 0.0, 1.0, 0.0, 1.0, 3),
    },
    ParamDef {
        name: P_OUTER_LEVEL,
        label: "Outer Level",
        hint: "Gradient only: trimap value at the outer line (< 0.5 leans toward BG).",
        group: G_SPLINE,
        kind: dbl(0.25, 0.0, 1.0, 0.0, 1.0, 3),
    },
    ParamDef {
        name: P_GRAD_BIAS,
        label: "Gradient Bias",
        hint: "Gradient only: 0.5 = even; lower keeps Inner Level further out, higher brings Outer Level further in.",
        group: G_SPLINE,
        kind: dbl(0.5, 0.01, 0.99, 0.01, 0.99, 3),
    },
    ParamDef {
        name: P_OUTER2_ON,
        label: "Outer 2 On",
        hint: "A second band outside the outer line (e.g. 'probably background' for flyaways) at Outer 2 Level.",
        group: G_SPLINE,
        kind: Kind::Bool { def: false },
    },
    ParamDef {
        name: P_OUTER2_WIDTH,
        label: "Outer 2 Width",
        hint: "Width (px, full resolution) of the Outer 2 band beyond the outer line. Follows the points' local Out offsets.",
        group: G_SPLINE,
        kind: dbl(40.0, 0.0, 1000.0, 0.0, 200.0, 0),
    },
    ParamDef {
        name: P_OUTER2_LEVEL,
        label: "Outer 2 Level",
        hint: "Trimap value inside the Outer 2 band (0 = BG, 0.5 = unknown).",
        group: G_SPLINE,
        kind: dbl(0.2, 0.0, 1.0, 0.0, 1.0, 3),
    },
    ParamDef {
        name: P_CLEAR_POINTS,
        label: "Clear Points",
        hint: "Turn every control point off (their values stay, so On brings one back).",
        group: G_SPLINE,
        kind: Kind::Button,
    },
    ParamDef { name: G_POINTS, label: "Spline Points", hint: "", group: "", kind: Kind::Group },
    ParamDef { name: "cp1On", label: "Point 1 On", hint: HINT_ON, group: G_POINTS, kind: Kind::Bool { def: false } },
    ParamDef { name: "cp1U", label: "Point 1 U", hint: HINT_U, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp1V", label: "Point 1 V", hint: HINT_V, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp1Out", label: "Point 1 Out", hint: HINT_OUT, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -50.0, 200.0, 0) },
    ParamDef { name: "cp1In", label: "Point 1 In", hint: HINT_IN, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -20.0, 100.0, 0) },
    ParamDef { name: "cp1Width", label: "Point 1 Width", hint: HINT_W, group: G_POINTS, kind: dbl(150.0, 1.0, 5000.0, 10.0, 600.0, 0) },
    ParamDef { name: "cp2On", label: "Point 2 On", hint: HINT_ON, group: G_POINTS, kind: Kind::Bool { def: false } },
    ParamDef { name: "cp2U", label: "Point 2 U", hint: HINT_U, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp2V", label: "Point 2 V", hint: HINT_V, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp2Out", label: "Point 2 Out", hint: HINT_OUT, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -50.0, 200.0, 0) },
    ParamDef { name: "cp2In", label: "Point 2 In", hint: HINT_IN, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -20.0, 100.0, 0) },
    ParamDef { name: "cp2Width", label: "Point 2 Width", hint: HINT_W, group: G_POINTS, kind: dbl(150.0, 1.0, 5000.0, 10.0, 600.0, 0) },
    ParamDef { name: "cp3On", label: "Point 3 On", hint: HINT_ON, group: G_POINTS, kind: Kind::Bool { def: false } },
    ParamDef { name: "cp3U", label: "Point 3 U", hint: HINT_U, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp3V", label: "Point 3 V", hint: HINT_V, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp3Out", label: "Point 3 Out", hint: HINT_OUT, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -50.0, 200.0, 0) },
    ParamDef { name: "cp3In", label: "Point 3 In", hint: HINT_IN, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -20.0, 100.0, 0) },
    ParamDef { name: "cp3Width", label: "Point 3 Width", hint: HINT_W, group: G_POINTS, kind: dbl(150.0, 1.0, 5000.0, 10.0, 600.0, 0) },
    ParamDef { name: "cp4On", label: "Point 4 On", hint: HINT_ON, group: G_POINTS, kind: Kind::Bool { def: false } },
    ParamDef { name: "cp4U", label: "Point 4 U", hint: HINT_U, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp4V", label: "Point 4 V", hint: HINT_V, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp4Out", label: "Point 4 Out", hint: HINT_OUT, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -50.0, 200.0, 0) },
    ParamDef { name: "cp4In", label: "Point 4 In", hint: HINT_IN, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -20.0, 100.0, 0) },
    ParamDef { name: "cp4Width", label: "Point 4 Width", hint: HINT_W, group: G_POINTS, kind: dbl(150.0, 1.0, 5000.0, 10.0, 600.0, 0) },
    ParamDef { name: "cp5On", label: "Point 5 On", hint: HINT_ON, group: G_POINTS, kind: Kind::Bool { def: false } },
    ParamDef { name: "cp5U", label: "Point 5 U", hint: HINT_U, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp5V", label: "Point 5 V", hint: HINT_V, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp5Out", label: "Point 5 Out", hint: HINT_OUT, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -50.0, 200.0, 0) },
    ParamDef { name: "cp5In", label: "Point 5 In", hint: HINT_IN, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -20.0, 100.0, 0) },
    ParamDef { name: "cp5Width", label: "Point 5 Width", hint: HINT_W, group: G_POINTS, kind: dbl(150.0, 1.0, 5000.0, 10.0, 600.0, 0) },
    ParamDef { name: "cp6On", label: "Point 6 On", hint: HINT_ON, group: G_POINTS, kind: Kind::Bool { def: false } },
    ParamDef { name: "cp6U", label: "Point 6 U", hint: HINT_U, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp6V", label: "Point 6 V", hint: HINT_V, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp6Out", label: "Point 6 Out", hint: HINT_OUT, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -50.0, 200.0, 0) },
    ParamDef { name: "cp6In", label: "Point 6 In", hint: HINT_IN, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -20.0, 100.0, 0) },
    ParamDef { name: "cp6Width", label: "Point 6 Width", hint: HINT_W, group: G_POINTS, kind: dbl(150.0, 1.0, 5000.0, 10.0, 600.0, 0) },
    ParamDef { name: "cp7On", label: "Point 7 On", hint: HINT_ON, group: G_POINTS, kind: Kind::Bool { def: false } },
    ParamDef { name: "cp7U", label: "Point 7 U", hint: HINT_U, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp7V", label: "Point 7 V", hint: HINT_V, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp7Out", label: "Point 7 Out", hint: HINT_OUT, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -50.0, 200.0, 0) },
    ParamDef { name: "cp7In", label: "Point 7 In", hint: HINT_IN, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -20.0, 100.0, 0) },
    ParamDef { name: "cp7Width", label: "Point 7 Width", hint: HINT_W, group: G_POINTS, kind: dbl(150.0, 1.0, 5000.0, 10.0, 600.0, 0) },
    ParamDef { name: "cp8On", label: "Point 8 On", hint: HINT_ON, group: G_POINTS, kind: Kind::Bool { def: false } },
    ParamDef { name: "cp8U", label: "Point 8 U", hint: HINT_U, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp8V", label: "Point 8 V", hint: HINT_V, group: G_POINTS, kind: dbl(0.5, -1.0, 2.0, 0.0, 1.0, 3) },
    ParamDef { name: "cp8Out", label: "Point 8 Out", hint: HINT_OUT, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -50.0, 200.0, 0) },
    ParamDef { name: "cp8In", label: "Point 8 In", hint: HINT_IN, group: G_POINTS, kind: dbl(0.0, -500.0, 2000.0, -20.0, 100.0, 0) },
    ParamDef { name: "cp8Width", label: "Point 8 Width", hint: HINT_W, group: G_POINTS, kind: dbl(150.0, 1.0, 5000.0, 10.0, 600.0, 0) },
];

/// True for the spline UI params (the Spline / Spline Points groups and everything in them).
pub fn is_spline_param(d: &ParamDef) -> bool {
    [G_SPLINE, G_POINTS].contains(&d.name) || [G_SPLINE, G_POINTS].contains(&d.group)
}

/// The params the plugin describes: [`PARAMS`] without the spline UI unless
/// the `spline` feature is on ([`crate::SPLINE_UI`]).
pub fn described() -> Vec<ParamDef> {
    PARAMS.iter().copied().filter(|d| crate::SPLINE_UI || !is_spline_param(d)).collect()
}


#[derive(Clone, Debug)]
pub struct Values {
    pub view: i32,
    pub colorspace: i32,
    pub use_trimap_clip: bool,
    pub trimap: TrimapParams,
    pub pre_dilate: usize,
    pub invert_seed: bool,
    /// Mask Black Clip: mask levels <= this become 0 (0 = off).
    pub black_clip: u8,
    pub padding: usize,
    pub max_crop: usize,
    pub remote: bool,
    pub use_gpu: bool,
    pub tf32: bool,
    pub free_vram: bool,
    pub cache: bool,
    /// Resolved core ONNX path (Model choice + folder, or Custom Model).
    pub model_path: String,
    /// Model name for node messages ("ViTMatte", "MEMatte", "Custom model").
    pub model_name: String,
    pub ort_lib: String,
    /// Remote server base URL (from Server URL or the IP / Port ints).
    pub server: String,
    /// Remote request timeout (s).
    pub timeout_s: f64,
    /// Spline edits at the render scale (pixel quantities scaled).
    pub spline: crate::spline::SplineParams,
    /// Spline edits as set (full-resolution px), for the overlay geometry.
    pub spline_full: crate::spline::SplineParams,
    pub spline_edit: bool,
    pub spline_show: bool,
    /// Matte Out = Inverted (1 - refined matte on OutMatte, Matte / Premultiplied views).
    pub matte_inverted: bool,
    pub aux: [i32; 3],
}

fn px(v: f64, min: usize) -> usize {
    if v.is_finite() {
        (v.round().max(0.0) as usize).max(min)
    } else {
        min
    }
}

impl Values {
    /// The production hair recipe with display-referred input (what the parity tests use).
    pub fn production() -> Values {
        Values {
            view: VIEW_PLATE,
            colorspace: cwtools_core::color::CS_VIDEO,
            use_trimap_clip: true,
            trimap: TrimapParams::HAIR,
            pre_dilate: 0,
            invert_seed: false,
            black_clip: 0,
            padding: 128,
            max_crop: 2048,
            remote: false,
            use_gpu: true,
            tf32: false,
            free_vram: true,
            cache: true,
            model_path: model_file(MODEL_VITMATTE, &model_dir(""), "").0,
            model_name: MODELS[MODEL_VITMATTE as usize].label.into(),
            ort_lib: String::new(),
            server: cwtools_core::config::get().remote_server.clone(),
            timeout_s: cwtools_remote::plugin::DEFAULT_TIMEOUT_S,
            spline: crate::spline::SplineParams::DEFAULT,
            spline_full: crate::spline::SplineParams::DEFAULT,
            spline_edit: false,
            spline_show: false,
            matte_inverted: false,
            aux: [AUX_TRIMAP, AUX_SEED, AUX_DIFF],
        }
    }

    /// Scale pixel-sized knobs by the render scale (proxy renders).
    pub fn scaled(&self, rs: f64) -> Values {
        if (rs - 1.0).abs() < 1e-9 || !rs.is_finite() || rs <= 0.0 {
            return self.clone();
        }
        let s = |v: usize, min: usize| ((v as f64 * rs).round() as usize).max(min);
        let mut v = self.clone();
        v.trimap.kernel_size = s(v.trimap.kernel_size, 1);
        v.trimap.asym = v.trimap.asym.map(|(a, b)| (s(a, 1), s(b, 1)));
        v.pre_dilate = s(v.pre_dilate, 0);
        v.padding = s(v.padding, 0);
        v.spline = v.spline_full.scaled(rs);
        v
    }
}

/// Conv search for the model named `name` (Values.model_name); Heuristic for a custom model.
pub fn conv_search_for(name: &str) -> ConvSearch {
    MODELS.iter().find(|e| e.label == name).map(|e| e.conv_search).unwrap_or(ConvSearch::Heuristic)
}

/// Model choice -> (core ONNX path, name for messages). Custom Path (or an
/// out-of-range choice) uses `custom` as given (empty = not set).
pub fn model_file(model: i32, dir: &str, custom: &str) -> (String, String) {
    match model_entry(model) {
        Some(e) => (std::path::Path::new(dir).join(e.file).to_string_lossy().into_owned(), e.label.to_string()),
        None => (custom.trim().to_string(), "Custom model".to_string()),
    }
}

pub fn read_values(ps: OfxParamSetHandle, t: f64) -> Values {
    let d = |name: &str| get_double(ps, &n(name), t);
    let i = |name: &str| get_int(ps, &n(name), t);
    let b = |name: &str| get_int(ps, &n(name), t) != 0;
    let asym = Some((px(d(P_ERODE_IN), 1), px(d(P_DILATE_OUT), 1)));
    let mp = d(P_MID_POINT);
    let mid_point = if i(P_BAND_FROM) == 1 { Some(if mp.is_finite() { mp.clamp(0.01, 0.99) } else { 0.5 }) } else { None };
    // 0.5 = production (None = 128, exactly what 0.5 gives); the On toggle is gone
    let m = d(P_MID_BIAS);
    let mid_bias = if m.is_finite() && (m - 0.5).abs() > 1e-9 { Some(m.clamp(0.0, 1.0)) } else { None };
    // Spline UI off: its params are not described, so never query them;
    // identity spline = the production trimap.
    let sp = if crate::SPLINE_UI { read_spline(ps, t) } else { crate::spline::SplineParams::DEFAULT };
    let mc = i(P_MAX_CROP);
    let max_crop = if (0..3).contains(&mc) {
        MAX_CROP_CHOICES[mc as usize]
    } else {
        px(d(P_MAX_CROP_CUSTOM), 256).min(8192)
    };
    let (model_path, model_name) = model_file(i(P_MODEL), &model_dir(&get_string(ps, &n(P_MODEL_DIR))), &get_string(ps, &n(P_MODEL_PATH)));
    Values {
        view: i(P_VIEW),
        colorspace: i(P_COLORSPACE),
        use_trimap_clip: b(P_USE_TRIMAP),
        trimap: TrimapParams {
            kernel_size: TrimapParams::HAIR.kernel_size,
            asym,
            mid_bias,
            mid_point,
            seed_clamp: px(d(P_SEED_CLAMP), 1).min(255) as i32,
        },
        pre_dilate: px(d(P_PRE_DILATE), 0),
        invert_seed: b(P_INVERT_SEED),
        black_clip: px(d(P_MASK_BLACK), 0).min(254) as u8,
        padding: px(d(P_PADDING), 0),
        max_crop,
        remote: i(P_DEVICE) == 1,
        use_gpu: b(P_USE_GPU),
        tf32: i(P_GPU_MATH) == 1,
        free_vram: b(P_FREE_VRAM),
        cache: b(P_CACHE),
        model_path,
        model_name,
        ort_lib: get_string(ps, &n(P_ORT_LIB)),
        server: cwtools_remote::plugin::server_url(
            &get_string(ps, &n(P_SERVER)),
            &cwtools_core::config::get().remote_server,
        ),
        timeout_s: cwtools_remote::plugin::clamp_timeout(d(P_TIMEOUT)),
        spline: sp.clone(),
        spline_full: sp,
        spline_edit: crate::SPLINE_UI && b(P_SPLINE_EDIT),
        spline_show: crate::SPLINE_UI && b(P_SPLINE_SHOW),
        matte_inverted: i(P_MATTE_OUT) == MATTE_OUT_INVERTED,
        aux: [i(P_AUX_R), i(P_AUX_G), i(P_AUX_B)],
    }
}

/// Spline params (full resolution) at time `t`.
pub fn read_spline(ps: OfxParamSetHandle, t: f64) -> crate::spline::SplineParams {
    use crate::spline::{CtrlPoint, SplineParams, MAX_POINTS};
    let d = |name: &str| get_double(ps, &n(name), t);
    let b = |name: &str| get_int(ps, &n(name), t) != 0;
    let fin = |x: f64, def: f64| if x.is_finite() { x } else { def };
    let mut sp = SplineParams {
        enabled: b(P_SPLINE_ON),
        falloff: get_int(ps, &n(P_BAND_GRADIENT), t),
        inner_level: fin(d(P_INNER_LEVEL), 0.75),
        outer_level: fin(d(P_OUTER_LEVEL), 0.25),
        bias: fin(d(P_GRAD_BIAS), 0.5),
        outer2_on: b(P_OUTER2_ON),
        outer2_width: fin(d(P_OUTER2_WIDTH), 0.0),
        outer2_level: fin(d(P_OUTER2_LEVEL), 0.2),
        points: [CtrlPoint::OFF; MAX_POINTS],
    };
    for (i, p) in sp.points.iter_mut().enumerate() {
        *p = CtrlPoint {
            on: b(&cp_name(i, "On")),
            u: fin(d(&cp_name(i, "U")), 0.5),
            v: fin(d(&cp_name(i, "V")), 0.5),
            out_px: fin(d(&cp_name(i, "Out")), 0.0),
            in_px: fin(d(&cp_name(i, "In")), 0.0),
            width_px: fin(d(&cp_name(i, "Width")), 150.0),
        };
    }
    sp
}

/// Load-time defaults for the setup fields: what an empty field resolves to
/// (cwtools.conf), so a new node shows the real server / folder / library.
/// Clearing a field still falls back to cwtools.conf at render.
/// Empty setup fields of an existing node (older setup) get the same values.
pub fn fill_setup_fields(ps: OfxParamSetHandle) {
    let f: Vec<(Vec<u8>, String)> =
        [P_SERVER, P_MODEL_DIR, P_ORT_LIB].iter().filter_map(|nm| shown_default(nm).map(|v| (n(nm), v))).collect();
    let refs: Vec<(&[u8], String)> = f.iter().map(|(k, v)| (k.as_slice(), v.clone())).collect();
    cwtools_core::params::fill_empty_strings(ps, &refs);
}

pub fn shown_default(name: &str) -> Option<String> {
    let cfg = cwtools_core::config::get();
    match name {
        P_SERVER => Some(cwtools_remote::plugin::server_url("", &cfg.remote_server)),
        P_MODEL_DIR => Some(cfg.refine_models.clone()),
        P_ORT_LIB => Some(cwtools_core::config::shown_ort_lib()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn conv_search_per_model() {
        use cwtools_core::ort_rt::ConvSearch;
        assert_eq!(super::conv_search_for("MEMatte"), ConvSearch::Default);
        assert_eq!(super::conv_search_for("ViTMatte"), ConvSearch::Heuristic);
        assert_eq!(super::conv_search_for("Custom model"), ConvSearch::Heuristic);
    }

    use super::*;
    #[test]
    fn param_table_fits_flame_limits() {
        cwtools_core::params::validate(PARAMS).unwrap();
    }
    #[test]
    fn defaults_are_the_production_hair_recipe() {
        let get = |name: &str| PARAMS.iter().find(|p| p.name == name).unwrap().kind;
        let def = |name: &str| match get(name) {
            Kind::Double { def, .. } => def,
            Kind::Choice { def, .. } => def as f64,
            Kind::Bool { def } => def as i32 as f64,
            _ => f64::NAN,
        };
        assert_eq!(def(P_BAND_FROM), 0.0); // mask edges (production)
        // node default 15 (user 2026-10-22: a firmer core); the production recipe (parity) keeps 4
        assert_eq!(def(P_ERODE_IN), 15.0);
        assert_eq!(TrimapParams::HAIR.asym, Some((4, 30)));
        assert_eq!(def(P_DILATE_OUT), 30.0);
        assert_eq!(def(P_SEED_CLAMP), 250.0);
        assert_eq!(def(P_PRE_DILATE), 0.0);
        assert_eq!(def(P_MID_BIAS), 0.5);
        assert_eq!(def(P_PADDING), 128.0);
        assert_eq!(MAX_CROP_CHOICES[def(P_MAX_CROP) as usize], 2048);
        assert_eq!(TrimapParams::HAIR, Values::production().trimap);
        // spline defaults = identity (production trimap unchanged)
        assert_eq!(def(P_SPLINE_ON), 1.0);
        assert_eq!(def(P_BAND_GRADIENT), 0.0);
        assert_eq!(def(P_OUTER2_ON), 0.0);
        for i in 0..crate::spline::MAX_POINTS {
            assert_eq!(def(&cp_name(i, "On")), 0.0);
            assert_eq!(def(&cp_name(i, "Out")), 0.0);
            assert_eq!(def(&cp_name(i, "In")), 0.0);
        }
        assert!(Values::production().spline.is_identity());
    }

    #[test]
    fn default_model_is_mematte_and_production_stays_vitmatte() {
        let def = match PARAMS.iter().find(|p| p.name == P_MODEL).unwrap().kind {
            Kind::Choice { opts, def } => {
                assert_eq!(opts.len(), MODEL_LABELS.len());
                assert_eq!(opts[MODEL_CUSTOM as usize], "Custom Path");
                def
            }
            _ => panic!("Model is not a Choice"),
        };
        assert_eq!(def, MODEL_MEMATTE);
        // empty Model Folder = cwtools.conf refine_models
        let dir = cwtools_core::config::get().refine_models.clone();
        let (path, name) = model_file(def, &model_dir(DEFAULT_MODEL_DIR), "");
        assert_eq!(path, std::path::Path::new(&dir).join("mematte_vitb_core.onnx").to_string_lossy());
        assert_eq!(name, "MEMatte");
        // the parity recipe is still the validated ViTMatte setup
        let (vpath, vname) = model_file(MODEL_VITMATTE, &model_dir(DEFAULT_MODEL_DIR), "");
        assert_eq!(vname, Values::production().model_name);
        assert_eq!(Values::production().model_path, vpath);
        // an explicit folder in the node wins when it exists here (a missing one falls back to the installed folder)
        let tmp = std::env::temp_dir().to_string_lossy().trim_end_matches('/').to_string();
        assert_eq!(model_file(MODEL_VITMATTE, &model_dir(&format!(" {tmp} ")), "").0, format!("{tmp}/vitmatte_core.onnx"));
    }

    #[test]
    fn model_choice_resolves_files() {
        let (p, n) = model_file(MODEL_MEMATTE, "/m", "/x/y.onnx");
        assert_eq!((p.as_str(), n.as_str()), ("/m/mematte_vitb_core.onnx", "MEMatte"));
        #[cfg(feature = "test-models")]
        {
            let (p, n) = model_file(MODEL_SYNTH, "/m", "/x/y.onnx");
            assert_eq!((p.as_str(), n.as_str()), ("/m/mematte_synth_s1_core.onnx", "MEMatte Synth"));
            assert_eq!(MODEL_LABELS[MODEL_SYNTH as usize], "MEMatte Synth");
            assert_eq!(conv_search_for("MEMatte Synth"), ConvSearch::Default);
            let (p, n) = model_file(MODEL_SYNTH2, "/m", "");
            assert_eq!((p.as_str(), n.as_str()), ("/m/mematte_synth_s1b_core.onnx", "MEMatte Synth 2"));
            assert_eq!(MODEL_LABELS[MODEL_SYNTH2 as usize], "MEMatte Synth 2");
        }
        #[cfg(not(feature = "test-models"))]
        {
            assert_eq!(MODEL_LABELS.len(), 3);
            assert!(model_entry(MODEL_SYNTH).is_none() && model_entry(MODEL_SYNTH2).is_none());
        }
        let (p, n) = model_file(MODEL_CUSTOM, "/m", " /x/y.onnx ");
        assert_eq!((p.as_str(), n.as_str()), ("/x/y.onnx", "Custom model"));
        assert_eq!(model_file(7, "/m", "").0, "");
        assert_eq!(model_file(-1, "/m", "/c.onnx").0, "/c.onnx");
    }
    #[test]
    fn spline_slots_complete() {
        for i in 0..crate::spline::MAX_POINTS {
            for f in ["On", "U", "V", "Out", "In", "Width"] {
                let nm = cp_name(i, f);
                let p = PARAMS.iter().find(|p| p.name == nm).unwrap_or_else(|| panic!("{nm}"));
                if f != "On" {
                    assert!(matches!(p.kind, Kind::Double { .. }), "{nm} must animate (Double)");
                }
            }
        }
    }
    #[test]
    fn described_params_follow_the_spline_switch() {
        let d = described();
        cwtools_core::params::validate(&d).unwrap();
        let spline: Vec<_> = PARAMS.iter().filter(|p| is_spline_param(p)).map(|p| p.name).collect();
        assert_eq!(spline.len(), 2 + 11 + 6 * crate::spline::MAX_POINTS);
        for nm in [P_SPLINE_ON, P_SPLINE_EDIT, P_SPLINE_SHOW, P_BAND_GRADIENT, P_INNER_LEVEL, P_OUTER_LEVEL, P_GRAD_BIAS, P_OUTER2_ON, P_OUTER2_WIDTH, P_OUTER2_LEVEL, P_CLEAR_POINTS] {
            assert!(spline.contains(&nm), "{nm}");
        }
        if crate::SPLINE_UI {
            assert_eq!(d.len(), PARAMS.len());
        } else {
            assert_eq!(d.len(), PARAMS.len() - spline.len());
            assert!(d.iter().all(|p| !is_spline_param(p) && !p.name.starts_with("cp")));
            // everything else is still there
            for nm in [P_MASK_BLACK, P_MATTE_OUT, P_AUX_R, P_AUX_G, P_AUX_B, P_DEVICE, P_RELEASE, P_TEST_SERVER] {
                assert!(d.iter().any(|p| p.name == nm), "{nm}");
            }
        }
    }
    #[test]
    fn continuous_knobs_are_doubles() {
        for name in [P_ERODE_IN, P_DILATE_OUT, P_MID_POINT, P_SEED_CLAMP, P_PRE_DILATE, P_MID_BIAS, P_PADDING, P_MAX_CROP_CUSTOM] {
            let p = PARAMS.iter().find(|p| p.name == name).unwrap();
            assert!(matches!(p.kind, Kind::Double { .. }), "{name}");
        }
    }
    /// The Flame panel (cwtools_core::params::grid = the probed Flame rules):
    /// page 1 = daily controls, page 2 = setup. Version sits under Files.
    #[test]
    fn flame_panel_layout() {
        use cwtools_core::params::{grid_pages, grid_text};
        let d = described();
        let pages = grid_pages(&d);
        let heads: Vec<Vec<String>> = pages.iter().map(|p| p.iter().map(|c| c.header.clone()).collect()).collect();
        let want: Vec<Vec<&str>> = vec![
            vec!["Main", "Output", "Sources", "Mask Prep", "Trimap", "Model Input"],
            vec!["Local", "Result Channels", "Result Channels", "Files", "ONNX Runtime", "Remote"],
        ];
        assert_eq!(heads, want, "\n{}", grid_text(&d));
        assert_eq!(pages[0][0].rows, ["Model", "Device", "Release GPU Memory"], "\n{}", grid_text(&d));
    }
}
