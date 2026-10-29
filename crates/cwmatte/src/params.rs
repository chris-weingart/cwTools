//! cwMatte parameters (cwtools-core table DSL). Groups, not Pages (Flame);
//! continuous knobs are animatable Doubles; Double labels <= 15 chars, all
//! strings <= 255 bytes (guard tests below).

use cwtools_core::color::CS_ACESCG;
use cwtools_core::ofx_api::OfxParamSetHandle;
use cwtools_core::params::{get_double, get_int, get_string, get_xy, n, Kind, ParamDef};

/// Model Folder default: empty = `matte_models` from cwtools.conf.
pub const DEFAULT_MODEL_DIR: &str = "";

/// The Model Folder in effect: the param if set (and present here), else cwtools.conf `matte_models`.
pub fn model_dir(param: &str) -> String {
    cwtools_core::config::folder_or_installed(param, &cwtools_core::config::get().matte_models)
}

/// One Model menu entry: the ONNX used on the GPU and on the CPU (Use GPU
/// off, or no CUDA, e.g. macOS). HR Matting and General 1536 ship only as fp16
/// for the GPU (faster, less VRAM, matte within 0.007 of fp32); on the CPU they
/// would take 20-80 s a frame, so `cpu` is None and the node points to Lite.
pub struct ModelEntry {
    pub label: &'static str,
    pub gpu: &'static str,
    pub cpu: Option<&'static str>,
}

/// Model choice → files in the model folder (index = Model choice value),
/// largest first. HR Matting = BiRefNet_HR-matting (2048), General 1536 =
/// BiRefNet_dynamic (trained on 256..2304 px inputs), Lite = Swin-T general
/// (1024; the CPU / small-GPU model). Chosen on AIM-500 / P3M-500 (HR Matting
/// best on every content group, 1536 second; HR General, General 1024 / 768
/// dropped: never better than one of these).
pub const MODELS: [ModelEntry; 3] = [
    ModelEntry { label: "HR Matting", gpu: "BiRefNet_HR-matting-epoch_135_gs_fp16.onnx", cpu: None },
    ModelEntry { label: "General 1536", gpu: "BiRefNet_dynamic-general-epoch_174_1536_gs_fp16.onnx", cpu: None },
    ModelEntry {
        label: "Lite",
        gpu: "BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs.onnx",
        cpu: Some("BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs.onnx"),
    },
];
pub const MODEL_LABELS: [&str; 4] = [MODELS[0].label, MODELS[1].label, MODELS[2].label, "Custom Path"];
pub const MODEL_HR_MATTING: i32 = 0;
pub const MODEL_GENERAL_1536: i32 = 1;
pub const MODEL_LITE: i32 = 2;
pub const MODEL_CUSTOM: i32 = 3;
/// Default: HR Matting (best edges). It needs ~12 GB+ free on the GPU: next to
/// Flame on a 24 GB card use General 1536 or Device = Remote.
pub const MODEL_DEFAULT: i32 = MODEL_HR_MATTING;

/// Running on the CPU: Use GPU off, or a build without CUDA (macOS).
pub fn cpu_only(use_gpu: bool) -> bool {
    !use_gpu || cfg!(target_os = "macos")
}

pub const G_OUTPUT: &str = "grpOutput";
pub const G_INPUT: &str = "grpInput";
pub const G_MATTE: &str = "grpMatte";
pub const G_ROI: &str = "grpRoi";
pub const G_MAIN: &str = "grpMain";
pub const G_LOCAL: &str = "grpLocal";
pub const G_FILES: &str = "grpFiles";
pub const G_RUNTIME: &str = "grpRuntime";
pub const G_REMOTE: &str = "grpRemote";

pub const P_VIEW: &str = "view";
pub const P_COLORSPACE: &str = "inputColorSpace";
pub const P_GAMMA: &str = "gamma";
pub const P_CONTRAST: &str = "contrast";
pub const P_CLIP_BLACK: &str = "clipBlack";
pub const P_CLIP_WHITE: &str = "clipWhite";
pub const P_MATTE_OUT: &str = "matteOut";
pub const P_GARBAGE: &str = "garbageFromAlpha";
pub const P_ROI_MODE: &str = "roiMode";
pub const P_ROI_PAD: &str = "roiPadding";
pub const P_ROI_A: &str = "roiCornerA";
pub const P_ROI_B: &str = "roiCornerB";
pub const P_MODEL: &str = "model";
pub const P_MODEL_DIR: &str = "modelFolder";
pub const P_CUSTOM_MODEL: &str = "customModel";
pub const P_DEVICE: &str = "device";
pub const P_USE_GPU: &str = "useGpu";
pub const P_FREE_VRAM: &str = "freeVramPerFrame";
pub const P_CACHE: &str = "cacheResults";
pub const P_ORT_LIB: &str = "ortLibrary";
pub const P_SERVER: &str = "serverUrl";
pub const P_RELEASE: &str = "releaseGpu";
pub const P_TIMEOUT: &str = "timeoutSec";
pub const P_TEST_SERVER: &str = "testServer";

/// Result RGB (Flame's Result socket). Indices of the old Output menu kept
/// (Matte → Plate, Matte Only → Matte); Premultiplied appended.
pub const VIEW_PLATE: i32 = 0;
pub const VIEW_MATTE: i32 = 1;
pub const VIEW_OVERLAY: i32 = 2;
pub const VIEW_PREMULT: i32 = 3;

/// Matte Out (Flame's OutMatte socket).
pub const MATTE_OUT_MATTE: i32 = 0;
pub const MATTE_OUT_INVERTED: i32 = 1;

pub const ROI_OFF: i32 = 0;
pub const ROI_ALPHA: i32 = 1;
pub const ROI_MANUAL: i32 = 2;

const fn dbl(def: f64, min: f64, max: f64, dmin: f64, dmax: f64, digits: i32) -> Kind {
    Kind::Double { def, min, max, dmin, dmax, digits }
}

pub const PARAMS: &[ParamDef] = &[
    // Flame: 5-row columns, a Choice = 2 rows, every group starts a column,
    // 6 columns a page (cwtools_core::params::grid, checked in the tests).
    // ── Page 1 ──────────────────────────────────────────────────────────
    ParamDef { name: G_MAIN, label: "Main", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_MODEL,
        label: "Model",
        hint: "BiRefNet. HR Matting (default, 2048): best edges, needs ~12 GB free on the GPU (or Device = Remote). \
               General 1536: ~7 GB, fits next to Flame on 24 GB. Lite: small and fast, the CPU / macOS model.",
        group: G_MAIN,
        kind: Kind::Choice { opts: &MODEL_LABELS, def: MODEL_DEFAULT },
    },
    ParamDef {
        name: P_DEVICE,
        label: "Device",
        hint: "Local = BiRefNet on this workstation (Flame's onnxruntime). Remote = the same ONNX on the cwtools-infer \
               server (Remote group): no local VRAM, so the HR (2048) models work. Same pre/post as Local.",
        group: G_MAIN,
        kind: Kind::Choice { opts: &["Local", "Remote"], def: 0 },
    },
    ParamDef {
        name: P_RELEASE,
        label: "Release GPU Memory",
        hint: "Unload the local BiRefNet session(s) shared by all cwMatte nodes and free their VRAM; the next Local \
               render reloads. Remote: nothing on the server is touched (it frees its own memory).",
        group: G_MAIN,
        kind: Kind::Button,
    },
    ParamDef { name: G_OUTPUT, label: "Output", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_VIEW,
        label: "Result RGB",
        hint: "What Flame's Result socket carries. Plate = the plate unchanged. Matte = the Matte Out signal in RGB. \
               Overlay = plate with the subject tinted red, ROI box in yellow. Premultiplied = plate x Matte Out.",
        group: G_OUTPUT,
        kind: Kind::Choice { opts: &["Plate", "Matte", "Overlay", "Premultiplied"], def: VIEW_PLATE },
    },
    ParamDef {
        name: P_MATTE_OUT,
        label: "Matte Out",
        hint: "What Flame's OutMatte socket carries, whatever Result RGB shows. Matte = the subject. Inverted = 1 - \
               matte (the background), applied before the garbage mask.",
        group: G_OUTPUT,
        kind: Kind::Choice { opts: &["Matte", "Inverted"], def: MATTE_OUT_MATTE },
    },
    ParamDef { name: G_INPUT, label: "Input", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_COLORSPACE,
        label: "Input Colour",
        hint: "Encoding of the Source RGB. ACEScg, ACEScct and Linear Rec.709 go through the ACES 2.0 SDR Rec.709 view (as Flame's View Transform); Video / sRGB passes as-is (display plates, or after a View Transform).",
        group: G_INPUT,
        kind: Kind::Choice { opts: &["ACEScg (linear)", "Linear Rec.709", "Video / sRGB", "ACEScct (log)"], def: CS_ACESCG },
    },
    ParamDef {
        name: P_ROI_MODE,
        label: "ROI",
        hint: "Crop the plate before BiRefNet so the subject gets more model pixels (the model always sees a fixed \
               square). Source Alpha BBox = bbox of alpha > 0. Outside the ROI the matte is 0.",
        group: G_INPUT,
        kind: Kind::Choice { opts: &["Off (Full Frame)", "Source Alpha BBox", "Manual Box"], def: ROI_OFF },
    },
    ParamDef {
        name: P_GARBAGE,
        label: "Source Alpha = Garbage",
        hint: "Multiply the final matte by the Source alpha socket (a roto / garbage matte). Off = the incoming alpha \
               is ignored (unless ROI = Source Alpha BBox).",
        group: G_INPUT,
        kind: Kind::Bool { def: false },
    },
    ParamDef { name: G_MATTE, label: "Matte", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_CLIP_BLACK,
        label: "Clip Black",
        hint: "Matte values at or below this become 0 (the range Clip Black..Clip White is stretched to 0..1). Cleans \
               visible background haze; the model's near-zero floor (< 1/4096) is always cut to exactly 0.",
        group: G_MATTE,
        kind: dbl(0.0, 0.0, 1.0, 0.0, 1.0, 3),
    },
    ParamDef {
        name: P_CLIP_WHITE,
        label: "Clip White",
        hint: "Matte values at or above this become 1. Solidifies a slightly transparent core.",
        group: G_MATTE,
        kind: dbl(1.0, 0.0, 1.0, 0.0, 1.0, 3),
    },
    ParamDef {
        name: P_GAMMA,
        label: "Gamma",
        hint: "Matte gamma (after the clip): matte^(1/Gamma). > 1 thickens soft edges and hair, < 1 thins them. 1 = off.",
        group: G_MATTE,
        kind: dbl(1.0, 0.05, 10.0, 0.2, 5.0, 3),
    },
    ParamDef {
        name: P_CONTRAST,
        label: "Contrast",
        hint: "Matte contrast around 0.5 (after gamma): > 1 hardens edges, < 1 softens them. 1 = off.",
        group: G_MATTE,
        kind: dbl(1.0, 0.0, 10.0, 0.0, 4.0, 3),
    },
    ParamDef { name: G_ROI, label: "ROI Box", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_ROI_A,
        label: "ROI Corner A",
        hint: "Manual Box: one corner (0..1 of the frame, or canonical pixels).",
        group: G_ROI,
        kind: Kind::Xy { dx: 0.25, dy: 0.25 },
    },
    ParamDef {
        name: P_ROI_B,
        label: "ROI Corner B",
        hint: "Manual Box: the opposite corner.",
        group: G_ROI,
        kind: Kind::Xy { dx: 0.75, dy: 0.75 },
    },
    ParamDef {
        name: P_ROI_PAD,
        label: "ROI Padding",
        hint: "Pixels added around the ROI box (context for the model; clipped to the frame).",
        group: G_ROI,
        kind: dbl(64.0, 0.0, 4096.0, 0.0, 512.0, 0),
    },
    ParamDef { name: G_LOCAL, label: "Local", hint: "", group: "", kind: Kind::Group },
    ParamDef {
        name: P_USE_GPU,
        label: "Use GPU (CUDA)",
        hint: "Local only: CUDA execution provider. Off = CPU, Lite only (~3 s a frame); HR Matting and General 1536 \
               need an NVIDIA GPU or Device = Remote. macOS always runs on the CPU.",
        group: G_LOCAL,
        kind: Kind::Bool { def: true },
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
        hint: "Remember mattes keyed on the exact model input, so re-renders and matte/view tweaks are instant.",
        group: G_LOCAL,
        kind: Kind::Bool { def: true },
    },
    // ── Page 2 ──────────────────────────────────────────────────────────
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
        hint: "Give up on the server after this many seconds; the plate passes through with a node message. Raise it \
               for HR (2048) or slow networks.",
        group: G_REMOTE,
        kind: dbl(cwtools_remote::plugin::DEFAULT_TIMEOUT_S, 2.0, 120.0, 5.0, 120.0, 0),
    },
    ParamDef {
        name: P_TEST_SERVER,
        label: "Test Server",
        hint: "Ask the server for its health (VRAM, loaded models) and whether it has this node's model with the same \
               sha256. Result in the node message and /tmp/cwMatte.log.",
        group: G_REMOTE,
        kind: Kind::Button,
    },
    ParamDef { name: G_FILES, label: "Files", hint: "", group: "", kind: Kind::Group },
    ParamDef { name: "capModelDir", label: "Model Folder", hint: "", group: G_FILES, kind: Kind::Caption },
    ParamDef {
        name: P_MODEL_DIR,
        label: "Model Folder",
        hint: "Folder with the BiRefNet *_gs*.onnx exports (and their .json sidecars). Empty = the installed models folder \
               (cwtools.conf).",
        group: G_FILES,
        kind: Kind::Str { def: DEFAULT_MODEL_DIR },
    },
    ParamDef { name: "capCustomModel", label: "Custom Model", hint: "", group: G_FILES, kind: Kind::Caption },
    ParamDef {
        name: P_CUSTOM_MODEL,
        label: "Custom Model",
        hint: "Model = Custom Path: full path of a BiRefNet-style ONNX (RGB [1,3,S,S] -> matte). A same-named .json \
               sidecar sets normalisation / sigmoid; without one: ImageNet norm, sigmoid if the output is logits.",
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
];


/// Matte post-processing knobs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatteOps {
    pub clip_black: f32,
    pub clip_white: f32,
    pub gamma: f32,
    pub contrast: f32,
    pub invert: bool,
}

impl MatteOps {
    pub const IDENTITY: MatteOps = MatteOps { clip_black: 0.0, clip_white: 1.0, gamma: 1.0, contrast: 1.0, invert: false };
}

#[derive(Clone, Debug)]
pub struct Values {
    pub view: i32,
    pub colorspace: i32,
    pub ops: MatteOps,
    pub garbage: bool,
    pub roi_mode: i32,
    pub roi_pad: usize,
    /// Manual box corners (canonical / normalised param values).
    pub roi_a: (f64, f64),
    pub roi_b: (f64, f64),
    pub model: i32,
    pub model_dir: String,
    pub custom_model: String,
    pub remote: bool,
    pub use_gpu: bool,
    pub free_vram: bool,
    pub cache: bool,
    pub ort_lib: String,
    /// Remote server base URL (from Server URL or the IP / Port ints).
    pub server: String,
    /// Remote request timeout (s).
    pub timeout_s: f64,
}

fn finite_or(v: f64, d: f64) -> f32 {
    if v.is_finite() {
        v as f32
    } else {
        d as f32
    }
}

impl Values {
    pub fn defaults() -> Values {
        Values {
            view: VIEW_PLATE,
            colorspace: cwtools_core::color::CS_VIDEO,
            ops: MatteOps::IDENTITY,
            garbage: false,
            roi_mode: ROI_OFF,
            roi_pad: 64,
            roi_a: (0.25, 0.25),
            roi_b: (0.75, 0.75),
            model: MODEL_DEFAULT,
            model_dir: model_dir(DEFAULT_MODEL_DIR),
            custom_model: String::new(),
            remote: false,
            use_gpu: true,
            free_vram: true,
            cache: true,
            ort_lib: String::new(),
            server: cwtools_core::config::get().remote_server.clone(),
            timeout_s: cwtools_remote::plugin::DEFAULT_TIMEOUT_S,
        }
    }

    /// The ONNX file for the Model choice.
    pub fn model_path(&self) -> String {
        if self.model == MODEL_CUSTOM || !(0..MODELS.len() as i32).contains(&self.model) {
            return self.custom_model.clone();
        }
        let e = &MODELS[self.model as usize];
        // Remote always runs on the server's GPU: the GPU file. A GPU-only model on
        // the CPU keeps its GPU file name; gpu_only_error() stops it before loading.
        let f = if self.remote || !cpu_only(self.use_gpu) { e.gpu } else { e.cpu.unwrap_or(e.gpu) };
        std::path::Path::new(&self.model_dir).join(f).to_string_lossy().into_owned()
    }

    /// Error for a GPU-only Model running locally on the CPU (None = fine).
    pub fn gpu_only_error(&self) -> Option<String> {
        if self.remote || !cpu_only(self.use_gpu) || !(0..MODELS.len() as i32).contains(&self.model) {
            return None;
        }
        let e = &MODELS[self.model as usize];
        e.cpu.is_none().then(|| {
            format!("{} needs an NVIDIA GPU (CUDA) or Device = Remote; on the CPU use Model = Lite", e.label)
        })
    }

    /// Scale pixel-sized knobs by the render scale (proxy renders).
    pub fn scaled(&self, rs: f64) -> Values {
        let mut v = self.clone();
        if rs.is_finite() && rs > 0.0 && (rs - 1.0).abs() > 1e-9 {
            v.roi_pad = (v.roi_pad as f64 * rs).round() as usize;
        }
        v
    }
}

pub fn read_values(ps: OfxParamSetHandle, t: f64) -> Values {
    let d = |name: &str| get_double(ps, &n(name), t);
    let i = |name: &str| get_int(ps, &n(name), t);
    let b = |name: &str| get_int(ps, &n(name), t) != 0;
    let pad = d(P_ROI_PAD);
    Values {
        view: i(P_VIEW),
        colorspace: i(P_COLORSPACE),
        ops: MatteOps {
            clip_black: finite_or(d(P_CLIP_BLACK), 0.0).clamp(0.0, 1.0),
            clip_white: finite_or(d(P_CLIP_WHITE), 1.0).clamp(0.0, 1.0),
            gamma: finite_or(d(P_GAMMA), 1.0).clamp(0.05, 10.0),
            contrast: finite_or(d(P_CONTRAST), 1.0).clamp(0.0, 10.0),
            invert: i(P_MATTE_OUT) == MATTE_OUT_INVERTED,
        },
        garbage: b(P_GARBAGE),
        roi_mode: i(P_ROI_MODE),
        roi_pad: if pad.is_finite() { pad.round().clamp(0.0, 16384.0) as usize } else { 0 },
        roi_a: get_xy(ps, &n(P_ROI_A), t),
        roi_b: get_xy(ps, &n(P_ROI_B), t),
        model: i(P_MODEL),
        model_dir: model_dir(&get_string(ps, &n(P_MODEL_DIR))),
        custom_model: get_string(ps, &n(P_CUSTOM_MODEL)),
        remote: i(P_DEVICE) == 1,
        use_gpu: b(P_USE_GPU),
        free_vram: b(P_FREE_VRAM),
        cache: b(P_CACHE),
        ort_lib: get_string(ps, &n(P_ORT_LIB)),
        server: cwtools_remote::plugin::server_url(
            &get_string(ps, &n(P_SERVER)),
            &cwtools_core::config::get().remote_server,
        ),
        timeout_s: cwtools_remote::plugin::clamp_timeout(d(P_TIMEOUT)),
    }
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
        P_MODEL_DIR => Some(cfg.matte_models.clone()),
        P_ORT_LIB => Some(cwtools_core::config::shown_ort_lib()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn param_table_fits_flame_limits() {
        cwtools_core::params::validate(PARAMS).unwrap();
    }
    #[test]
    fn continuous_knobs_are_doubles() {
        for name in [P_GAMMA, P_CONTRAST, P_CLIP_BLACK, P_CLIP_WHITE, P_ROI_PAD] {
            let p = PARAMS.iter().find(|p| p.name == name).unwrap();
            assert!(matches!(p.kind, Kind::Double { .. }), "{name}");
        }
    }
    #[test]
    fn default_model_is_hr_matting() {
        let v = Values { use_gpu: true, ..Values::defaults() };
        if !cfg!(target_os = "macos") {
            assert!(v.model_path().ends_with("/BiRefNet_HR-matting-epoch_135_gs_fp16.onnx"));
            assert!(v.gpu_only_error().is_none());
        }
        // empty Model Folder = cwtools.conf matte_models; an explicit folder wins
        assert!(v.model_path().starts_with(&cwtools_core::config::get().matte_models));
        // (a folder that exists here: a missing one falls back to an installed folder)
        let tmp = std::env::temp_dir().to_string_lossy().trim_end_matches('/').to_string();
        assert_eq!(model_dir(&format!(" {tmp} ")), tmp);
        let p = PARAMS.iter().find(|p| p.name == P_MODEL).unwrap();
        match p.kind {
            Kind::Choice { opts, def } => {
                assert_eq!(def, MODEL_HR_MATTING);
                assert_eq!(opts, &["HR Matting", "General 1536", "Lite", "Custom Path"]);
                assert_eq!(opts.len(), MODELS.len() + 1);
                assert_eq!(opts[MODEL_CUSTOM as usize], "Custom Path");
                for o in opts {
                    assert!(o.len() <= 15, "{o}");
                }
            }
            _ => panic!(),
        }
        // CPU: Lite runs, the fp16-only models refuse with a pointer to Lite.
        for m in [MODEL_HR_MATTING, MODEL_GENERAL_1536] {
            let x = Values { model: m, use_gpu: false, ..v.clone() };
            assert!(x.gpu_only_error().unwrap().contains("Lite"), "{m}");
            assert!(x.model_path().ends_with("_fp16.onnx"));
        }
        let x = Values { model: MODEL_LITE, use_gpu: false, ..v.clone() };
        assert!(x.gpu_only_error().is_none());
        assert!(x.model_path().ends_with("/BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs.onnx"));
        // Remote always runs the GPU file, never refused
        let r = Values { use_gpu: false, remote: true, ..v.clone() };
        assert!(r.model_path().ends_with("/BiRefNet_HR-matting-epoch_135_gs_fp16.onnx"));
        assert!(r.gpu_only_error().is_none());
        let mut c = v.clone();
        c.model = MODEL_CUSTOM;
        c.custom_model = "/x/y.onnx".into();
        assert_eq!(c.model_path(), "/x/y.onnx");
    }

    /// The Flame panel (cwtools_core::params::grid = the probed Flame rules).
    #[test]
    fn flame_panel_layout() {
        use cwtools_core::params::{grid_pages, grid_text};
        let d = PARAMS.to_vec();
        let heads: Vec<Vec<String>> = grid_pages(&d).iter().map(|p| p.iter().map(|c| c.header.clone()).collect()).collect();
        let want: Vec<Vec<&str>> =
            vec![vec!["Main", "Output", "Input", "Matte", "ROI Box", "Local"], vec!["Remote", "Files", "ONNX Runtime"]];
        assert_eq!(heads, want, "\n{}", grid_text(&d));
    }
}
