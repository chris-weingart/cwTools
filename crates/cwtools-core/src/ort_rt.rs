//! onnxruntime for cwTools: Flame's bundled `libonnxruntime` 1.22 loaded
//! dynamically (ort `load-dynamic`, crate pinned to rc.10 = ORT 1.22 C API),
//! found automatically ([`select_ort`]: the copy already loaded in the
//! process, else the running Flame's, else the newest under
//! `/opt/Autodesk/lib64`), CUDA/cuDNN preloaded from the same folder (Linux),
//! and a PROCESS-WIDE session cache:
//!
//! - one session per [`ModelKey`] (model path + execution provider + GPU
//!   precision), shared by every node instance of the plugin;
//! - loads are serialised (two nodes asking for the same model at once load
//!   it once);
//! - [`release_all`] / [`release_path`] drop sessions so their CUDA arena is
//!   returned to the driver (in-flight renders keep their `Arc` until done).

use ort::execution_providers::cuda::CuDNNConvAlgorithmSearch;
use ort::execution_providers::{ArenaExtendStrategy, CUDAExecutionProvider};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub use crate::ort_find::*;

static ORT_INIT: OnceLock<Result<String, String>> = OnceLock::new();

extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flag: std::os::raw::c_int) -> *mut std::os::raw::c_void;
}

/// Preload the CUDA/cuDNN libraries that libonnxruntime_providers_cuda.so
/// needs, by absolute path from the ORT library's own directory (Flame's
/// lib64/<version>). The provider has no RPATH, so without this it only
/// resolves if LD_LIBRARY_PATH happens to contain that dir. Libraries already
/// loaded by Flame are simply re-referenced. Best effort: failures are logged.
/// Linux only (no CUDA on macOS).
pub fn preload_cuda_deps(ort_lib: &str) {
    if cfg!(target_os = "macos") {
        return;
    }
    let Some(dir) = std::path::Path::new(ort_lib).parent() else { return };
    const LIBS: &[&str] = &[
        "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcurand.so.10", "libcufft.so.11",
        "libnvrtc.so.12", "libcudnn_graph.so.9", "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9", "libcudnn_heuristic.so.9", "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9", "libcudnn_adv.so.9", "libcudnn.so.9",
    ];
    let mut missing = Vec::new();
    for l in LIBS {
        let p = dir.join(l);
        if !p.exists() {
            missing.push(*l);
            continue;
        }
        let c = std::ffi::CString::new(p.to_string_lossy().as_bytes()).unwrap_or_default();
        // RTLD_NOW | RTLD_GLOBAL
        let h = unsafe { dlopen(c.as_ptr(), 0x2 | 0x100) };
        if h.is_null() {
            missing.push(*l);
        }
    }
    if !missing.is_empty() {
        crate::log::log(&format!("local: could not preload {missing:?} from {} (CUDA EP may fail)", dir.display()));
    }
}

/// Load onnxruntime once per process (first caller wins). `lib` is the
/// "ONNX Runtime" param: a path, or "" / "auto" = cwtools.conf `ort_lib` /
/// auto-detect ([`resolve_ort_lib`]). Returns the library loaded.
pub fn init_ort(lib: &str) -> Result<String, String> {
    ORT_INIT
        .get_or_init(|| {
            let (lib, why) = resolve_ort_lib(lib).map_err(|e| {
                crate::log::log(&format!("ort: {e}"));
                e
            })?;
            crate::log::log(&format!("ort: using {lib} ({why})"));
            if !Path::new(&lib).exists() {
                return Err(format!("onnxruntime library not found: {lib}"));
            }
            if !is_ort_122(&Path::new(&lib).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()) {
                crate::log::log(&format!("ort: WARNING {lib} does not look like onnxruntime 1.22 (the only version this build supports)"));
            }
            preload_cuda_deps(&lib);
            // Must happen before any other ort call; only once per process.
            let l = lib.clone();
            let r = std::panic::catch_unwind(move || ort::init_from(&l).with_name(crate::log::name()).commit());
            match r {
                Ok(Ok(_)) => {
                    release_env_at_exit();
                    Ok(lib)
                }
                Ok(Err(e)) => Err(format!("onnxruntime init failed: {e}")),
                Err(_) => Err(format!("onnxruntime init panicked (wrong version? need 1.22): {lib}")),
            }
        })
        .clone()
}

/// macOS: release ONNX Runtime's environment from an exit handler. Otherwise ORT's own
/// static env is destroyed during exit after its logging mutex, and the host aborts on
/// quit (`libc++abi: mutex lock failed`, Mac Flame 2027.2 + render_host). Registered after
/// ORT's statics, so it runs before them. Linux keeps its working teardown (CUDA keeper).
#[cfg(target_os = "macos")]
fn release_env_at_exit() {
    extern "C" fn at_exit() {
        release_all();
        if let Ok(env) = ort::environment::get_environment() {
            use ort::AsPointer;
            unsafe { (ort::api().ReleaseEnv)(env.ptr() as *mut _) };
        }
    }
    unsafe extern "C" {
        fn atexit(cb: extern "C" fn()) -> std::os::raw::c_int;
    }
    unsafe { atexit(at_exit) };
}

#[cfg(not(target_os = "macos"))]
fn release_env_at_exit() {}

/// Build a session with default CUDA EP options (falls back to CPU). This is
/// the exact cwSegment 0.1 behaviour (plus the CUDA keeper); new code should
/// use [`shared_session`].
pub fn build_session(path: &std::path::Path, gpu: bool) -> Result<(Session, bool), String> {
    let base = || -> Result<ort::session::builder::SessionBuilder, String> {
        Session::builder()
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .map_err(|e| format!("session builder: {e}"))
    };
    if gpu {
        // Same heap-corruption guard as the shared cache: never let a dropped
        // session be the process's last CUDA-EP session.
        ensure_cuda_keeper();
        let b = base()?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
            .map_err(|e| format!("CUDA EP: {e}"));
        match b.and_then(|b| b.commit_from_file(path).map_err(|e| format!("load {}: {e}", path.display()))) {
            Ok(s) => return Ok((s, true)),
            Err(e) => crate::log::log(&format!("local: CUDA session failed, falling back to CPU: {e}")),
        }
    }
    let s = base()?
        .commit_from_file(path)
        .map_err(|e| format!("load {}: {e}", path.display()))?;
    Ok((s, false))
}

// ---------------------------------------------------------------------------
// Process-wide session cache
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ep {
    Cuda,
    Cpu,
}

/// GPU math precision for an fp32 model on the CUDA EP.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    /// Full fp32 GEMMs (`use_tf32=0`): closest to the PyTorch reference.
    Fp32,
    /// TF32 tensor-core GEMMs (Ampere+): faster, ~1e-3 relative error.
    Tf32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelKey {
    pub path: String,
    pub ep: Ep,
    pub precision: Precision,
    /// cuDNN `cudnn_conv_use_max_workspace` (ORT default true). With it, the
    /// heuristic may pick conv algorithms that want a multi-GB workspace
    /// (BiRefNet's decoder asks for 13 GB at 1024 on a 3090); false caps the
    /// workspace and makes big-conv models fit next to Flame.
    pub conv_max_workspace: bool,
    /// cuDNN conv algorithm search. NOTE: ORT's Heuristic mode takes cuDNN's
    /// first-ranked algorithm whatever its workspace (the cap above is only
    /// honoured by Exhaustive); for static-shape models Exhaustive benchmarks
    /// once per session within the cap.
    pub conv_search: ConvSearch,
}

/// cuDNN convolution algorithm selection (`cudnn_conv_algo_search`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConvSearch {
    /// Instant pick, any workspace. Right for dynamic shapes (cwRefine crops).
    Heuristic,
    /// Benchmark every algorithm once per shape, within the workspace cap.
    Exhaustive,
    /// IMPLICIT_PRECOMP_GEMM, no search, small workspace.
    Default,
}

impl ModelKey {
    pub fn describe(&self) -> String {
        let ws = if self.conv_max_workspace { "" } else { "/ws-capped" };
        let cs = if self.conv_search == ConvSearch::Heuristic { String::new() } else { format!("/{:?}", self.conv_search) };
        format!("{} [{:?}/{:?}{ws}{cs}]", self.path, self.ep, self.precision)
    }
}

pub struct SharedSession {
    pub key: ModelKey,
    /// `run` needs `&mut Session`; renders using the same model serialise here.
    pub session: Mutex<Session>,
    /// False when CUDA was requested but the session fell back to CPU.
    pub on_gpu: bool,
    pub load_ms: f64,
}

static CACHE: Mutex<Vec<Arc<SharedSession>>> = Mutex::new(Vec::new());

/// 88-byte ONNX model (one Relu on a [1] float) used to keep a CUDA EP alive.
pub const CUDA_KEEPER_ONNX: &[u8] = include_bytes!("../assets/cuda_keeper.onnx");
static CUDA_KEEPER: OnceLock<Option<Mutex<Session>>> = OnceLock::new();

/// Keep one tiny CUDA-EP session alive for the life of the process (never
/// dropped: statics are not destructed at exit).
///
/// Why: with Flame's ORT 1.22 + cuDNN 9, destroying the LAST CUDA-EP session
/// of the process corrupts the heap (glibc aborts in cuDNN's exit handlers,
/// ~50% of runs on an RTX 3090 Linux workstation; never with a second CUDA session alive,
/// never on the CPU EP). With the keeper alive, releasing a model session
/// still frees its weights + arena but never triggers that teardown.
fn ensure_cuda_keeper() {
    CUDA_KEEPER.get_or_init(|| {
        if std::env::var("CWTOOLS_NO_CUDA_KEEPER").map(|v| v == "1").unwrap_or(false) {
            // Diagnostics only (keeper regression test): reproduce the teardown abort.
            return None;
        }
        let r = Session::builder()
            .and_then(|b| b.with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()]))
            .and_then(|b| b.commit_from_memory(CUDA_KEEPER_ONNX));
        match r {
            Ok(s) => Some(Mutex::new(s)),
            Err(e) => {
                crate::log::log(&format!("ort: CUDA keeper session failed (release may be unsafe): {e}"));
                None
            }
        }
    });
}
static LOADS: AtomicUsize = AtomicUsize::new(0);

/// Number of sessions created by [`shared_session`] since process start (tests).
pub fn load_count() -> usize {
    LOADS.load(Ordering::Relaxed)
}

/// Keys of the sessions currently held by the cache.
pub fn cached_keys() -> Vec<String> {
    CACHE.lock().map(|g| g.iter().map(|s| s.key.describe()).collect()).unwrap_or_default()
}

fn cuda_ep(key: &ModelKey) -> ort::execution_providers::ExecutionProviderDispatch {
    let search = match key.conv_search {
        ConvSearch::Heuristic => CuDNNConvAlgorithmSearch::Heuristic,
        ConvSearch::Exhaustive => CuDNNConvAlgorithmSearch::Exhaustive,
        ConvSearch::Default => CuDNNConvAlgorithmSearch::Default,
    };
    CUDAExecutionProvider::default()
        .with_device_id(0)
        // Grow the arena by what is requested, not by powers of two: crops
        // change size every frame and a doubling arena balloons VRAM.
        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
        // Dynamic H/W (cwRefine): EXHAUSTIVE would re-benchmark convolutions
        // for every new crop size (seconds per frame); the heuristic picks
        // instantly. Static-shape models may choose otherwise (ModelKey).
        .with_conv_algorithm_search(search)
        .with_tf32(key.precision == Precision::Tf32)
        .with_conv_max_workspace(key.conv_max_workspace)
        .build()
        .error_on_failure()
}

fn load(key: &ModelKey) -> Result<SharedSession, String> {
    let path = std::path::Path::new(&key.path);
    if !path.is_file() {
        return Err(format!("model file not found: {}", key.path));
    }
    // Diagnostics (VRAM experiments): CWTOOLS_ORT_OPT=0..3 graph optimisation
    // level, CWTOOLS_ORT_MEMPATTERN=0 disables the memory pattern planner.
    let opt = || match std::env::var("CWTOOLS_ORT_OPT").ok().as_deref() {
        Some("0") => GraphOptimizationLevel::Disable,
        Some("1") => GraphOptimizationLevel::Level1,
        Some("2") => GraphOptimizationLevel::Level2,
        _ => GraphOptimizationLevel::Level3,
    };
    let mem_pattern = std::env::var("CWTOOLS_ORT_MEMPATTERN").map(|v| v != "0").unwrap_or(true);
    let base = || -> Result<ort::session::builder::SessionBuilder, String> {
        Session::builder()
            .and_then(|b| b.with_optimization_level(opt()))
            .and_then(|b| b.with_memory_pattern(mem_pattern))
            .map_err(|e| format!("session builder: {e}"))
    };
    let t0 = std::time::Instant::now();
    let mut on_gpu = false;
    let mut sess = None;
    if key.ep == Ep::Cuda {
        ensure_cuda_keeper();
        let r = base()?
            .with_execution_providers([cuda_ep(key)])
            .map_err(|e| format!("CUDA EP: {e}"))
            .and_then(|b| b.commit_from_file(path).map_err(|e| format!("load {}: {e}", key.path)));
        match r {
            Ok(s) => {
                sess = Some(s);
                on_gpu = true;
            }
            Err(e) => crate::log::log(&format!("ort: CUDA session failed, falling back to CPU: {e}")),
        }
    }
    let session = match sess {
        Some(s) => s,
        None => base()?.commit_from_file(path).map_err(|e| format!("load {}: {e}", key.path))?,
    };
    LOADS.fetch_add(1, Ordering::Relaxed);
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    crate::log::log(&format!("ort: loaded {} in {:.0} ms (gpu={on_gpu})", key.describe(), load_ms));
    Ok(SharedSession { key: key.clone(), session: Mutex::new(session), on_gpu, load_ms })
}

/// Get the shared session for `key`, loading it on first use. `ort_lib` is
/// the onnxruntime library to initialise with (first call per process wins).
pub fn shared_session(ort_lib: &str, key: &ModelKey) -> Result<Arc<SharedSession>, String> {
    init_ort(ort_lib)?;
    // Holding the cache lock during a load serialises loads: a second node
    // asking for the same model waits and then finds it cached.
    let mut g = CACHE.lock().map_err(|_| "session cache lock poisoned".to_string())?;
    if let Some(s) = g.iter().find(|s| &s.key == key) {
        return Ok(s.clone());
    }
    let s = Arc::new(load(key)?);
    g.push(s.clone());
    Ok(s)
}

/// Drop every cached session. Returns how many were released. Sessions still
/// used by an in-flight render are freed when that render finishes.
pub fn release_all() -> usize {
    let mut g = match CACHE.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    let n = g.len();
    g.clear();
    if n > 0 {
        crate::log::log(&format!("ort: released {n} session(s)"));
    }
    n
}

/// Drop cached sessions whose model path is `path`.
pub fn release_path(path: &str) -> usize {
    let mut g = match CACHE.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    let before = g.len();
    g.retain(|s| s.key.path != path);
    before - g.len()
}

/// Is a session for `key` currently cached?
pub fn is_cached(key: &ModelKey) -> bool {
    CACHE.lock().map(|g| g.iter().any(|s| &s.key == key)).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real machine check (Flame workstations): prints what auto would pick.
    #[test]
    fn real_lib64_if_present() {
        let lib64 = Path::new(AUTODESK_LIB64);
        if lib64.is_dir() {
            let r = select_ort(&loaded_libraries(), None, lib64);
            eprintln!("auto-detect on this machine: {r:?}");
            // macOS: loading Flame's dylib works, but its exit-time teardown aborts the
            // test process ("mutex lock failed") after all tests passed -> opt-in there.
            let load = cfg!(target_os = "linux") || std::env::var_os("CWTOOLS_TEST_ORT_LOAD").is_some();
            if load && r.is_ok() && std::env::var_os(crate::config::ENV_VAR).is_none() {
                // "" = cwtools.conf ort_lib / auto: load it for real and run a CPU session.
                let lib = init_ort("").unwrap();
                eprintln!("loaded {lib}");
                // now it is in the process: step (1) finds this very copy
                let again = select_ort(&loaded_libraries(), None, Path::new("/nonexistent")).unwrap();
                assert_eq!(std::fs::canonicalize(&again.0).unwrap(), std::fs::canonicalize(&lib).unwrap());
                assert_eq!(again.1, "already loaded in this process");
                Session::builder().and_then(|b| b.commit_from_memory(CUDA_KEEPER_ONNX)).unwrap();
            }
        }
    }
}
