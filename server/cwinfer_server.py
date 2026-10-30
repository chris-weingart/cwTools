#!/usr/bin/env python3
"""cwTools remote inference server (Remote mode of cwRefine / cwMatte / cwSegment).

Runs the SAME ONNX files the plugins run locally, with the same onnxruntime
version (1.22) and the same CUDA EP options, on the server's GPU. The plugin
keeps all pre/post-processing and sends exactly the tensors it would feed
the model locally (CWT1 container, lossless, see cwtensor.py / API.md), so
Remote == Local by construction up to GPU kernel differences.

Direct HTTP, NO authentication: clients are filtered only by CWINFER_ALLOW
(default loopback). Run it on a trusted LAN or VPN, never on the internet.

Integrity (guards against silent memory or disk corruption, e.g. on machines
without ECC RAM): every model is sha256'd at
load with its page-cache pages evicted first, and the session is built from
the exact hashed bytes; all float initializers must be finite; a canary
inference against a stored reference runs at startup and after every load;
every output is checked for NaN/Inf.

GPU etiquette (shared card): models load on demand, an LRU keeps at most
CWINFER_MAX_SESSIONS sessions, sessions idle for CWINFER_IDLE_S are released,
one inference runs at a time (FIFO), the CUDA arena shrinks after every run.

Tensor store (API 2): a request may keep outputs on the server under
client-chosen handles ("keep") and feed earlier kept tensors as inputs
("refs", concatenated along an axis). cwSegment keeps the SAM3 encoder
features and the tracker's per-frame memory here, so only the frame goes up
and small masks come back; its memory-bank bookkeeping stays in the plugin.

Config (env): CWINFER_HOST (127.0.0.1), CWINFER_PORT (8201),
CWINFER_ALLOW (127.0.0.0/8,::1/128), CWINFER_MODELS (~/cwtools_infer/models), CWINFER_CANARY
(<models>/../canary), CWINFER_IDLE_S (600), CWINFER_MAX_SESSIONS (3),
CWINFER_MAX_QUEUE (8), CWINFER_SHRINK (1), CWINFER_GPU_MEM_LIMIT_GB (48),
CWINFER_STARTUP_CANARY (vitmatte_core,BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs),
CWINFER_MAX_BODY_MB (512), CWINFER_OUT_LEVEL (3), CWINFER_LOG_DIR (<models>/..),
CWINFER_STORE_MB (4096), CWINFER_STORE_IDLE_S (1800)
"""
from __future__ import annotations

import collections
import gc
import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cwtensor as ct  # noqa: E402
import netallow  # noqa: E402

API_VERSION = 2
HOST = os.environ.get("CWINFER_HOST", "127.0.0.1")
PORT = int(os.environ.get("CWINFER_PORT", "8201"))
MODELS = Path(os.path.expanduser(os.environ.get("CWINFER_MODELS", "~/cwtools_infer/models")))
CANARY = Path(os.path.expanduser(os.environ.get("CWINFER_CANARY", str(MODELS.parent / "canary"))))
IDLE_S = float(os.environ.get("CWINFER_IDLE_S", "600"))
MAX_SESSIONS = int(os.environ.get("CWINFER_MAX_SESSIONS", "3"))
MAX_QUEUE = int(os.environ.get("CWINFER_MAX_QUEUE", "8"))
SHRINK = os.environ.get("CWINFER_SHRINK", "1") != "0"
GPU_MEM_LIMIT = int(float(os.environ.get("CWINFER_GPU_MEM_LIMIT_GB", "48")) * (1 << 30))
STARTUP_CANARY = [s for s in os.environ.get(
    "CWINFER_STARTUP_CANARY", "vitmatte_core,BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs").split(",") if s.strip()]
OUT_LEVEL = int(os.environ.get("CWINFER_OUT_LEVEL", "3"))
MAX_BODY = int(float(os.environ.get("CWINFER_MAX_BODY_MB", "512")) * (1 << 20))
STORE_MB = float(os.environ.get("CWINFER_STORE_MB", "4096"))
STORE_IDLE_S = float(os.environ.get("CWINFER_STORE_IDLE_S", "1800"))
ALLOW_SPEC = os.environ.get("CWINFER_ALLOW", netallow.DEFAULT_ALLOW)
try:
    ALLOWED_NETS = netallow.parse_allow(ALLOW_SPEC)
except ValueError as _exc:
    sys.exit(f"cwtools-infer: {_exc}; refusing to start (example: CWINFER_ALLOW=127.0.0.0/8,192.168.1.0/24)")

try:
    import setproctitle
    setproctitle.setproctitle(f"cwtools-infer-{PORT}")
except Exception:  # noqa: BLE001 — cosmetic only
    pass

LOG_DIR = Path(os.path.expanduser(os.environ.get("CWINFER_LOG_DIR", str(MODELS.parent))))
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / f"server_{PORT}.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("cwinfer")

import onnxruntime as ort  # noqa: E402

ort.set_default_logger_severity(3)  # errors only (ORT warns per conv on some fp16 graphs)
try:
    ort.preload_dlls()  # CUDA / cuDNN from the venv's nvidia-* wheels
except Exception as exc:  # noqa: BLE001 — logged; the CUDA EP check below fails loudly
    log.warning(f"ort.preload_dlls failed: {exc}")

# EP option defaults per model family = what the plugin runs locally
# (cwrefine/src/local.rs, cwmatte/src/local.rs). A request may override them
# (field "ep"); they are part of the session key.
EP_DEFAULTS = {
    "vitmatte": {"precision": "fp32", "conv_search": "heuristic", "conv_max_workspace": True, "device": "cuda"},
    # MEMatte core: same contract and Local session setup as ViTMatte (cwRefine Model menu).
    "mematte": {"precision": "fp32", "conv_search": "heuristic", "conv_max_workspace": True, "device": "cuda"},
    "birefnet": {"precision": "fp32", "conv_search": "default", "conv_max_workspace": False, "device": "cuda"},
    # cwSegment (cwsegment/src/local.rs key_for): onnxruntime's CUDA defaults. Its text
    # encoder runs on the CPU locally; the plugin asks for device "cpu" for it.
    "sam3": {"precision": "tf32", "conv_search": "exhaustive", "conv_max_workspace": True, "device": "cuda"},
}
EP_FALLBACK = {"precision": "fp32", "conv_search": "heuristic", "conv_max_workspace": True, "device": "cuda"}
_SEARCH = {"heuristic": "HEURISTIC", "default": "DEFAULT", "exhaustive": "EXHAUSTIVE"}
_ORT_TYPES = {"tensor(float)": "float32", "tensor(float16)": "float16", "tensor(uint8)": "uint8",
              "tensor(int64)": "int64", "tensor(int32)": "int32", "tensor(bool)": "bool"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, msg: str, extra: dict | None = None):
        super().__init__(msg)
        self.status, self.code, self.msg, self.extra = status, code, msg, extra or {}


# ──────────────────────────────────────────────────────────── integrity ──
def evict_page_cache(path: Path):
    """Drop the file's pages from the page cache so the next read comes from
    disk, not from a possibly corrupted cached copy."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError as exc:
        log.warning(f"page-cache eviction failed for {path}: {exc} (sha256 still guards)")


def read_verified(path: Path, want: str, attempts: int = 2) -> bytes:
    """Evict, read, sha256. Returns the exact bytes that matched `want`."""
    got = None
    for i in range(attempts):
        evict_page_cache(path)
        raw = path.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got == want:
            if i:
                log.warning(f"{path.name}: sha256 matched on attempt {i + 1} (transient flip on an earlier read)")
            return raw
        log.error(f"{path.name}: sha256 {got} != manifest {want} (attempt {i + 1})")
        del raw
    raise ApiError(500, "sha_mismatch", f"{path.name}: sha256 on disk {got} != manifest {want}; refusing to serve")


def check_initializers_finite(raw: bytes, name: str) -> int:
    import onnx
    from onnx import numpy_helper
    m = onnx.load_from_string(raw)
    n = 0
    for init in m.graph.initializer:
        if init.data_type in (1, 10, 11, 16):  # float, float16, double, bfloat16
            a = numpy_helper.to_array(init)
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                raise ApiError(500, "nonfinite_weights", f"{name}: initializer {init.name} has NaN/Inf; refusing to serve")
            n += 1
    del m
    return n


# ────────────────────────────────────────────────────────────── registry ──
class Model:
    def __init__(self, name, family, path: Path, sha, sidecar, sidecar_ok):
        self.name, self.family, self.path, self.sha = name, family, path, sha
        self.sidecar, self.sidecar_ok = sidecar, sidecar_ok
        st = path.stat()
        self.size, self.mtime = st.st_size, st.st_mtime_ns
        self.io = None          # filled lazily by describe()
        self.state = "ok" if sha else "no_manifest"
        self.state_msg = "" if sha else "file not listed in SHA256SUMS"

    def canary_path(self) -> Path | None:
        if not self.sha:
            return None
        p = CANARY / f"{self.name}.{self.sha[:16]}.npz"
        return p if p.is_file() else None

    def ep_defaults(self):
        return dict(EP_DEFAULTS.get(self.family, EP_FALLBACK))


def _read_manifest(d: Path) -> dict:
    out = {}
    p = d / "SHA256SUMS"
    if p.is_file():
        for line in p.read_text().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and len(parts[0]) == 64:
                out[parts[1].lstrip("*").strip()] = parts[0].lower()
    return out


def _graph_io(path: Path):
    """Inputs/outputs + initializer precision from the graph (no session)."""
    import collections as _c

    import onnx
    m = onnx.load(str(path), load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    types = _c.Counter(i.data_type for i in m.graph.initializer)

    def dims(t):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.type.tensor_type.shape.dim]

    tname = {1: "float32", 10: "float16", 2: "uint8", 7: "int64", 6: "int32", 9: "bool"}
    ins = [{"name": i.name, "dtype": tname.get(i.type.tensor_type.elem_type, "?"), "shape": dims(i)}
           for i in m.graph.input if i.name not in inits]
    outs = [{"name": o.name, "dtype": tname.get(o.type.tensor_type.elem_type, "?"), "shape": dims(o)}
            for o in m.graph.output]
    precision = "fp16" if types.get(10, 0) > types.get(1, 0) else "fp32"
    return {"inputs": ins, "outputs": outs, "weights": precision}


class Registry:
    """Folder scan: <MODELS>/<family>/*.onnx with SHA256SUMS + .json sidecars.
    New files appear on the next scan (every /models and /infer checks the
    directory mtimes); partially-copied files (rsync dot-temp names) are ignored."""

    def __init__(self):
        self.lock = threading.Lock()
        self.models: dict[str, Model] = {}
        self._sig = None
        self._io_cache: dict = {}

    def _signature(self):
        sig = []
        if MODELS.is_dir():
            for d in sorted(MODELS.iterdir()):
                if d.is_dir():
                    sig.append((d.name, d.stat().st_mtime_ns,
                                (d / "SHA256SUMS").stat().st_mtime_ns if (d / "SHA256SUMS").exists() else 0))
        return tuple(sig)

    def scan(self, force=False):
        with self.lock:
            sig = self._signature()
            if sig == self._sig and not force:
                return
            new = {}
            for d in sorted(p for p in MODELS.iterdir() if p.is_dir()) if MODELS.is_dir() else []:
                man = _read_manifest(d)
                for f in sorted(d.glob("*.onnx")):
                    if f.name.startswith("."):
                        continue
                    name = f.stem if f.stem not in new else f"{d.name}/{f.stem}"
                    sc_path = f.with_suffix(".json")
                    sidecar, sidecar_ok = None, None
                    if sc_path.is_file():
                        raw = sc_path.read_bytes()
                        want = man.get(sc_path.name)
                        sidecar_ok = (hashlib.sha256(raw).hexdigest() == want) if want else None
                        try:
                            sidecar = json.loads(raw)
                        except ValueError:
                            sidecar = None
                    old = self.models.get(name)
                    m = Model(name, d.name, f, man.get(f.name), sidecar, sidecar_ok)
                    if old and old.sha == m.sha and old.size == m.size and old.mtime == m.mtime:
                        m.io, m.state, m.state_msg = old.io, old.state, old.state_msg
                    new[name] = m
            added = sorted(set(new) - set(self.models))
            if added and self._sig is not None:
                log.info(f"registry: new models {added}")
            self.models = new
            self._sig = sig

    def get(self, name) -> Model:
        self.scan()
        m = self.models.get(name)
        if m is None:
            raise ApiError(404, "unknown_model", f"unknown model {name!r} (GET /models)")
        return m

    def describe(self, m: Model) -> dict:
        if m.io is None:
            key = (str(m.path), m.size, m.mtime)
            if key not in self._io_cache:
                try:
                    self._io_cache[key] = _graph_io(m.path)
                except Exception as exc:  # noqa: BLE001 — reported per model
                    self._io_cache[key] = {"error": f"{type(exc).__name__}: {exc}"}
            m.io = self._io_cache[key]
        return {
            "name": m.name, "family": m.family, "file": m.path.name, "sha256": m.sha, "size": m.size,
            "precision": m.io.get("weights"), "inputs": m.io.get("inputs"), "outputs": m.io.get("outputs"),
            "io_error": m.io.get("error"),
            "sidecar": m.sidecar, "sidecar_sha_ok": m.sidecar_ok, "ep_defaults": m.ep_defaults(),
            "canary": m.canary_path() is not None, "state": m.state, "state_msg": m.state_msg,
            "servable": m.state == "ok", "loaded": SESSIONS.is_loaded(m.name),
        }


REGISTRY = Registry()


# ──────────────────────────────────────────────────────────────── sessions ──
def norm_ep(m: Model, req_ep: dict | None) -> dict:
    ep = m.ep_defaults()
    for k, v in (req_ep or {}).items():
        if k not in ep:
            raise ApiError(400, "bad_request",
                           f"unknown ep option {k!r} (precision, conv_search, conv_max_workspace, device)")
        ep[k] = v
    if ep["precision"] not in ("fp32", "tf32"):
        raise ApiError(400, "bad_request", "ep.precision must be fp32 or tf32")
    if ep["conv_search"] not in _SEARCH:
        raise ApiError(400, "bad_request", "ep.conv_search must be heuristic, default or exhaustive")
    if ep["device"] not in ("cuda", "cpu"):
        raise ApiError(400, "bad_request", "ep.device must be cuda or cpu")
    ep["conv_max_workspace"] = bool(ep["conv_max_workspace"])
    return ep


def ep_key(ep: dict) -> str:
    if ep.get("device") == "cpu":
        return "cpu"
    return f"{ep['precision']}/{ep['conv_search']}/{'ws-max' if ep['conv_max_workspace'] else 'ws-capped'}"


class Entry:
    def __init__(self, model: Model, ep: dict, sess, load_ms: float, canary: dict | None):
        self.model, self.ep, self.sess, self.load_ms, self.canary = model, ep, sess, load_ms, canary
        self.last = time.time()
        self.runs = 0
        self.in_meta = {i.name: (_ORT_TYPES.get(i.type, i.type), i.shape) for i in sess.get_inputs()}
        self.out_names = [o.name for o in sess.get_outputs()]


def _keeper_model() -> bytes:
    """88-byte-class ONNX (one Relu on [1] float), kept alive on the CUDA EP for
    the life of the process: the cwTools plugins found that destroying the
    LAST CUDA-EP session of a process can corrupt the heap with ORT 1.22 +
    cuDNN 9; the keeper means releasing a model never triggers that teardown."""
    import onnx
    from onnx import TensorProto, helper
    g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "keeper",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return m.SerializeToString()


class Sessions:
    def __init__(self):
        self.lock = threading.Lock()        # protects `entries`
        self.gpu = threading.Lock()         # one load / inference at a time (FIFO-ish)
        self.entries: "collections.OrderedDict[tuple, Entry]" = collections.OrderedDict()
        self.waiting = 0
        self.running = None                 # description of the current job
        self.n_runs = 0
        self.n_loads = 0
        self.keeper = None

    def ensure_keeper(self):
        if self.keeper is None:
            self.keeper = ort.InferenceSession(_keeper_model(), providers=["CUDAExecutionProvider"])
            if self.keeper.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("CUDA EP not available (keeper session)")

    def is_loaded(self, name) -> bool:
        with self.lock:
            return any(k[0] == name for k in self.entries)

    def _build(self, m: Model, ep: dict) -> Entry:
        if m.state != "ok":
            # a previous sha/canary failure is re-checked on every load attempt
            log.info(f"{m.name}: previous state {m.state}, re-verifying")
        if not m.sha:
            raise ApiError(409, "no_manifest", f"{m.name}: not listed in SHA256SUMS; refusing to serve")
        t0 = time.perf_counter()
        try:
            raw = read_verified(m.path, m.sha)
            n_init = check_initializers_finite(raw, m.name)
        except ApiError as e:
            m.state, m.state_msg = e.code, e.msg
            raise
        t_hash = time.perf_counter() - t0
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.log_severity_level = 3
        cuda = {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": _SEARCH[ep["conv_search"]],
            "cudnn_conv_use_max_workspace": "1" if ep["conv_max_workspace"] else "0",
            "use_tf32": "1" if ep["precision"] == "tf32" else "0",
            "gpu_mem_limit": str(GPU_MEM_LIMIT),
        }
        on_cpu = ep.get("device") == "cpu"
        sess = ort.InferenceSession(raw, so, providers=["CPUExecutionProvider"] if on_cpu
                                    else [("CUDAExecutionProvider", cuda)])
        del raw
        if not on_cpu and sess.get_providers()[0] != "CUDAExecutionProvider":
            raise ApiError(500, "no_cuda", f"{m.name}: CUDA EP not active ({sess.get_providers()})")
        load_ms = (time.perf_counter() - t0) * 1e3
        e = Entry(m, ep, sess, load_ms, None)
        e.canary = self._canary(e)
        m.state, m.state_msg = "ok", ""
        self.n_loads += 1
        log.info(f"loaded {m.name} [{ep_key(ep)}] sha {m.sha[:12]} ok, {n_init} float initializers finite, "
                 f"hash+check {t_hash:.1f}s, total {load_ms / 1e3:.1f}s, canary {e.canary}")
        return e

    def _canary(self, e: Entry) -> dict | None:
        import canary
        p = e.model.canary_path()
        if p is None:
            return None
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        feeds = canary.inputs_for(REGISTRY.describe(e.model))
        outs = e.sess.run(None, feeds)
        worst = 0.0
        for name, a in zip(e.out_names, outs):
            if not np.isfinite(a).all():
                e.model.state, e.model.state_msg = "canary_failed", f"canary output {name} non-finite"
                raise ApiError(500, "canary_failed", f"{e.model.name}: canary output {name} non-finite")
            if f"blocks_{name}" in z.files:
                worst = max(worst, canary.compare(a, z[f"blocks_{name}"], z[f"samples_{name}"]))
        tol = float(meta["tol"])
        if worst > tol:
            e.model.state, e.model.state_msg = "canary_failed", f"canary max diff {worst:.3g} > tol {tol:.3g}"
            raise ApiError(500, "canary_failed",
                           f"{e.model.name}: canary max diff {worst:.3g} > tol {tol:.3g}; refusing to serve "
                           f"(possible memory corruption; restart the server, reboot the machine if it persists)")
        return {"max_diff": float(f"{worst:.3g}"), "tol": tol, "pass": True}

    def release(self, key=None) -> int:
        with self.lock:
            keys = [key] if key else list(self.entries)
            n = 0
            for k in keys:
                if self.entries.pop(k, None) is not None:
                    n += 1
        if n:
            gc.collect()
            log.info(f"released {n} session(s)")
        return n

    def reap(self):
        now = time.time()
        with self.lock:
            stale = [k for k, e in self.entries.items() if now - e.last > IDLE_S]
        for k in stale:
            # don't drop a session the GPU worker is using right now
            if self.gpu.acquire(timeout=0.1):
                try:
                    self.release(k)
                finally:
                    self.gpu.release()

    def get(self, m: Model, ep: dict) -> tuple[Entry, float]:
        """Caller must hold self.gpu. MAX_SESSIONS counts model FAMILIES: the graphs of
        one family (cwSegment's SAM3 encoder, detector, decoder, tracker) share a slot,
        so one tool's pipeline never evicts its own graphs."""
        key = (m.name, m.sha, ep_key(ep))
        with self.lock:
            e = self.entries.get(key)
            if e is not None:
                self.entries.move_to_end(key)
                return e, 0.0
            fams: list[str] = []  # least recently used first
            for x in self.entries.values():
                if x.model.family in fams:
                    fams.remove(x.model.family)
                fams.append(x.model.family)
            victims = []
            if m.family not in fams:
                drop = fams[:max(0, len(fams) + 1 - MAX_SESSIONS)]
                victims = [k for k, x in self.entries.items() if x.model.family in drop]
        for k in victims:
            self.release(k)
        self.ensure_keeper()
        e = self._build(m, ep)
        with self.lock:
            self.entries[key] = e
        return e, e.load_ms

    def snapshot(self):
        now = time.time()
        with self.lock:
            return [{"model": e.model.name, "family": e.model.family, "ep": ep_key(e.ep), "idle_s": round(now - e.last), "runs": e.runs,
                     "load_ms": round(e.load_ms), "canary": e.canary} for e in self.entries.values()]


SESSIONS = Sessions()


# ─────────────────────────────────────────────────────────── tensor store ──
class TensorStore:
    """Outputs kept on the server between requests, by client-chosen handle.
    LRU within CWINFER_STORE_MB, dropped after CWINFER_STORE_IDLE_S unused. Each
    entry carries a CRC-32 checked on every use: a flipped bit (non-ECC RAM) makes
    it 'missing', so the client recomputes it instead of using damaged data."""

    def __init__(self):
        self.lock = threading.Lock()
        self.items: "collections.OrderedDict[str, tuple[np.ndarray, int, float]]" = collections.OrderedDict()
        self.bytes = 0
        self.cap = int(STORE_MB * (1 << 20))
        self.hits = self.misses = self.evicted = self.corrupt = 0

    @staticmethod
    def _crc(a: np.ndarray) -> int:
        return zlib.crc32(memoryview(a).cast("B"))

    def put(self, handle: str, a: np.ndarray):
        if not isinstance(handle, str) or not 0 < len(handle) <= 200:
            raise ApiError(400, "bad_request", "keep handles are strings of 1-200 characters")
        a = np.ascontiguousarray(a)
        if a.nbytes > self.cap:
            raise ApiError(413, "too_large", f"output of {a.nbytes >> 20} MB exceeds CWINFER_STORE_MB")
        crc = self._crc(a)
        with self.lock:
            old = self.items.pop(handle, None)
            if old is not None:
                self.bytes -= old[0].nbytes
            self.items[handle] = (a, crc, time.time())
            self.bytes += a.nbytes
            while self.bytes > self.cap and len(self.items) > 1:
                _, (b, _, _) = self.items.popitem(last=False)
                self.bytes -= b.nbytes
                self.evicted += 1

    def get(self, handles: list) -> list[np.ndarray]:
        out, missing = [], []
        with self.lock:
            for h in handles:
                it = self.items.get(h) if isinstance(h, str) else None
                if it is None:
                    missing.append(h)
                    continue
                self.items.move_to_end(h)
                self.items[h] = (it[0], it[1], time.time())
                out.append(it)
        bad = [h for h, it in zip([h for h in handles if h not in missing], out) if self._crc(it[0]) != it[1]]
        if bad:
            log.error(f"tensor store: CRC mismatch on {bad} (memory fault?); dropped")
            with self.lock:
                for h in bad:
                    it = self.items.pop(h, None)
                    if it is not None:
                        self.bytes -= it[0].nbytes
                self.corrupt += len(bad)
            missing += bad
        if missing:
            self.misses += len(missing)
            raise ApiError(409, "ref_missing", f"{len(missing)} kept tensor(s) no longer on the server "
                                               f"(evicted, expired or restarted): {missing[:4]}",
                           {"missing": missing})
        self.hits += len(out)
        return [it[0] for it in out]

    def forget(self, prefix: str | None = None, handles: list | None = None) -> int:
        with self.lock:
            ks = [k for k in self.items if (prefix is not None and k.startswith(prefix)) or
                  (handles is not None and k in handles)]
            for k in ks:
                self.bytes -= self.items.pop(k)[0].nbytes
        return len(ks)

    def reap(self):
        now = time.time()
        with self.lock:
            ks = [k for k, it in self.items.items() if now - it[2] > STORE_IDLE_S]
            for k in ks:
                self.bytes -= self.items.pop(k)[0].nbytes
        if ks:
            log.info(f"tensor store: {len(ks)} idle tensor(s) dropped")

    def snapshot(self) -> dict:
        with self.lock:
            return {"tensors": len(self.items), "mb": round(self.bytes / (1 << 20), 1), "cap_mb": STORE_MB,
                    "idle_drop_s": STORE_IDLE_S, "hits": self.hits, "misses": self.misses,
                    "evicted": self.evicted, "crc_failures": self.corrupt}


STORE = TensorStore()


def _decode_input(d: dict, area: memoryview) -> np.ndarray:
    """A CWT1 tensor, or kept tensors ("refs": [handles], concatenated along "axis", default 0)."""
    refs = d.get("refs")
    if refs is None:
        return ct.decode_tensor(d, area)
    if not isinstance(refs, list) or not refs:
        raise ApiError(400, "bad_tensor", f"input {d.get('name')}: refs must be a non-empty list of handles")
    arrs = STORE.get(refs)
    if len(arrs) == 1:
        return arrs[0]
    try:
        return np.concatenate(arrs, axis=int(d.get("axis", 0)))
    except ValueError as exc:
        raise ApiError(400, "bad_tensor", f"input {d.get('name')}: cannot concatenate refs: {exc}") from exc


def _is_oom(msg: str) -> bool:
    s = msg.lower()
    return any(t in s for t in ("failed to allocate memory", "out of memory", "cudaerrormemoryallocation",
                                "cuda_error_out_of_memory", "bfc_arena"))


def infer(header: dict, area: memoryview) -> tuple[dict, bytes]:
    t0 = time.perf_counter()
    name = header.get("model")
    if not isinstance(name, str):
        raise ApiError(400, "bad_request", "header.model (string) required")
    m = REGISTRY.get(name)
    want_sha = str(header.get("sha256") or "").lower()
    if not want_sha:
        raise ApiError(400, "bad_request", "header.sha256 required (the client's model file sha256)")
    if want_sha != m.sha:
        raise ApiError(409, "sha_mismatch",
                       f"{name}: client sha256 {want_sha[:16]}… != server {str(m.sha)[:16]}… (different model file)")
    ep = norm_ep(m, header.get("ep"))
    try:
        feeds = {}
        for d in header.get("inputs") or []:
            feeds[d["name"]] = _decode_input(d, area)
    except ct.CodecError as exc:
        raise ApiError(400, "bad_tensor", str(exc)) from exc
    except (KeyError, TypeError) as exc:
        raise ApiError(400, "bad_tensor", f"malformed tensor descriptor: {exc}") from exc
    t_dec = time.perf_counter()
    want_out = header.get("outputs")
    keep = header.get("keep") or {}
    if not isinstance(keep, dict):
        raise ApiError(400, "bad_request", "keep must be an object {output name: handle}")
    out_dtype = header.get("output_dtype")
    if out_dtype not in (None, "float32", "float16"):
        raise ApiError(400, "bad_request", "output_dtype must be float32 (default) or float16 (lossy)")
    quant = header.get("output_quantize")
    if quant not in (None, "u8_trunc255"):
        raise ApiError(400, "bad_request", "output_quantize must be u8_trunc255 (or absent)")
    if quant and out_dtype:
        raise ApiError(400, "bad_request", "output_quantize and output_dtype are exclusive")
    out_codec = header.get("output_codec", "xor-shuf-zstd")
    if out_codec not in ("raw", "zstd", "xor-shuf-zstd"):
        raise ApiError(400, "bad_request", "output_codec must be raw, zstd or xor-shuf-zstd")

    with SESSIONS.lock:
        if SESSIONS.waiting >= MAX_QUEUE:
            raise ApiError(503, "busy", f"queue full ({SESSIONS.waiting} waiting); retry later")
        SESSIONS.waiting += 1
    try:
        SESSIONS.gpu.acquire()
    finally:
        with SESSIONS.lock:
            SESSIONS.waiting -= 1
    t_q = time.perf_counter()
    try:
        SESSIONS.running = f"{name} {[list(a.shape) for a in feeds.values()]}"
        e, load_ms = SESSIONS.get(m, ep)
        # validate feeds against the session
        if set(feeds) != set(e.in_meta):
            raise ApiError(400, "bad_inputs", f"inputs {sorted(feeds)} != model inputs {sorted(e.in_meta)}")
        for k, a in feeds.items():
            dt, shape = e.in_meta[k]
            if str(a.dtype) != dt:
                raise ApiError(400, "bad_inputs", f"input {k}: dtype {a.dtype}, model wants {dt}")
            if len(shape) != a.ndim or any(isinstance(s, int) and s != n for s, n in zip(shape, a.shape)):
                raise ApiError(400, "bad_inputs", f"input {k}: shape {list(a.shape)}, model wants {shape}")
        names = list(e.out_names if want_out is None else want_out)
        run_names = names + [k for k in keep if k not in names]
        bad = [n for n in run_names if n not in e.out_names]
        if bad:
            raise ApiError(400, "bad_request", f"unknown outputs {bad}; model has {e.out_names}")
        ro = ort.RunOptions()
        if SHRINK and ep.get("device") != "cpu":
            ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
        tr = time.perf_counter()
        try:
            outs = e.sess.run(run_names, feeds, ro)
        except Exception as exc:  # noqa: BLE001 — classified below
            msg = str(exc)
            SESSIONS.release((m.name, m.sha, ep_key(ep)))
            if _is_oom(msg):
                raise ApiError(503, "gpu_oom", f"GPU out of memory on the server for {name} "
                                               f"{[list(a.shape) for a in feeds.values()]}; model released, "
                                               f"retry or use a smaller crop / model") from exc
            raise ApiError(500, "run_failed", f"{name}: {msg[:300]}") from exc
        run_ms = (time.perf_counter() - tr) * 1e3
        e.runs += 1
        e.last = time.time()
        SESSIONS.n_runs += 1
        for n, a in zip(run_names, outs):
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                # suspicious: drop the session so the next load re-hashes + re-canaries
                SESSIONS.release((m.name, m.sha, ep_key(ep)))
                log.error(f"{name}: non-finite output {n} ({int((~np.isfinite(a)).sum())} values); session dropped")
                raise ApiError(500, "nonfinite_output",
                               f"{name}: output {n} contains NaN/Inf (GPU/memory fault?); session dropped, retry")
    finally:
        SESSIONS.running = None
        SESSIONS.gpu.release()
    t_run = time.perf_counter()
    by_name = dict(zip(run_names, outs))
    for n, h in keep.items():
        STORE.put(h, by_name[n])
    outs = [by_name[n] for n in names]
    try:
        lvl = int(header.get("output_level", OUT_LEVEL))
    except (TypeError, ValueError):
        raise ApiError(400, "bad_request", "output_level must be an int") from None
    b = ct.Builder(level=max(1, min(lvl, 19)))
    descs = []
    for n, a in zip(names, outs):
        if out_dtype == "float16" and a.dtype == np.float32:
            a = a.astype(np.float16)
        if quant == "u8_trunc255" and a.dtype == np.float32 and a.ndim >= 2:
            # = cwRefine recipe::alpha_to_u8 / production (alpha * 255).clip(0, 255).astype(uint8)
            a = np.clip(a * np.float32(255.0), np.float32(0.0), np.float32(255.0)).astype(np.uint8)
            descs.append(b.tensor(n, a, "grad-zstd"))
            continue
        descs.append(b.tensor(n, a, out_codec if a.dtype.kind == "f" or out_codec != "xor-shuf-zstd" else "zstd"))
    t_enc = time.perf_counter()
    res = {
        "ok": True, "api": API_VERSION, "model": name, "sha256": m.sha, "ep": ep, "outputs": descs,
        "timings_ms": {"decode": round((t_dec - t0) * 1e3, 1), "queue": round((t_q - t_dec) * 1e3, 1),
                       "load": round(load_ms, 1), "run": round(run_ms, 1),
                       "encode": round((t_enc - t_run) * 1e3, 1), "total": round((t_enc - t0) * 1e3, 1)},
    }
    if keep:
        res["kept"] = sorted(keep.values())
    log.info(f"infer {name} {[list(a.shape) for a in feeds.values()]} -> {[d['shape'] for d in descs]} "
             f"{f'kept {len(keep)} ' if keep else ''}t={res['timings_ms']}")
    return res, b.pack(res)


def fetch(header: dict) -> bytes:
    """Kept tensors back to the client, optionally only some indices along an axis
    (cwSegment: the detector's masks of the queries above its threshold)."""
    reqs = header.get("tensors")
    if not isinstance(reqs, list) or not reqs:
        raise ApiError(400, "bad_request", "tensors: [{ref, axis?, index?}] required")
    arrs = STORE.get([r.get("ref") for r in reqs])
    b = ct.Builder(level=OUT_LEVEL)
    descs = []
    for r, a in zip(reqs, arrs):
        if r.get("index") is not None:
            try:
                a = np.take(a, np.asarray(r["index"], np.int64), axis=int(r.get("axis", 0)))
            except (IndexError, ValueError, TypeError) as exc:
                raise ApiError(400, "bad_request", f"fetch {r.get('ref')}: {exc}") from exc
        descs.append(b.tensor(str(r.get("name") or r["ref"]), np.ascontiguousarray(a),
                              "xor-shuf-zstd" if a.dtype.kind == "f" else "zstd"))
    return b.pack({"ok": True, "api": API_VERSION, "outputs": descs})


# ──────────────────────────────────────────────────────────────── health ──
_NVML = None


def gpu_info() -> dict:
    global _NVML
    try:
        import pynvml
        if _NVML is None:
            pynvml.nvmlInit()
            _NVML = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML)
        mine = 0
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(_NVML):
            if p.pid == os.getpid():
                mine = (p.usedGpuMemory or 0)
        return {"gpu_used_mb": mem.used >> 20, "gpu_total_mb": mem.total >> 20, "gpu_free_mb": mem.free >> 20,
                "process_mb": mine >> 20}
    except Exception as exc:  # noqa: BLE001 — health stays up without NVML
        return {"error": f"nvml: {exc}"}


T_START = time.time()
STARTUP = {}


def health() -> dict:
    REGISTRY.scan()
    return {
        "ok": True, "api": API_VERSION, "service": "cwtools-infer", "uptime_s": round(time.time() - T_START),
        "onnxruntime": ort.__version__, "providers": ort.get_available_providers(),
        "vram": gpu_info(), "sessions": SESSIONS.snapshot(),
        "queue": {"running": SESSIONS.running, "waiting": SESSIONS.waiting, "max_waiting": MAX_QUEUE},
        "stats": {"runs": SESSIONS.n_runs, "loads": SESSIONS.n_loads},
        "settings": {"idle_release_s": IDLE_S, "max_sessions": MAX_SESSIONS, "arena_shrink": SHRINK,
                     "gpu_mem_limit_gb": GPU_MEM_LIMIT / (1 << 30)},
        "models": len(REGISTRY.models), "startup_canary": STARTUP,
        "features": ["refs", "keep", "fetch", "device"], "store": STORE.snapshot(),
    }


# ────────────────────────────────────────────────────────────────── HTTP ──
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "cwtools-infer/1"

    def log_message(self, fmt, *args):  # quiet default access log
        pass

    def _allowed(self) -> bool:
        return netallow.is_allowed(self.client_address[0], ALLOWED_NETS)

    def _forbidden(self):
        return self._err(ApiError(403, "forbidden", f"client {self.client_address[0]} not in CWINFER_ALLOW"))

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: dict):
        self._send(status, json.dumps(obj, separators=(",", ":")).encode(), "application/json")

    def _err(self, e: ApiError):
        extra = {"Retry-After": "2"} if e.status == 503 else None
        body = json.dumps({"ok": False, "code": e.code, "error": e.msg, **e.extra}).encode()
        self._send(e.status, body, "application/json", extra)

    def do_GET(self):
        if not self._allowed():
            return self._forbidden()
        try:
            if self.path.rstrip("/") == "/health":
                return self._json(200, health())
            if self.path.rstrip("/") == "/models":
                REGISTRY.scan()
                return self._json(200, {"ok": True, "api": API_VERSION,
                                        "models": [REGISTRY.describe(m) for m in REGISTRY.models.values()]})
            return self._err(ApiError(404, "not_found", f"no route {self.path}"))
        except Exception as exc:  # noqa: BLE001 — never kill the handler thread silently
            log.exception("GET failed")
            return self._err(ApiError(500, "internal", f"{type(exc).__name__}: {exc}"))

    def do_POST(self):
        if not self._allowed():
            return self._forbidden()
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            self.close_connection = True
            return self._err(ApiError(413, "too_large", f"body of {n} bytes (limit {MAX_BODY})"))
        body = self.rfile.read(n) if n else b""
        route = self.path.rstrip("/")
        try:
            if route == "/infer":
                header, area = ct.unpack(body)
                res, out = infer(header, area)
                return self._send(200, out, ct.CONTENT_TYPE)
            if route == "/release":
                with SESSIONS.gpu:
                    k = SESSIONS.release()
                return self._json(200, {"ok": True, "released": k})
            if route == "/fetch":
                header, _ = ct.unpack(body)
                return self._send(200, fetch(header), ct.CONTENT_TYPE)
            if route == "/forget":
                try:
                    j = json.loads(body or b"{}")
                except ValueError:
                    raise ApiError(400, "bad_request", "body must be JSON {prefix} or {handles}") from None
                prefix, handles = j.get("prefix"), j.get("handles")
                if (prefix is None) == (handles is None) or (prefix is not None and not prefix):
                    raise ApiError(400, "bad_request", "give exactly one of prefix (non-empty) or handles")
                return self._json(200, {"ok": True, "forgotten": STORE.forget(prefix, handles)})
            if route == "/rescan":
                REGISTRY.scan(force=True)
                return self._json(200, {"ok": True, "models": len(REGISTRY.models)})
            return self._err(ApiError(404, "not_found", f"no route {self.path}"))
        except ct.CodecError as exc:
            return self._err(ApiError(400, "bad_container", str(exc)))
        except ApiError as e:
            log.warning(f"{route}: {e.status} {e.code}: {e.msg}")
            return self._err(e)
        except Exception as exc:  # noqa: BLE001 — reported to the client, logged with traceback
            log.error(f"{route} failed: {traceback.format_exc()}")
            return self._err(ApiError(500, "internal", f"{type(exc).__name__}: {exc}"))


def _reaper():
    while True:
        time.sleep(min(30.0, max(1.0, IDLE_S / 4)))
        try:
            SESSIONS.reap()
            STORE.reap()
        except Exception:  # noqa: BLE001 — logged, the loop keeps running
            log.exception("reaper")


def startup_canary():
    """Load each startup model, run its canary, release it. Any failure = exit."""
    for name in STARTUP_CANARY:
        m = REGISTRY.get(name.strip())
        if m.canary_path() is None:
            raise RuntimeError(f"startup canary: no reference for {m.name} in {CANARY}")
        with SESSIONS.gpu:
            e, _ = SESSIONS.get(m, m.ep_defaults())
            STARTUP[m.name] = e.canary
            SESSIONS.release()


def main():
    log.info(f"cwtools-infer starting on {HOST}:{PORT}, models {MODELS}, onnxruntime {ort.__version__}, "
             f"idle {IDLE_S}s, max sessions {MAX_SESSIONS}")
    log.info(f"allowed clients (CWINFER_ALLOW): {', '.join(str(n) for n in ALLOWED_NETS)}; "
             f"NO authentication - the allow-list is the only gate")
    if not netallow.host_is_loopback(HOST) and netallow.loopback_only(ALLOWED_NETS):
        log.warning(f"!!! listening on {HOST} but CWINFER_ALLOW is loopback-only: every network client "
                    f"gets 403. Add your network, e.g. CWINFER_ALLOW=127.0.0.0/8,192.168.1.0/24")
    if any(n.prefixlen == 0 for n in ALLOWED_NETS):
        log.warning("!!! CWINFER_ALLOW contains 0.0.0.0/0 or ::/0: ANY client can use this GPU (no auth). "
                    "Only acceptable on an isolated network.")
    REGISTRY.scan(force=True)
    log.info(f"registry: {len(REGISTRY.models)} models: {sorted(REGISTRY.models)}")
    try:
        SESSIONS.ensure_keeper()
        startup_canary()
    except Exception:
        log.exception("startup integrity check FAILED; refusing to serve")
        sys.exit(2)
    threading.Thread(target=_reaper, daemon=True, name="reaper").start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    log.info(f"serving on http://{HOST}:{PORT} (startup canary {STARTUP})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
