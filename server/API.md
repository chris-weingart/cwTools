# cwtools-infer: remote inference server (cwSegment / cwRefine / cwMatte Remote mode)

cwtools-infer runs the models of cwSegment, cwRefine and cwMatte on a separate GPU machine, so
a Flame workstation can segment, track and refine without spending its own VRAM. Set a node's
**Device** to **Remote** and point `remote_server` in `cwtools.conf` (or the node's Server URL)
at `http://<server>:<port>`.

- Install: `install/install.sh --server` (Linux + NVIDIA GPU). It creates the venv, copies the
  models, builds the integrity references, writes `cwtools-infer.env` and a systemd unit.
- Default address: `http://127.0.0.1:8201`. Clients on other machines need both
  `CWINFER_HOST` (listen address) and `CWINFER_ALLOW` (which clients may connect), see below.

## Security

- **There is no authentication.** Anyone who can reach the port and is inside `CWINFER_ALLOW`
  can run models on the GPU. The allow-list is the only gate.
- Run it on a trusted studio LAN or a VPN. **Never expose it to the internet** (no port
  forwarding, no public IP).
- Default: listens on `127.0.0.1` and accepts loopback clients only. To serve the studio,
  listen on the machine's LAN address and allow just your subnet, e.g.
  `CWINFER_HOST=192.168.1.50` and `CWINFER_ALLOW=127.0.0.0/8,192.168.1.0/24`.
- Plain HTTP: requests are not encrypted. Inputs are image crops of your plates; use a VPN if
  the network between workstation and server is not yours.

## Configuration

Environment variables (the installer writes them to `cwtools-infer.env`; restart the unit after
editing):

| variable | default | meaning |
|---|---|---|
| `CWINFER_HOST` | `127.0.0.1` | listen address (`0.0.0.0` = all interfaces) |
| `CWINFER_PORT` | `8201` | port |
| `CWINFER_ALLOW` | `127.0.0.0/8,::1/128` | comma-separated client networks (CIDR, IPv4/IPv6; a bare address = one host). Others get 403. An invalid entry stops the server at startup. |
| `CWINFER_MODELS` | `~/cwtools_infer/models` | model folder (`<family>/*.onnx` + `SHA256SUMS`) |
| `CWINFER_CANARY` | `<models>/../canary` | canary references (`make_canary.py`) |
| `CWINFER_LOG_DIR` | `<models>/..` | log file `server_<port>.log` |
| `CWINFER_STARTUP_CANARY` | `vitmatte_core,BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs` | models loaded and canary-checked at startup (empty = none) |
| `CWINFER_IDLE_S` | `600` | release a model after this many idle seconds |
| `CWINFER_MAX_SESSIONS` | `3` | model families loaded at once (least recently used goes first; the graphs of one family, e.g. cwSegment's SAM3 graphs, share a slot). 3 = SAM3 + one BiRefNet + ViTMatte/MEMatte, the full cwSegment → cwMatte → cwRefine chain without reloads (measured on a 24 GB GPU: ~3.5 GB between frames, ~8.5 GB peak during a BiRefNet HR 2048 run); lower it on GPUs too small for that |
| `CWINFER_MAX_QUEUE` | `8` | waiting requests before 503 `busy` |
| `CWINFER_SHRINK` | `1` | shrink the CUDA arena after every run (`0` = off) |
| `CWINFER_GPU_MEM_LIMIT_GB` | `48` | per-session CUDA arena cap |
| `CWINFER_MAX_BODY_MB` | `512` | request size limit |
| `CWINFER_OUT_LEVEL` | `3` | default zstd level for outputs |
| `CWINFER_STORE_MB` | `4096` | RAM for kept tensors (tensor store, API 2); least recently used go first |
| `CWINFER_STORE_IDLE_S` | `1800` | drop a kept tensor after this many seconds unused |

At startup the server logs the address, the allow-list and a warning when it listens on a
network address but only loopback clients are allowed (nobody could connect), or when the
allow-list is open to everyone.

## Idea

The server runs the **same ONNX files** the plugins run locally, with **onnxruntime 1.22.0**
(the version Flame 2027.2 bundles), cuDNN 9.9 / cuBLAS 12.8 (Flame's versions), and the
**same CUDA EP options**. The plugin keeps all pre- and post-processing. It sends exactly the
tensor(s) it would feed the model locally and gets the raw model output back. Remote equals
Local by construction, apart from GPU kernel differences between the two GPUs.

Each request names the model **and its sha256**. If the server's file has a different hash
you get HTTP 409 `sha_mismatch`, never a silently different result.

## Endpoints

| method | path | what |
|---|---|---|
| GET | `/health` | status, VRAM, loaded sessions, queue, startup canary result |
| GET | `/models` | every model in the folder: name, sha256, inputs/outputs (name, dtype, shape), weight precision, sidecar JSON, EP defaults, state |
| POST | `/infer` | run one model on CWT1 tensors (or kept tensors), returns CWT1 tensors |
| POST | `/fetch` | download kept tensors, optionally only some indices along an axis (API 2) |
| POST | `/forget` | drop kept tensors by handle prefix or list (API 2) |
| POST | `/release` | drop all loaded sessions now (frees VRAM) |
| POST | `/rescan` | force a model-folder rescan (it also happens automatically) |

### `GET /models`

```json
{"ok":true,"api":2,"models":[
  {"name":"vitmatte_core","family":"vitmatte","file":"vitmatte_core.onnx","sha256":"ea4484…",
   "size":387731564,"precision":"fp32",
   "inputs":[{"name":"pixel_values","dtype":"float32","shape":[1,4,"h","w"]}],
   "outputs":[{"name":"alphas","dtype":"float32","shape":[1,1,"h","w"]}],
   "sidecar":null,"sidecar_sha_ok":null,
   "ep_defaults":{"precision":"fp32","conv_search":"heuristic","conv_max_workspace":true},
   "canary":true,"state":"ok","servable":true,"loaded":false}, …]}
```

- Model **name = file stem**. Folder layout: `<models>/<family>/*.onnx` + `SHA256SUMS` +
  optional `<stem>.json` sidecar (Flame Inference format; cwMatte reads normalisation and
  sigmoid from it). New files show up by themselves (the folder is rescanned when its
  mtime changes). Files whose names start with `.` (rsync temp files) are ignored.
- A file that is not in its folder's `SHA256SUMS` is listed with `state:"no_manifest"` and is
  not served.
- `state`: `ok` | `no_manifest` | `sha_mismatch` | `nonfinite_weights` | `canary_failed`.
  A failed state is re-checked on the next load attempt.
- Dynamic dims are strings (`"h"`, `"w"`). ViTMatte core needs multiples of 32.

### `GET /health`

```json
{"ok":true,"api":2,"service":"cwtools-infer","uptime_s":33,"onnxruntime":"1.22.0",
 "vram":{"gpu_used_mb":12145,"gpu_total_mb":97887,"gpu_free_mb":85742,"process_mb":610},
 "sessions":[{"model":"vitmatte_core","ep":"fp32/heuristic/ws-max","idle_s":12,"runs":40,
              "load_ms":1712,"canary":{"max_diff":1.8e-07,"tol":0.002,"pass":true}}],
 "queue":{"running":null,"waiting":0,"max_waiting":8},
 "settings":{"idle_release_s":600,"max_sessions":2,"arena_shrink":true,"gpu_mem_limit_gb":48},
 "startup_canary":{"vitmatte_core":{…},"BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs":{…}}, "stats":{"runs":0,"loads":2}}
```

`process_mb` is this server's VRAM. With no session loaded it is about 0.6 GB (CUDA context plus
a tiny "keeper" session).

## `POST /infer`

```
POST /infer
Content-Type: application/x-cwtensor
<CWT1 body>
```

Response: `200` + CWT1 body (`Content-Type: application/x-cwtensor`), or a JSON error.

### CWT1 container

```
b"CWT1" | u32 little-endian header length | header JSON (UTF-8) | blob area
```

Tensors in the header refer to byte ranges of the blob area as `[offset, length]`.

Request header:

```json
{"model":"vitmatte_core",
 "sha256":"ea44846492f8b7733eb0cec5ab3f6faaa6996b58854fc8116125a9e06562fd08",
 "inputs":[{"name":"pixel_values","dtype":"float32","shape":[1,4,800,800],
            "codec":"lut8-webp","lut":[0,4096],"webp":[4096,512345],"data":[516441,70000],
            "crc32":123456789}],
 "ep":{"precision":"fp32","conv_search":"heuristic","conv_max_workspace":true},
 "outputs":["alphas"],
 "output_dtype":"float32",
 "output_codec":"xor-shuf-zstd"}
```

| field | notes |
|---|---|
| `model` | required. Name from `/models`. |
| `sha256` | required. sha256 of the model file the plugin would run locally. Mismatch → 409. |
| `inputs` | required. One tensor per model input (names, dtypes and static dims must match). |
| `ep` | optional. CUDA EP options, part of the session key. Default = `ep_defaults` of the model's family, which equal what the plugins use locally: ViTMatte `fp32/heuristic/max workspace` (cwRefine "GPU Math TF32" → `precision:"tf32"`), BiRefNet `fp32/default/capped workspace`, SAM3 `tf32/exhaustive/max workspace` (onnxruntime's CUDA defaults). `device:"cpu"` (API 2) runs the model on the server's CPU, as cwSegment runs its text encoder locally. |
| `keep` | optional (API 2). `{output name: handle}`: keep these outputs in the tensor store for later requests. Only `outputs` are downloaded, so `"outputs": []` with `keep` downloads nothing. |
| `outputs` | optional. Subset of output names (default all). |
| `output_dtype` | optional. `float32` (default, native, lossless) or `float16` (**lossy**, halves the download). |
| `output_codec` | optional. `xor-shuf-zstd` (default), `zstd`, `raw`. |
| `output_quantize` | optional. `u8_trunc255`: float outputs come back as uint8 `clip(x * 255, 0, 255)` truncated, codec `grad-zstd`. That is exactly the 8-bit production recipe's `alpha_to_u8` (`astype(uint8)`), ~20× smaller (800×800 alphas: 575 KB → 29 KB); used by the parity tooling. The nodes ask for native float (cwRefine keeps sub-1/255 hair levels; cwMatte needs the float logits). |
| `output_level` | optional. zstd level for outputs (default 3; 9 / 19 save only 4-8%). |

Response header: `{"ok":true,"api":2,"model":…,"sha256":…,"ep":{…},"outputs":[tensors],
"timings_ms":{"decode","queue","load","run","encode","total"}}`. `load` is non-zero when this
request had to load the model (first use or after the idle release).

### Tensor store (API 2)

An input may name kept tensors instead of carrying data:

```json
{"name": "mem_feats", "refs": ["h1", "h2", "h3"], "axis": 0}
```

One handle is that tensor; several are concatenated along `axis` (default 0). Handles are the
client's strings (1-200 characters); cwSegment prefixes them per Flame process, e.g.
`cws1a2b/e/7/image_embed` (encoder features of a frame) and `cws1a2b/t/<node>/<session>/<frame>/<object>`
(tracker memory). Every kept tensor carries a CRC-32 checked on every use: a mismatch (memory
fault) drops it and is reported like a missing one. A missing handle (evicted, idle-dropped,
server restarted) answers HTTP 409 `ref_missing` with `"missing": [handles]`, and the client
recomputes it (encoder features) or asks the artist to start over (tracking memory).

`POST /fetch` (CWT1 body, header only): `{"tensors": [{"ref": h, "name": n, "axis": 1, "index": [3, 17]}]}`
returns the kept tensors (`index` = only those entries along `axis`), CWT1 like `/infer`.
cwSegment keeps the detector's 200 masks and fetches only those above Detect Thresh.

`POST /forget` (JSON): `{"prefix": "cws1a2b/t/4/9/"}` or `{"handles": [...]}` → `{"ok":true,"forgotten":n}`.

`GET /health` (API 2) adds `"features": ["refs","keep","fetch","device"]` and `"store"`
(tensors, MB, hits, misses, evictions, CRC failures).

### Codecs (all lossless)

| codec | blobs | use |
|---|---|---|
| `raw` | `data` | little-endian bytes as-is |
| `grad-zstd` | `data` | uint8 tensors: 2-D gradient filter (last two axes, mod 256), zstd. Used for `u8_trunc255` outputs. |
| `zstd` | `data` | zstd of the raw bytes (frames carry a content checksum) |
| `xor-shuf-zstd` | `data` | float tensors: each element's bit pattern XOR its left neighbour (last axis), byte-shuffle, zstd. Default for outputs. |
| `lut8-grad-zstd` | `lut`, `data` | tensors whose every channel (axis 1) is a function of an 8-bit value. `lut` = C×256 values (tensor dtype, raw). `data` = uint8 indices, optional `xform:"gdiff"` (channels 0 and 2 minus channel 1, mod 256, needs C≥3), then a 2-D gradient filter per H×W plane (`e = v − left − up + upleft`, mod 256), then zstd. Value = `lut[c][idx]`, bit-exact. |
| `lut8-webp` | `lut`, `webp`, `data` | as above, shape `[1,C≥3,H,W]`: index channels 0-2 as ONE lossless WebP RGB image, channels 3.. (if any) as gradient + zstd in `data`. |

`crc32` (optional, recommended for lut8): CRC-32 of all uint8 indices (NCHW). The server
rejects a mismatch with 400 `bad_tensor`. WebP has no checksum of its own.

Why this design: both plugins' model inputs are per-channel functions of 8-bit display RGB
(ViTMatte: HF normalisation of the crop plus the trimap; BiRefNet: ImageNet normalisation of
the stretched square). The client builds the table **from the tensor it already computed**
(`TensorIn::lut8_from_source`, every element checked), so nothing is re-implemented on the
server and parity holds by construction. Using the real pixel values as indices keeps the
planes natural images: lossless WebP (method 0) is about 20% smaller than gradient + zstd.

Measured upload sizes: ViTMatte 800×800 crop: float32 10.2 MB raw → 593 KB
(`lut8-webp`) / 700 KB (`lut8-grad-zstd`).

### Errors

JSON `{"ok":false,"code":"…","error":"…"}`:

| HTTP | code | meaning |
|---|---|---|
| 400 | `bad_request`, `bad_container`, `bad_tensor`, `bad_inputs` | malformed request / tensors, wrong names / dtypes / shapes, crc32 mismatch |
| 403 | `forbidden` | client address not in `CWINFER_ALLOW` |
| 404 | `unknown_model`, `not_found` | |
| 409 | `sha_mismatch` | client and server have different model files (or `no_manifest`) |
| 409 | `ref_missing` | a kept tensor is gone (`missing` lists the handles) |
| 413 | `too_large` | body larger than `CWINFER_MAX_BODY_MB` |
| 500 | `sha_mismatch` (server-side: file on disk ≠ manifest), `nonfinite_weights`, `canary_failed`, `nonfinite_output`, `run_failed`, `no_cuda` | integrity / runtime failures; the model is not served |
| 503 | `busy` (queue full), `gpu_oom` | retry later (`Retry-After: 2`); on OOM the session is released |

## Integrity

Guards against silent corruption (bit flips in RAM without ECC, bad disks, damaged copies):


- Every load: the model file's page-cache pages are evicted (`posix_fadvise DONTNEED`), the
  file is read, sha256'd and compared with `SHA256SUMS` (one retry, logged), and the session is
  built **from those exact bytes**. All float initializers must be finite.
- Canary: `canary/<name>.<sha16>.npz` holds output fingerprints (32×32 block means + 4096 sampled
  values) of a synthetic input (`canary.py`). Made by `make_canary.py`: CUDA reference, a second
  fresh CUDA session for the run-to-run spread, CPU EP as an independent cross-check. Tolerance =
  max(10 × spread, 2e-3 × p99 |output|). It runs after every load; at startup for
  `vitmatte_core` and `BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs` (then released). Startup failure = exit 2,
  per-model failure = that model is refused.
- Every output is checked for NaN/Inf. Non-finite → 500 and the session is dropped, so the next
  request re-hashes and re-canaries.
- The client also rejects non-finite outputs and a response whose sha256 differs.

## GPU etiquette

- Models load on demand. At most `CWINFER_MAX_SESSIONS` (3) model families; the least recently used
  is released first. A session idle for `CWINFER_IDLE_S` (600 s) is released.
- One load / inference at a time (FIFO lock). Up to `CWINFER_MAX_QUEUE` (8) requests wait; more → 503 `busy`.
- The CUDA arena shrinks back after every run (`memory.enable_memory_arena_shrinkage`), so VRAM
  between requests is only weights + context. Per-session arena cap `CWINFER_GPU_MEM_LIMIT_GB` (48).
- Never touches other services.

## Running it

The installer sets up a systemd unit (system unit by default, `cwtools-infer`):

```
sudo systemctl status cwtools-infer        # or: systemctl --user … for a user unit
sudo systemctl restart cwtools-infer       # after editing cwtools-infer.env
journalctl -u cwtools-infer -f             # plus <log dir>/server_<port>.log
curl http://127.0.0.1:8201/health
```

By hand: `env CWINFER_HOST=… CWINFER_ALLOW=… venv/bin/python server/cwinfer_server.py`.

Install layout: `<root>/{server/, venv/, models/{vitmatte,mematte,birefnet,sam3}/, canary/, logs/, cwtools-infer.env}`.
venv: Python ≥ 3.10, onnxruntime-gpu 1.22.0, nvidia-cudnn-cu12 9.x, nvidia-cublas-cu12 12.x,
numpy, zstandard, onnx, pillow, nvidia-ml-py (see `server/requirements.txt`).

Adding a model: copy `<stem>.onnx` (+ `.json`) into `models/<family>/`, add its line to that
folder's `SHA256SUMS` (use the published checksum, not one computed on the server, so a copy
that was damaged on the way is caught), run `venv/bin/python server/make_canary.py <stem>`
(optional but recommended; graphs without image-like inputs, such as the SAM3 decoder,
detector and tracker, have no canary recipe and are skipped). It is served on the next request.
Families with plugin-matching EP defaults: `vitmatte`, `mematte` (cwRefine cores), `birefnet`,
`sam3` (cwSegment).

Test a running server: `venv/bin/python server/client_test.py http://<host>:<port>`.

## Client

Rust: crate `cwtools-remote` (`crates/cwtools-remote`): `infer(server, &Request)`,
`TensorIn::lut8_from_source` / `f32_auto` / `refs` / `i64`, `Request::keep`, `fetch`, `forget`,
`Ep::vitmatte(tf32)` / `Ep::birefnet()` / `Ep::sam3()` / `Ep::sam3_cpu()`,
`model_sha256(path)` (reads the `SHA256SUMS` next to the model, else hashes the file),
`get_json(server, "/health" | "/models")`. Test / parity tool: `crates/cwremote-parity`
(`cwremote refine|matte|health|models`).
Python: `server/cwtensor.py` (`Builder`, `unpack`, `decode_tensor`).
