# cwTools changelog

One version for cwSegment, cwMatte and cwRefine (shared core and Remote client).
Each node shows it in Flame as the read-only **Version** parameter. While the
version is 0.x only the minor number moves, because Flame matches saved setups on
the major version. cwFill has its own version (tags `cwfill-vX.Y.Z`).

## 0.3.1 (2026-10-30)

**cwSegment Remote** runs the same SAM3 models on the cwtools-infer server,
the one cwMatte and cwRefine use. Masks and tracking match Local (identical
on the same GPU in our tests), and tracking now works Remote too. Each frame
is sent once, losslessly; the server keeps its features and the tracking
memory, so only masks come back. The Server URL defaults to `remote_server`;
the separate SAM3 server, the `segment_server` setting and Send Size are gone
(Remote uses the same 1008 px image as Local). Timeout goes up to 120 s, and
Test Server checks that the server has the same SAM3 files.

**cwtools-infer** (server API 2): a tensor store keeps model outputs on the
server between requests (`keep`, `refs`, `/fetch`, `/forget`, checked with a
CRC on every use, limited by `CWINFER_STORE_MB` and `CWINFER_STORE_IDLE_S`);
a model can run on the server's CPU (`ep.device`); `CWINFER_MAX_SESSIONS`
counts model families, so one tool's graphs share a slot, and defaults to 3
(was 2), so a cwSegment → cwMatte → cwRefine tree no longer reloads a model
on every frame. `--server` installs
the SAM3 models by default (about 2.5 GB).

**Bit depth:** every node now outputs the Source clip's bit depth (8-bit,
16-bit integer, 16-bit half or 32-bit float) instead of always 32-bit float.
Flame gives all clips of an OFX node one depth; the nodes now ask for the
Source's. Non-float images are converted to float for processing and back on
output (integers clamped to 0..1). cwFill already kept half plates as half;
its 8-bit and 16-bit integer sources came out as float.

**cwFill 0.3.2** (own version): the bit-depth fix above; nothing else changed.

Installer: one Remote server line for all three tools; a 0.3.0 server is
flagged when cwSegment is chosen. The cwSegment tokenizer (`vocab.json`,
`merges.txt`) is its own 1.4 MB model set, `segment-sam3-tokenizer`, installed
with cwSegment in every models mode: text prompts now work on Remote-only
workstations (`--models skip`), which before got the checksums but not the
tokenizer.

## 0.3.0 (2026-10-29): first public release

All nodes: the Server URL, Model Folder and ONNX Runtime fields show the values
in effect (cwtools.conf / auto-detect) instead of starting empty.

Installer: shows the whole plan first (Enter installs, a line number changes
that line); finds model archives next to the package and in Downloads; checks
disk space, whether the servers answer and what they are; asks again after a
wrong answer; line editing; installs the cwExtract Matchbox with cwSegment;
writes install.log. Device = Remote no longer needs the model on the
workstation (every model's checksum is recorded at install).

**cwSegment**: SAM3 object masks from text prompts.
- Comma-separated concepts, several objects at once, Best Instance Only, Pick
  R / G / B / Matte in the viewer.
- Outputs: Object Channels (default; up to four objects in R, G, B, Matte;
  channels follow Concept # 1-4 by default; the plate until an object is found), Mask, ID Map, Object IDs (object n = n x 0.02, up to 50)
  plus the **cwExtract** Matchbox that turns IDs back into mattes, Overlay.
- Auto Concepts: an optional vision language model (any OpenAI-compatible
  endpoint, local Ollama by default) suggests the concepts.
- Local: SAM3 in ONNX through Flame's ONNX Runtime, fp16 encoder on the GPU
  (24 GB NVIDIA GPU: ~4 GB peak, ~0.45 s a frame with Free VRAM/Frame).

- Track (Local): follows picked objects through the shot with the SAM3 tracker
  (ONNX, the model set segment-sam3-track). Pick Objects in the viewer, then
  Track Range reads the frames (a dialog says how long Flame pauses) and tracks
  them in the background while Flame stays usable (24 GB NVIDIA GPU: about 0.4 s a
  frame). Track From / To and messages use Batch frame numbers (Batch Start,
  default 1001); untracked frames say so in the viewer.

**cwMatte**: automatic subject matte with BiRefNet.
- HR Matting (default), General 1536 (fits next to Flame on 24 GB), Lite (CPU and
  macOS), Custom Path. HR Matting and 1536 run fp16 on the GPU.
- ROI from the Source alpha or a box, matte shaping, Source Alpha = Garbage.
- Result RGB (Plate, Matte, Overlay, Premultiplied) and Matte Out (Matte,
  Inverted) are chosen separately; the model's sub-1/4096 haze is cut to 0.

**cwRefine**: rough mask to hair-detail matte.
- MEMatte (default; about a third of ViTMatte's memory, full UHD in one pass) and
  ViTMatte, or any core ONNX via Custom Path.
- Trimap from the mask (Band From Mask Edges / Mid Point, Erode In, Dilate Out) or
  from a Trimap clip; crop at native resolution with Max Crop; faint hair beyond
  the crop padding is kept inside the crop.
- Diagnostic outputs and Result RGB channel pickers.
- Result RGB (Plate, Matte, Premultiplied, Trimap, Crop Region, Mask vs Matte,
  Channel Pickers) and Matte Out (Matte, Inverted) are chosen separately: the
  diagnostic views never replace the matte. Mask Prep column (before Trimap).
- The refined matte stays float end to end (Local and Remote): no 8-bit step,
  faint hair keeps its sub-1/255 levels.

**cwFill** (own version, 0.3.1): foreground estimation and despill.
- Edge Tone (Despill column, -1..1): pulls edge fill toward the nearby solid
  foreground where a bright background brightened it or a dark one darkened
  it; negative pushes it further. Reaches far flyaway wisps; Tone Spread
  carries the correction outward (no ring at the core). Output-only, no
  re-solve.
- Limit to Soft Matte is on by default.
- 16-bit half in gives 16-bit out (it used to come out 32-bit).
- Picking a Screen Colour switches Despill Screen to Custom Colour.

**All three**
- Input Colour converts ACEScg / ACEScct / Linear Rec.709 through the ACES 2.0 SDR
  view (baked from Flame's aces2.0 config) before the model.
- Device Local (Flame's ONNX Runtime, CUDA or CPU) or Remote (the cwtools-infer
  GPU server, same ONNX and settings, model files checked by sha256).
- One loaded model shared by all nodes of a kind, Free VRAM/Frame, Release GPU
  Memory, result cache.
- `cwtools.conf` install-wide settings (models folder, Remote servers, ONNX
  Runtime library); ONNX Runtime found automatically.
- Same panel layout everywhere: Main column (main control, Device, Release GPU
  Memory), daily controls on page 1, setup after.

**Install**
- `install.sh` for Linux and macOS: per tool or the whole suite, merges into an
  existing install, `--check`, `--uninstall`.
- Models as separate downloads, verified by sha256, into a local folder or a
  shared studio folder that every workstation reads.
- `install.sh --server` sets up the Remote server (systemd, model checks, canary
  inference). Clients are limited by an allow-list (loopback by default); there is
  no authentication, so keep it on a trusted network.

**Known limits**
- Tested on Flame 2027.2 on Rocky Linux with a 24 GB NVIDIA GPU and on macOS (Apple
  silicon). macOS runs the models on the CPU: HR Matting and General 1536 need
  Device = Remote there, and tracking takes seconds per frame.
- cwSegment Remote needs a SAM3 server, which is not part of this release.
- Without Track, objects are numbered per frame. Track runs Local only.
