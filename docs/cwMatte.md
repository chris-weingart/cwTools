# cwMatte

Automatic matte of the main subject with no prompts. Plug in a plate and get
a matte, using BiRefNet models.

<!-- screenshot -->

## Models

| Model | Input size | VRAM (approx.) | Use |
|---|---|---|---|
| **HR Matting** (default) | 2048 | 12+ GB | best edges and hair on every kind of subject. Next to Flame on a 24 GB card use General 1536 or Device = Remote |
| **General 1536** | 1536 | ~7 GB | second best, fits next to Flame on a 24 GB card |
| **Lite** | 1024 | 2–3 GB | small and fast; the CPU and macOS model. Fine on solid subjects, weak on glass and fine detail |
| Custom Path | any | | your own ONNX, see [Your own models](custom-models.md) |

HR Matting and General 1536 run in fp16 on an NVIDIA GPU (or on the Remote server); on the
CPU, which includes every Mac, use Lite (about 3 s a frame).

How they compare on the AIM-500 matting test set (mean error per image, lower is better):

| | People | Animals | Objects | Plants | Glass / transparent |
|---|---|---|---|---|---|
| HR Matting | 4.1 | 4.4 | 6.3 | 14.2 | 32.4 |
| General 1536 | 5.3 | 6.4 | 7.7 | 16.4 | 37.3 |
| Lite | 5.9 | 8.9 | 9.2 | 23.1 | 80.5 |

BiRefNet is MIT licensed upstream (github.com/ZhengPeng7/BiRefNet).

## Useful controls

- **ROI:** crop to a region first (the bounding box of the Source alpha, or a
  manual box) so the subject fills the model's input. This gives a big gain
  on wide shots.
- **Result RGB / Matte Out:** the two sockets are set separately. Result RGB
  is the Plate (default), the Matte, an Overlay, or the plate Premultiplied by
  the matte; Matte Out is the Matte or the Inverted (background) matte, whatever
  Result RGB shows.
- **Clip Black / Clip White, Gamma, Contrast:** shape the matte (thicker
  hair, solid core). The model's near-zero floor (below 1/4096) is always cut to
  exactly 0, so no faint haze survives a grade; Clip Black is for visible haze.
- **Source Alpha = Garbage:** use the incoming alpha as a garbage mask.
- **Device:** Local or Remote ([Remote server](remote-server.md)).

## Tips

- For hair-level detail, feed cwMatte's result into [cwRefine](cwRefine.md).
- **One object or one person out of several:** cwMatte mattes the most salient subject, so
  narrow down what it sees first. Make a mask of the object with [cwSegment](cwSegment.md)
  (e.g. `person`, Output = Object Channels, the object in one channel), put that channel in the
  alpha of the plate, and feed it to cwMatte with **ROI = Source Alpha BBox** (the object then
  fills the model's input) and **Source Alpha = Garbage** (everything outside the object is
  removed). Blur or grow the mask a little first: the garbage step multiplies by it, so hair
  beyond a tight mask edge would be cut. Then refine with cwRefine.

## Download and install

**[cwMatte for Linux](../../../releases/latest/download/cwMatte-linux.tar.gz)** ·
**[cwMatte for macOS](../../../releases/latest/download/cwMatte-macos.tar.gz)** ·
[all releases](../../../releases)

Unpack and run `./install.sh`. It asks where to install, with a default
for every answer, and adds cwMatte to an existing cwTools install.
The installer then downloads the models it needs and checks them, or uses
a [shared models folder](../README.md#models).

Prefer everything at once? Get the [full suite](../README.md#full-suite).

[← all tools](../README.md)
