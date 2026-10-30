# cwRefine

Turns a rough mask into a production matte with hair and soft-edge detail.
cwRefine builds a trimap from the mask, crops to the subject at native
resolution, runs a matting model on the crop and pastes the matte back.
Every step is a parameter.

<!-- screenshot -->

## Inputs

- **Source:** the plate (RGB) with the input mask in its alpha.
- **Trimap (optional):** a hand-made trimap that replaces the generated one.

## How it works

1. **Mask Prep:** the incoming mask is cleaned first: *Invert Mask*, *Black
   Clip* (faint haze to 0), *White Clip* (near-white interiors to solid),
   *Pre-Dilate* (grow tight binary masks such as SAM's).
2. **Trimap from the mask:** the prepared mask is eroded into a solid core (*Erode
   In*) and dilated into an unknown band (*Dilate Out*). That band is where
   the model works out hair and transparency. With **Band From = Mask Edges**
   (default) the band follows the mask's own softness; with **Mid Point** the
   mask is first cut at that level, so the band is Erode + Dilate wide however
   soft the mask is.
3. **Crop:** the subject plus *Padding* is cut out at native resolution. Faint
   mask detail beyond that (stray hair) is included too, so it isn't clipped at
   the crop edge; raise *Black Clip* to ignore haze. Crops longer than
   *Max Crop* are scaled down for the model and back up afterwards.
4. **Model:** the crop and trimap go through the selected model.
5. **Paste back:** the refined matte goes back into the full frame.

## Models

| Model | Notes |
|---|---|
| **ViTMatte** | ViTMatte-B, the reference model for hair. About 12.5 GB of VRAM at a 2048 crop, so 1536 is the safer choice next to Flame on a 24 GB card. |
| **MEMatte** (default) | ViT-B with routed attention: the same inputs and outputs at about a third of the memory, and a full UHD frame in one pass. Measured on a 24 GB NVIDIA GPU: 3.7 GB / 0.8 s at a 2048 crop, 6.6 GB / 1.6 s at UHD. |
| **Custom Path** | Any ONNX model with the same interface, see [Your own models](custom-models.md). |

Both models are MIT licensed upstream. Measured on a 24 GB NVIDIA GPU: 0.1 to 0.5 s
per frame, depending on the crop size.

## Useful controls

- **Result RGB / Matte Out:** the two sockets are set separately. Result RGB is
  the Plate (default), the Matte, the plate Premultiplied by it, or a
  diagnostic: Trimap (what the model sees), Crop Region, Mask vs Matte, or
  Channel Pickers (R / G / B chosen on the Result Channels page). Matte Out is
  always the refined matte (or Inverted), whatever Result RGB shows.
- **Erode In / Dilate Out:** core and band widths. Widen *Dilate Out* for
  flyaway hair.
- **Mask Prep (Invert Mask, Black Clip, White Clip, Pre-Dilate):** clean up the
  input mask before the trimap is built.
- **Use Trimap Clip:** use a connected trimap instead of the generated one.
- **Max Crop:** detail versus memory; see the models above.
- **Device:** Local (workstation GPU) or Remote (a GPU server, see
  [Remote server](remote-server.md)). Both give the same matte.
- **Free VRAM/Frame**, **Release GPU Memory:** give memory back to Flame.
- **Cache Results:** re-renders and view switches don't re-run the model.

## Tips

- Out of GPU memory: lower *Max Crop*, use **MEMatte** (the default), or use Remote.
- One loaded model is shared by every cwRefine node in the Flame session.

## Download and install

**[cwRefine for Linux](../../../releases/latest/download/cwRefine-linux.tar.gz)** ·
**[cwRefine for macOS](../../../releases/latest/download/cwRefine-macos.tar.gz)** ·
[all releases](../../../releases)

Unpack and run `./install.sh`. It asks where to install, with a default
for every answer, and adds cwRefine to an existing cwTools install.
The installer then downloads the models it needs and checks them, or uses
a [shared models folder](../README.md#models).

Prefer everything at once? Get the [full suite](../README.md#full-suite).

[← all tools](../README.md)
