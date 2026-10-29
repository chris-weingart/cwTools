# Your own models

cwMatte and cwRefine run any ONNX model with the right inputs and outputs. Set
**Model** to **Custom Path** and put the file's full path in **Custom Model**
(Files group). cwSegment is built around SAM3 and takes no custom models.

## cwMatte

| | |
|---|---|
| Input | one float32 tensor `[1, 3, H, W]`, RGB. A fixed size is used as is; dynamic H / W run at 1024 x 1024. |
| Output | one tensor holding the matte at the input size (`[1, 1, H, W]` or `[1, H, W]`). |
| Pre-processing | the frame (or ROI) is stretched to the input size, scaled to 0..1, then normalised. |
| Post-processing | sigmoid if the model outputs logits, then stretched back to the frame. |

Normalisation and sigmoid come from a **sidecar**: a `.json` file with the same name as the
`.onnx`, in the format Flame's Inference node uses, so a model prepared for the Inference node
works unchanged:

```json
{
  "ModelDescription": {
    "Inputs":  [{ "Name": "input_image",
                  "Normalization": { "Mean": [0.485, 0.456, 0.406],
                                     "StandardDeviation": [0.229, 0.224, 0.225] } }],
    "Outputs": [{ "Name": "output_image", "PostProcessing": "Sigmoid" }]
  }
}
```

- No sidecar: ImageNet mean / std (as above), and sigmoid only if the output leaves 0..1.
- No `Normalization` block: the model gets plain 0..1 RGB (normalisation baked into the graph).
- `PostProcessing` other than `Sigmoid`: the output is used as a matte and clamped to 0..1.

## cwRefine

| | |
|---|---|
| Input | `pixel_values`, float32 `[1, 4, h, w]`: RGB as `(x / 255 - 0.5) / 0.5`, then the trimap as 0 / 0.5 / 1. h and w are multiples of 32 and change per frame, so both must be dynamic. |
| Output | `alphas`, float32 `[1, 1, h, w]`, 0..1. |

This is the interface of the ViTMatte and MEMatte models cwRefine ships; cwRefine handles the
trimap, crop, padding and paste-back around it.

## Exporting from PyTorch

```python
torch.onnx.export(model, dummy, "my_model.onnx", opset_version=17,
                  input_names=["input_image"], output_names=["output_image"])
```

- Export float32. An fp16 copy (weights in fp16, float32 input and output) runs faster and in
  less memory on NVIDIA GPUs.
- Test the file on a frame in Flame with **Use GPU** off first: a CPU run shows whether the
  graph itself is sound before GPU memory comes into it.
- The log (`/tmp/cwMatte.log`, `/tmp/cwRefine.log`) prints the input / output names and sizes
  the plugin found, and why a model was refused.

## Remote

A custom model also runs on the [Remote server](remote-server.md): put the same file (and its
`.json`) into the server's models folder and add it to that folder's `SHA256SUMS`, as described
under "Adding a model" in [server/API.md](../server/API.md). The plugin and the server compare
sha256 checksums, so both sides must hold the identical file.
