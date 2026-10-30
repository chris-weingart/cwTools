<h1 align="center">cwTools</h1>

<p align="center"><b>ML matting and foreground tools for Autodesk Flame.</b><br>
Segment and refine mattes, then fix the foreground's edge colour, right in Batch,
as native OpenFX nodes with full parameter control.<br>
Runs on the workstation, or optionally on a <a href="#remote-gpu-server-optional">shared GPU server</a> on your network.</p>

<!-- screenshot / GIF of a Batch tree: mask -> refine -> clean -->

## Three steps

From plate to composite-ready edge.

Each tool installs on its own, so take only the ones you need, or get the
[full suite](#full-suite): all four tools and the Remote server in one package.

<table>
<tr>
<td rowspan="2" width="120"><b>1</b>&ensp;Mask</td>
<td><a href="docs/cwSegment.md"><b>cwSegment</b></a></td>
<td>Object masks from a prompt: text, points or a box. Many objects at once.</td>
</tr>
<tr>
<td><a href="docs/cwMatte.md"><b>cwMatte</b></a></td>
<td>Automatic matte of the main subject, no prompt needed.</td>
</tr>
<tr>
<td width="120"><b>2</b>&ensp;Refine</td>
<td><a href="docs/cwRefine.md"><b>cwRefine</b></a></td>
<td>Rough mask to production matte with fine detail.</td>
</tr>
<tr>
<td width="120"><b>3</b>&ensp;Foreground</td>
<td><a href="docs/cwFill.md"><b>cwFill</b></a></td>
<td>True foreground colour in the soft edge, plus despill. No model needed, runs in any OFX host.</td>
</tr>
</table>

Click a tool for its page and download.

## Requirements

- Autodesk Flame 2027.2 (tested). cwSegment, cwRefine and cwMatte use the
  ONNX Runtime 1.22 that ships with Flame, so 2026.2 and 2027.0 should work
  but are untested. cwFill runs in Flame 2026.2.2 and later and in other
  OFX hosts.
- Linux with an NVIDIA GPU for fast local inference (tested on Rocky Linux
  with a 24 GB NVIDIA GPU). macOS runs the models on the CPU, or
  use a Remote server.

## Install

Download a package, unpack it and run:

```
./install.sh
```

The installer shows its plan first: plugins, folders (default: the system OFX
folder), where the models come from (it finds model archives next to the
package or in Downloads), the Remote server and whether it answers, and the
cwExtract Matchbox. Press Enter to install, or type a line number to change
that line. Each install writes a log (`install.log` in the cwTools folder). Installing another tool later adds it to the existing install.
`./install.sh --check` verifies an install, `./install.sh --uninstall`
removes it, `./install.sh --help` lists every option.

## Models

The plugin packages are small; the models are separate downloads, one per
model set, published with the release and checked by sha256. The installer
fetches the sets the chosen tools need (`./install.sh --list-models` shows
them with sizes and licences). A full install of the ML tools downloads about
4.5 GB on Linux and 5.4 GB on macOS (CPU versions of the SAM3 models).

- **Offline:** download the `cwtools-models-*.tar.gz` files yourself and run
  `./install.sh --models from:/path/to/folder`.
- **Studio share:** keep one models folder on a network share and point every
  workstation at it with `./install.sh --models shared:/mnt/share/cwTools/models`.
  The installer checks the files there and copies nothing; if a set is missing
  and the share is writable it offers to download it there once.
  `./install.sh --check` re-verifies the shared files. Each Flame session reads
  a model from the share once when it first loads it (0.2 to 1.5 GB per model).

## Your own models

cwMatte and cwRefine also run your own ONNX models (Model = Custom Path), including models
prepared for Flame's Inference node. See [Your own models](docs/custom-models.md).

## Remote GPU server (optional)

cwSegment, cwRefine and cwMatte can send the model steps to a separate Linux
GPU machine, so the large models work next to Flame and on workstations
without a big GPU, such as Macs. cwSegment tracks Remote too. Local and Remote
give the same matte.
See [Remote server](docs/remote-server.md).

## Full suite

All four tools plus the Remote server files in one package (models are
downloaded by the installer as above):

**[cwTools for Linux](../../releases/latest/download/cwTools-linux.tar.gz)** ·
**[cwTools for macOS](../../releases/latest/download/cwTools-macos.tar.gz)** ·
[all releases](../../releases)

---

[Changelog](CHANGELOG.md) · [Building from source](docs/BUILDING.md) · Licence: [Apache 2.0](LICENSE).
The models have their own licences: [third-party notices](THIRD_PARTY_NOTICES.txt) and [licence texts](licenses/).
