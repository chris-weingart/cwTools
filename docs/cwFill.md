<p><picture><source media="(prefers-color-scheme: light)" srcset="img/cwFill-light.svg"><img src="img/cwFill.svg" alt="cwFill" height="56"></picture></p>

# cwFill

Foreground estimation for soft matte edges. cwFill works out the true
foreground colour in the semi-transparent band of a matte and fills it back
into the plate, so edges composite cleanly instead of fringing toward the old
background. It also has a luminance-preserving despill.

No ML model and no GPU needed. It runs in Flame (macOS and Linux) and in
other OFX hosts.

<!-- screenshot -->

## What it does

- **Edge fill:** a Rust port of pymatting's multilevel foreground estimator
  (Germer et al. 2020), a real global solve rather than a per-pixel trick.
  Optional *Image Awareness* keeps busy background detail out of the fill. At
  0 the solve matches pymatting exactly.
- **Outputs:** foreground, premultiplied foreground, the recovered
  background, or a contamination check.
- **Despill:** Green / Blue / Red presets or a custom colour, with hue
  range and softness. It keeps luminance, so blonde hair stays bright and
  skin is spared.
- **Edge Tone (-1..1):** positive rolls off edge fill brightened by a bright
  background, or lifts fill darkened by a dark one, toward the nearby solid
  foreground; negative pushes it the other way (more glow, deeper edge).
  Reaches flyaway wisps up to about a quarter of the frame from the body.
  **Tone Spread** (px, default 32) carries the correction from the dense
  edge outward into the wisps, fading with distance, so it isn't a ring
  around the core; it never moves a pixel past the hair's own level. No
  re-solve.
- **Limit to Soft Matte** is on by default: despill cleans the edge and spares
  the solid body.
- **Bit depth:** a 16-bit plate gives a 16-bit result.

## Inputs

The plate (RGB) and a matte (alpha), e.g. from [cwRefine](cwRefine.md).

Full parameter reference: [crates/cwfill/README.md](../crates/cwfill/README.md).

## Download and install

**[cwFill for Linux](../../../releases/latest/download/cwFill-linux.tar.gz)** ·
**[cwFill for macOS](../../../releases/latest/download/cwFill-macos.tar.gz)** ·
[all releases](../../../releases)

Unpack and run `./install.sh`. It asks where to install, with a default
for every answer, and adds cwFill to an existing cwTools install.
No models, no GPU needed.

Prefer everything at once? Get the [full suite](../README.md#full-suite).

[← all tools](../README.md)
