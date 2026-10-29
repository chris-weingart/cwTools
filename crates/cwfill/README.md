# cwFill

An OpenFX plugin for **foreground estimation** — it computes the true foreground
colour for the soft edge of a matte (the per-pixel RGB that belongs with a given
alpha) and fills it back into the plate, so composited edges hold up instead of
fringing toward whatever was behind them. It runs the exact estimation natively
inside Flame, and in any OFX host (Nuke, Resolve).

The core is a faithful Rust port of pymatting's multilevel foreground estimator
(`estimate_foreground_ml`, Germer et al. 2020) — a global multigrid linear solve,
not a per-pixel approximation. Given the source RGB and a matte, it solves for the
foreground colour in the semi-transparent band where foreground and background
are mixed, then writes a clean, composite-ready edge. One opt-in extension goes
beyond the original algorithm: *Image Awareness* (see Parameters), which stops
busy background detail from contaminating the fill; at its default of 0 the
solve is exact pymatting behaviour.

## What it does

- **Foreground estimation / edge fill** — in the soft matte band (`0 < α < 1`) it
  unmixes the observed colour into clean foreground and fills it into the plate, so
  edges composite without a dark or background-tinted fringe. This is the point of
  the tool.
- **Composite-ready outputs** — foreground, premultiplied foreground, the recovered
  background, or a contamination diagnostic.
- **Despill** — hue-targeted screen-spill suppression. Projects onto the screen hue's
  zero-luminance chroma axis, so it removes the colour cast while **preserving luminance
  by construction** (warm/blonde hair keeps its brightness) and sparing off-hue colours
  (skin) via a hue-band weight. Green/Blue/Red presets or an exact Custom Colour pick;
  Hue Range + Softness controls; optional Limit-to-Soft-Matte (alpha-gated) to spare the body.

## Status

Working and in daily production use.

- **Solver** — pure-math Rust port, unit-tested against pymatting (`cargo test`):
  recovers a known foreground from a synthetic composite; pyramid sizing matches
  Python's banker's rounding.
- **In Flame** — loads and renders in Flame 2027 and 2026.2.2, on macOS
  (arm64) and Linux (x86-64).
- **Colour handling** — `Input Colour Space` picker (ACEScg default): ACEScg sources
  are encoded to ACEScct internally for the solve and decoded back to scene-linear;
  ACEScct / sRGB sources are already perceptual and solve as-is. Scene-linear is never
  solved raw — sparse bright pixels (hair rims, speculars) corrupt the multilevel
  solve's mean-based seeding (regression-tested).
- **Performance (0.3.0)** — the solve runs multi-threaded (pipelined rows, output
  bit-identical to the single-threaded solve) and every per-pixel stage is
  row-parallel. Solved F/B are cached per node, so changing only despill, Limit to
  Soft Matte or the Output mode re-renders without re-solving. Environment
  overrides: `CWFILL_THREADS=<n>` (default: cores, max 16), `CWFILL_CACHE_MB=<n>`
  (default 1024, `0` = off), `CWFILL_DEBUG=1` (one line per render to
  `/tmp/cwFill.log`: the parameter values the plugin read, alpha stats, timings).
- **Bit depth** — 16-bit half and 32-bit float. The output has the Source's depth
  (a 16-bit plate gives a 16-bit result); the solve always runs in 32-bit float.

## Usage (Flame)

- Add the node in a Batch/BFX schematic: **OpenFX → Load Plugin → cwTools →
  cwFill**.
- Flame splits the node's Source into **RGB + Alpha sockets** — plug the plate into
  RGB and the matte into Alpha (or connect one RGBA clip carrying the matte in its
  alpha). No separate matte input needed.
- Set **Input Colour Space** to match what the source is encoded as
  (default: ACEScg).
- **Output** modes: *Foreground* (the default) fills recovered edge colour into
  the plate; *Foreground (premultiplied)* is the same result premultiplied by
  the matte — use whichever your comp expects (Flame's comp ops can premultiply
  on the fly). Judge edge quality **premultiplied or comped** — unpremultiplied
  foreground legitimately shows extrapolated colour where alpha is near zero.
- **Noisy mattes:** if the matte's background isn't quite zero (some AI matting
  models leave a faint noise floor), that noise joins the solve band — raise
  *Pin Low* just above the floor (try 0.01–0.05).
- **Busy backgrounds:** if the soft band crosses strong background structure
  (pattern edges, seams, text) and coloured chips appear in the fill, raise
  *Image Awareness* (Solver group; 0.5 is a good working value). It lets the
  recovered background break at image edges instead of pushing them into the
  foreground.

## Install

Get cwFill from the [cwTools releases](https://github.com/chris-weingart/cwTools/releases)
(on its own or in the full suite), unpack it and run `./install.sh`. Then restart
your OFX host; Flame also needs its `ofxPluginCache.xml` deleted
(`/opt/Autodesk/.flame*/status/`) so the new plugin is rescanned.

Feedback is welcome as GitHub issues: real shots where the fill or despill
breaks down, and anything host-specific (which host and version, and what
happened).

## Build / install (from source)

```bash
cargo test                 # solver math, no Flame needed
scripts/build.sh           # → dist/cwFill.ofx.bundle  (native per-OS)
sudo scripts/install.sh    # → /Library/OFX/Plugins (macOS) | /usr/OFX/Plugins (Linux)
```

Needs only a Rust toolchain (`rustup`) and a C compiler. The bundle must be
**built natively per platform** (a macOS arm64 binary won't run on Linux):
clone the repo on the target machine and run `build.sh` + `install.sh` there.

### Other OFX hosts (Nuke, Resolve)

Nothing extra to install — the plugin installs to the standard OFX plugin
directory (`/Library/OFX/Plugins` / `/usr/OFX/Plugins`), which every OFX host
scans. The one usage difference: other hosts don't split the source into
RGB + Alpha sockets the way Flame does, so feed a single RGBA stream with the
**matte in the alpha channel**:

- **Nuke:** Copy/Shuffle the matte into `rgba.alpha` upstream of cwFill.
- **Resolve (Fusion page):** merge the matte into alpha with a ChannelBooleans
  node — plate → Background, matte → Foreground, **To Alpha = Red FG** — and
  feed its output into cwFill. (The triangle inputs on the node are Effect
  Masks and won't accept the matte image.)

The wrapper is plain OFX with no Flame-specific calls, but Flame is the host it
is tested in; reports from Nuke and Resolve are welcome.

### Dev loop (no install)

```bash
scripts/flame_dev_launch.sh    # launches Flame with OFX_PLUGIN_PATH=dist, cache off,
                               # and OFX debug output enabled
```

After a rebuild, stale params/clips come from the host's plugin cache — the dev
launcher disables it (`DL_OFX_PLUGIN_CACHE_ENABLED=0`); for an installed bundle,
delete `ofxPluginCache.xml`.

## Parameters

**Matte Conditioning**
- *Self-Mul Strength* / *Mode* — tighten the soft edge before solving by multiplying
  alpha with itself (0–4; >1 repeats the multiply). Linear or body-protecting.
- *Soft-Band Only* — recover foreground only in the soft band; pass the plate through
  in solid/clear regions (default on).
- *Band Dilation* / *RoI Padding* — grow the recovered region past the matte edge, and
  pad the solve crop so the multigrid keeps surrounding hard-alpha pixels as anchors.
- *Pin Hard Alpha* + *Pin Low/High* — in full-frame mode, force fully-opaque/clear
  pixels to the exact plate.

**Solver**
- *Regularization* — smoothness vs. image fidelity (pymatting default 1e-5).
- *Quality* — solver iteration count (Low/Medium/High).
- *Gradient Weight* — how strongly colour propagation respects image edges.
- *Image Awareness* — the one deliberate departure from pymatting parity, and
  opt-in: pymatting's smoothness prior forces the recovered background to stay
  continuous across real background discontinuities, dumping the difference
  into the foreground as coloured chips wherever the soft band crosses
  background structure. Raising this attenuates only the *background*
  smoothness at image edges (alpha-mix-normalised, faded out near solid alpha)
  so the background can break there instead — roughly 10× lower fill error on
  a colour-bars torture test, with no effect on flat or grainy backgrounds.
  0 (default) = off, bit-exact pymatting behaviour; 0.5 = good working value.
  Foreground propagation is never touched.
- *Auto-Tune Solver* (experimental) — opt-in heuristic: measures the soft band's
  edge detail once per render and derives Regularization + Gradient Weight from it
  (hair → follow the image; smooth edges → more smoothing), overriding the sliders.
- *Input Colour Space* — what the source is encoded as: ACEScg (default; solved in
  ACEScct internally, output stays scene-linear), ACEScct (log), or Display (sRGB) —
  the latter two solve as-is and output stays in the source encoding.

**Despill**
- *Despill Screen* — None / Green / Blue / Red / Custom Colour (target hue; default Green,
  inert until strength > 0).
- *Screen Colour* — exact backing colour, used when Despill Screen = Custom Colour (only hue matters). Picking it switches Despill Screen to Custom Colour.
- *Despill Strength* — how much screen-hue chroma to remove (luma preserved at any strength).
- *Hue Range* / *Hue Softness* — width and rolloff of the hue band counted as spill
  (low range = tightest skin/hair protection).
- *Limit to Soft Matte* (on by default) + *Matte Threshold* — despill only where alpha is
  below the threshold (spare the solid body, clean just the edge); off = the whole foreground.
- *Edge Tone* (-1..1) — positive pulls the soft-band fill toward the nearby solid
  foreground where the background shifted it: over a bright background, fill brighter
  than the body is rolled off; over a dark one, fill darker than the body is lifted.
  Negative pushes that same deviation further (more glow / deeper edge, as far again at
  -1). About half a stop either side of the body stays untouched (rim light keeps some
  brightness). The reference is the solid body (alpha at or above *Pin High*): its local
  level within ~1/32 of the frame, else its wider level within ~1/4, so flyaway wisps
  far from the core are reached too. Full strength up to alpha 0.95; only the soft band
  (between the pins) is touched. 0 = off (default). No re-solve.
- *Tone Spread* — how far (px, default 32) the Edge Tone correction carries outward from
  the dense edge into the wisps: each pixel takes the strongest correction around it,
  decaying as exp(-distance / spread), applied only where the pixel itself sits on the
  background's side of the body and never past the body level. Removes the ring around
  the core; 0 = each pixel on its own. Output-only.

**Output** (last group)
- *Output* — Foreground · Foreground (premultiplied) · Contamination (Source − FG) · Background.

## Roadmap

- **Despill v2** — despill coupled to the FG/BG solve, plus an edge-localized
  respill stage.
- **Automatic colour space** — read the clip's tagged colour space via the OFX
  Colour Management suite where the host supports it (Flame 2026.1+), falling
  back to the manual picker elsewhere.
- **Optional Matte input** — a second, optional input clip so hosts without
  Flame's RGB+Alpha socket split can connect the matte directly instead of
  copying it into the source's alpha channel upstream.

## Credits

- Foreground estimation: pymatting, *Fast Multi-Level Foreground Estimation*
  (Germer, Uelwer, Conrad, Harmeling, 2020).

## License

Apache-2.0.
