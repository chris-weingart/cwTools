//! Parameter definitions + readback.
//!
//! Flame-specific constraints baked in (per Philippe Jean / flame_ofx_host_facts):
//!  - No Pages: we organise with param Groups (shown as column headers in Flame).
//!  - No parametric params: falloff/tolerance are plain Doubles.
//!  - Continuous knobs = Double (animatable); mode selectors = Choice
//!    (animation only Flame 2025.1+, which we don't rely on).

use crate::ofx_api::*;
use crate::{param, prop_set_double, prop_set_int, prop_set_string};
use std::os::raw::{c_char, c_int};

// Parameter names (also used as readback keys).
pub const P_SELF_MUL_STRENGTH: &[u8] = b"selfMulStrength\0";
pub const P_SELF_MUL_MODE: &[u8] = b"selfMulMode\0"; // 0 linear, 1 body-protect
pub const P_SOFT_BAND_ONLY: &[u8] = b"softBandOnly\0";
pub const P_BAND_DILATION: &[u8] = b"bandDilation\0";
pub const P_PIN_HARD: &[u8] = b"pinHardAlpha\0";
pub const P_PIN_LOW: &[u8] = b"pinLow\0";
pub const P_PIN_HIGH: &[u8] = b"pinHigh\0";
pub const P_REGULARIZATION: &[u8] = b"regularization\0";
pub const P_QUALITY: &[u8] = b"quality\0"; // 0 low, 1 med, 2 high
pub const P_GRADIENT_WEIGHT: &[u8] = b"gradientWeight\0";
pub const P_IMAGE_AFFINITY: &[u8] = b"imageAffinity\0";
pub const P_AUTO_SOLVE: &[u8] = b"autoSolve\0";
pub const P_INPUT_COLORSPACE: &[u8] = b"inputColorSpace\0"; // 0 acescg, 1 acescct, 2 display
pub const P_ROI_PADDING: &[u8] = b"roiPadding\0";
pub const P_OUTPUT_MODE: &[u8] = b"outputMode\0"; // 0 fg, 1 fg premult, 2 contamination, 3 bg
pub const P_DESPILL_SCREEN: &[u8] = b"despillScreen\0"; // 0 none, 1 green, 2 blue, 3 red
pub const P_RESIDUAL_DESPILL: &[u8] = b"residualDespill\0"; // strength
pub const P_DESPILL_RANGE: &[u8] = b"despillRange\0";
pub const P_DESPILL_SOFTNESS: &[u8] = b"despillSoftness\0";
pub const P_SCREEN_COLOUR: &[u8] = b"screenColour\0"; // used when despillScreen = Custom
pub const DESPILL_CUSTOM: c_int = 4; // despillScreen "Custom Colour"
pub const P_DESPILL_SOFT_ONLY: &[u8] = b"despillSoftOnly\0";
pub const P_DESPILL_MATTE_THRESH: &[u8] = b"despillMatteThresh\0";
pub const P_EDGE_TONE_SPREAD: &[u8] = b"edgeToneSpread\0"; // px: carry the Edge Tone correction outward
pub const P_EDGE_TONE: &[u8] = b"edgeTone\0"; // -1 exaggerate .. 0 off .. 1 edge fill held at the body level

// Flame TRUNCATES OFX string properties at 255 chars (hints cut mid-word in the
// tooltip, observed in Flame) and renders roughly 15 chars of a Double param's
// label before clipping. Keep hints <= 255 bytes (guard test below) and labels
// short.
//
// Groups. OFX has no Pages in Flame (params show under group/column headers in
// declaration order), so group ORDER is how we express prominence: the foreground
// "unmix" groups (Matte Conditioning / Solver / Output) come first = dominant;
// Despill comes last. (A tabbed Despill/Unmix page with default-open is a Matchbox
// feature; not available to an OFX plugin, and cwFill must be OFX for the solve.)
const G_MATTE: &[u8] = b"grpMatte\0";
const G_SOLVER: &[u8] = b"grpSolver\0";
const G_OUTPUT: &[u8] = b"grpOutput\0";
const G_DESPILL: &[u8] = b"grpDespill\0";

fn define(param_set: OfxParamSetHandle, ty: &[u8], name: &[u8]) -> OfxPropertySetHandle {
    let mut props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        (param().paramDefine)(
            param_set,
            ty.as_ptr() as *const c_char,
            name.as_ptr() as *const c_char,
            &mut props,
        );
    }
    props
}

fn label(props: OfxPropertySetHandle, text: &[u8]) {
    prop_set_string(props, kOfxPropLabel, 0, text);
}
fn hint(props: OfxPropertySetHandle, text: &[u8]) {
    prop_set_string(props, kOfxParamPropHint, 0, text);
}
fn parent(props: OfxPropertySetHandle, group: &[u8]) {
    prop_set_string(props, kOfxParamPropParent, 0, group);
}

/// Define a Double param with a full range + a fine scrub increment. Flame's
/// slider granularity is derived from DisplayMin/DisplayMax and Increment; a
/// tight display range + small increment fixes the "jumps to full value" scrub.
#[allow(clippy::too_many_arguments)]
fn dbl(
    param_set: OfxParamSetHandle,
    name: &[u8],
    lbl: &[u8],
    hnt: &[u8],
    def: f64,
    mn: f64,
    mx: f64,
    dmn: f64,
    dmx: f64,
    inc: f64,
    digits: c_int,
    group: &[u8],
) {
    let p = define(param_set, kOfxParamTypeDouble, name);
    label(p, lbl);
    hint(p, hnt);
    prop_set_double(p, kOfxParamPropDefault, 0, def);
    prop_set_double(p, kOfxParamPropMin, 0, mn);
    prop_set_double(p, kOfxParamPropMax, 0, mx);
    prop_set_double(p, kOfxParamPropDisplayMin, 0, dmn);
    prop_set_double(p, kOfxParamPropDisplayMax, 0, dmx);
    prop_set_double(p, kOfxParamPropIncrement, 0, inc);
    prop_set_int(p, kOfxParamPropDigits, 0, digits);
    parent(p, group);
}

/// Define an Integer param with min/max + a sensible display range.
#[allow(clippy::too_many_arguments)]
fn intp(
    param_set: OfxParamSetHandle,
    name: &[u8],
    lbl: &[u8],
    hnt: &[u8],
    def: c_int,
    mn: c_int,
    mx: c_int,
    dmn: c_int,
    dmx: c_int,
    group: &[u8],
) {
    let p = define(param_set, kOfxParamTypeInteger, name);
    label(p, lbl);
    hint(p, hnt);
    prop_set_int(p, kOfxParamPropDefault, 0, def);
    prop_set_int(p, kOfxParamPropMin, 0, mn);
    prop_set_int(p, kOfxParamPropMax, 0, mx);
    prop_set_int(p, kOfxParamPropDisplayMin, 0, dmn);
    prop_set_int(p, kOfxParamPropDisplayMax, 0, dmx);
    parent(p, group);
}

/// Define the full parameter tree. Called from describeInContext.
pub fn define_params(param_set: OfxParamSetHandle) {
    // Flame panel: 5-row columns, a Choice = 2 rows, every group starts a column,
    // 6 columns a page (see the cwTools grid rules). Page 1 = Main | Matte
    // Conditioning x2 | Solver | Despill x2. Group order is still prominence order.

    let g = define(param_set, kOfxParamTypeGroup, G_OUTPUT);
    label(g, b"Main\0");

    let p = define(param_set, kOfxParamTypeChoice, P_OUTPUT_MODE);
    label(p, b"Output\0");
    hint(p, b"Foreground = recovered fill. Foreground (premultiplied) = fill x alpha. \
              Contamination = source minus foreground, the removed background bleed \
              (diagnostic). Background = recovered background.\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 0, b"Foreground\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 1, b"Foreground (premultiplied)\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 2, b"Contamination (Source - FG)\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 3, b"Background\0");
    prop_set_int(p, kOfxParamPropDefault, 0, 0);
    parent(p, G_OUTPUT);

    let p = define(param_set, kOfxParamTypeChoice, P_QUALITY);
    label(p, b"Quality\0");
    hint(p, b"Solver iteration count. Low = fastest / roughest. High = most converged / \
              cleanest. Raise if the recovered edge looks noisy or unresolved.\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 0, b"Low\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 1, b"Medium\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 2, b"High\0");
    prop_set_int(p, kOfxParamPropDefault, 0, 2);
    parent(p, G_OUTPUT);

    let p = define(param_set, kOfxParamTypeBoolean, P_AUTO_SOLVE);
    label(p, b"Auto-Tune Solver (experimental)\0");
    hint(p, b"EXPERIMENTAL. When ON, derives Regularization + Gradient Weight from the band's \
              measured edge detail (hair -> follow the image; smooth -> more smoothing), \
              ignoring the manual sliders. Measures once per render - a smart default, not \
              a parameter search.\0");
    prop_set_int(p, kOfxParamPropDefault, 0, 0);
    parent(p, G_OUTPUT);

    let g = define(param_set, kOfxParamTypeGroup, G_MATTE);
    label(g, b"Matte Conditioning\0");

    let p = define(param_set, kOfxParamTypeChoice, P_SELF_MUL_MODE);
    label(p, b"Self-Mul Mode\0");
    hint(p, b"How the self-multiply is shaped. Linear = plain alpha*alpha blend (uniform \
              choke). Body-Protect = preserves near-solid (high-alpha) areas and only chokes \
              the soft transition, so the subject body does not shrink.\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 0, b"Linear\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 1, b"Body-Protect\0");
    prop_set_int(p, kOfxParamPropDefault, 0, 1);
    parent(p, G_MATTE);

    dbl(param_set, P_SELF_MUL_STRENGTH, b"Self-Mul Strength\0",
        b"Tightens the soft matte edge BEFORE the solve by multiplying alpha by itself. \
          0 = no change, 1 = alpha squared; above 1 the multiply repeats (^3, ^4 ...) for a \
          stronger edge choke. Use to pull in a soft or over-grown matte.\0",
        0.0, 0.0, 4.0, 0.0, 4.0, 0.01, 3, G_MATTE);

    let p = define(param_set, kOfxParamTypeBoolean, P_SOFT_BAND_ONLY);
    label(p, b"Soft-Band Only\0");
    hint(p, b"ON (default): recover foreground colour only in the soft band (0 < alpha < 1); \
              solid and clear pixels pass the plate through untouched. OFF: solve the whole \
              frame (slower); 'Pin Hard Alpha' then controls the solid/clear regions.\0");
    prop_set_int(p, kOfxParamPropDefault, 0, 1);
    parent(p, G_MATTE);

    intp(param_set, P_BAND_DILATION, b"Band Dilation\0",
        b"Grow the soft-band region outward by this many pixels before solving, so the \
          recovered foreground extends slightly PAST the matte edge. Use to capture colour \
          spill the matte clipped off. 0 = exact band. (Soft-Band Only mode.)\0",
        2, 0, 256, 0, 32, G_MATTE);

    let p = define(param_set, kOfxParamTypeBoolean, P_PIN_HARD);
    label(p, b"Pin Hard Alpha to Plate\0");
    hint(p, b"Full-frame mode only (ignored when 'Soft-Band Only' is ON). Force fully-opaque \
              (alpha >= Pin High) and fully-clear (alpha <= Pin Low) pixels to the exact \
              original plate colour instead of the solved estimate.\0");
    prop_set_int(p, kOfxParamPropDefault, 0, 1);
    parent(p, G_MATTE);

    dbl(param_set, P_PIN_LOW, b"Pin Low\0",
        b"Alpha at or below this counts as fully background (clear) and is treated as OUTSIDE \
          the soft band. Default 0.002.\0",
        0.002, 0.0, 0.5, 0.0, 0.05, 0.001, 4, G_MATTE);

    dbl(param_set, P_PIN_HIGH, b"Pin High\0",
        b"Alpha at or above this counts as fully foreground (solid) and is treated as OUTSIDE \
          the soft band. Default 0.998.\0",
        0.998, 0.5, 1.0, 0.95, 1.0, 0.001, 4, G_MATTE);

    intp(param_set, P_ROI_PADDING, b"RoI Padding\0",
        b"Extra pixels around the solve region so the multigrid keeps surrounding \
          solid/clear pixels as anchors. Usually leave at default - mainly matters when \
          the soft band reaches the frame edge. (Soft-Band Only mode.)\0",
        64, 0, 512, 0, 256, G_MATTE);

    let g = define(param_set, kOfxParamTypeGroup, G_SOLVER);
    label(g, b"Solver\0");

    let p = define(param_set, kOfxParamTypeChoice, P_INPUT_COLORSPACE);
    label(p, b"Input Colour Space\0");
    hint(p, b"What encoding the Source RGB arrives in. ACEScg (default) is encoded to ACEScct \
              for the solve and decoded back (output stays scene-linear). ACEScct and \
              Display/sRGB solve as-is (output stays source-encoded). Scene-linear is \
              never solved raw.\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 0, b"ACEScg (scene-linear)\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 1, b"ACEScct (log)\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 2, b"Display (sRGB)\0");
    prop_set_int(p, kOfxParamPropDefault, 0, 0);
    parent(p, G_SOLVER);

    dbl(param_set, P_REGULARIZATION, b"Regularization\0",
        b"Solver smoothness vs. follow-the-image fidelity. Higher = smoother, more averaged \
          foreground colour. Lower = sharper, follows local image detail (and noise). \
          pymatting default 1e-5.\0",
        1e-5, 1e-8, 1.0, 1e-6, 1e-2, 1e-6, 6, G_SOLVER);

    dbl(param_set, P_GRADIENT_WEIGHT, b"Gradient Weight\0",
        b"How strongly the solve follows image edges (the alpha gradient) when propagating \
          foreground colour. Higher = colour respects edges more (less bleed across \
          contours). Lower = smoother propagation. Default 1.0.\0",
        1.0, 0.0, 10.0, 0.0, 4.0, 0.05, 2, G_SOLVER);

    dbl(param_set, P_IMAGE_AFFINITY, b"Image Awareness\0",
        b"Lets the recovered background break at image edges (seams, patterns) instead of \
          pushing background discontinuities into the fill as coloured chips. 0 (default) = \
          off, classic pymatting. 0.5 = good working value. Foreground propagation is \
          untouched.\0",
        0.0, 0.0, 1.0, 0.0, 1.0, 0.05, 2, G_SOLVER);

    let g = define(param_set, kOfxParamTypeGroup, G_DESPILL);
    label(g, b"Despill\0");

    let p = define(param_set, kOfxParamTypeChoice, P_DESPILL_SCREEN);
    label(p, b"Despill Screen\0");
    hint(p, b"Screen colour to suppress across the recovered foreground. None = off. \
              Green/Blue/Red = preset hue; Custom Colour uses the 'Screen Colour' pick. \
              Removes only the screen cast, preserving luminance. Inert until 'Despill \
              Strength' is raised.\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 0, b"None\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 1, b"Green\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 2, b"Blue\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 3, b"Red\0");
    prop_set_string(p, kOfxParamPropChoiceOption, 4, b"Custom Colour\0");
    prop_set_int(p, kOfxParamPropDefault, 0, 1);
    parent(p, G_DESPILL);

    let p = define(param_set, kOfxParamTypeRGB, P_SCREEN_COLOUR);
    label(p, b"Screen Colour\0");
    hint(p, b"The exact screen/spill colour to suppress - used when 'Despill Screen' = \
              Custom Colour. Sample it off your backing for hue-accurate despill. The \
              Green/Blue/Red presets ignore this. Only the hue (chroma direction) matters; \
              brightness is irrelevant.\0");
    prop_set_double(p, kOfxParamPropDefault, 0, 0.0);
    prop_set_double(p, kOfxParamPropDefault, 1, 1.0);
    prop_set_double(p, kOfxParamPropDefault, 2, 0.0);
    parent(p, G_DESPILL);

    dbl(param_set, P_RESIDUAL_DESPILL, b"Despill Strength\0",
        b"How much screen-hue chroma to remove. 0 = off, 1 = full. Luma is preserved at any \
          strength (only the colour cast is pulled, never the brightness).\0",
        0.0, 0.0, 1.0, 0.0, 1.0, 0.01, 2, G_DESPILL);

    dbl(param_set, P_DESPILL_RANGE, b"Hue Range\0",
        b"How wide a band of hues around the screen colour counts as spill. Low = only \
          near-exact screen hues (best skin/hair protection). Raise if green is left in \
          the edges; lower if the subject desaturates.\0",
        0.5, 0.0, 1.0, 0.0, 1.0, 0.01, 2, G_DESPILL);

    dbl(param_set, P_DESPILL_SOFTNESS, b"Hue Softness\0",
        b"Rolloff at the edge of the hue range - how gradually suppression fades for hues \
          leaving the band. Higher = smoother transition (avoids a hard chroma edge).\0",
        0.5, 0.0, 1.0, 0.0, 1.0, 0.01, 2, G_DESPILL);

    let p = define(param_set, kOfxParamTypeBoolean, P_DESPILL_SOFT_ONLY);
    label(p, b"Limit to Soft Matte\0");
    hint(p, b"ON (default): despill only where alpha is below 'Matte Threshold' - spare the \
              solid body, clean just the edge. OFF: despill the whole foreground (spill on the \
              body too).\0");
    prop_set_int(p, kOfxParamPropDefault, 0, 1);
    parent(p, G_DESPILL);

    dbl(param_set, P_DESPILL_MATTE_THRESH, b"Matte Threshold\0",
        b"Only used when 'Limit to Soft Matte' is ON. Despill applies where alpha is below this \
          (soft rolloff just under it). 1.0 = everything except perfectly-solid pixels; lower to \
          spare more of the body.\0",
        1.0, 0.0, 1.0, 0.0, 1.0, 0.01, 2, G_DESPILL);

    dbl(param_set, P_EDGE_TONE, b"Edge Tone\0",
        b"Edge fill vs the nearby solid foreground. + pulls it to the body level (rolls off edges \
          a bright background brightened, lifts edges a dark one darkened); - pushes the other \
          way (more glow / deeper edge). 0 = off. No re-solve.\0",
        0.0, -1.0, 1.0, -1.0, 1.0, 0.01, 2, G_DESPILL);

    dbl(param_set, P_EDGE_TONE_SPREAD, b"Tone Spread\0",
        b"How far (px) the Edge Tone correction carries outward from the dense edge into the \
          wisps, fading with distance (anti-halo). It never moves a pixel past the body level. \
          0 = each pixel on its own.\0",
        32.0, 0.0, 512.0, 0.0, 128.0, 1.0, 0, G_DESPILL);
}

// ---------------------------------------------------------------------------
// Readback (render time). Uses the variadic C shim.
// ---------------------------------------------------------------------------
fn handle(param_set: OfxParamSetHandle, name: &[u8]) -> OfxParamHandle {
    let mut h: OfxParamHandle = std::ptr::null_mut();
    unsafe {
        (param().paramGetHandle)(param_set, name.as_ptr() as *const c_char, &mut h, std::ptr::null_mut());
    }
    h
}

pub fn get_double(param_set: OfxParamSetHandle, name: &[u8]) -> f64 {
    let h = handle(param_set, name);
    let mut out: f64 = 0.0;
    unsafe {
        cw_param_get_double(param().paramGetValue, h, &mut out);
    }
    out
}

pub fn get_rgb(param_set: OfxParamSetHandle, name: &[u8]) -> [f32; 3] {
    let h = handle(param_set, name);
    let (mut r, mut g, mut b) = (0.0f64, 0.0f64, 0.0f64);
    unsafe {
        cw_param_get_rgb(param().paramGetValue, h, &mut r, &mut g, &mut b);
    }
    [r as f32, g as f32, b as f32]
}

pub fn set_int(param_set: OfxParamSetHandle, name: &[u8], v: c_int) {
    let h = handle(param_set, name);
    unsafe {
        cw_param_set_int(param().paramSetValue, h, v);
    }
}

pub fn get_int(param_set: OfxParamSetHandle, name: &[u8]) -> c_int {
    let h = handle(param_set, name);
    let mut out: c_int = 0;
    unsafe {
        cw_param_get_int(param().paramGetValue, h, &mut out);
    }
    out
}

/// Snapshot of all params, resolved at render time.
#[derive(Clone, Copy)]
pub struct Values {
    pub self_mul_strength: f32,
    pub self_mul_mode: i32,
    pub soft_band_only: bool,
    pub band_dilation: i32,
    pub pin_hard: bool,
    pub pin_low: f32,
    pub pin_high: f32,
    pub regularization: f32,
    pub quality: i32,
    pub gradient_weight: f32,
    pub image_affinity: f32,
    pub auto_solve: bool,
    pub input_colorspace: i32,
    pub roi_padding: i32,
    pub output_mode: i32,
    pub despill_screen: i32,
    pub residual_despill: f32,
    pub despill_range: f32,
    pub despill_softness: f32,
    pub screen_colour: [f32; 3],
    pub despill_soft_only: bool,
    pub despill_matte_thresh: f32,
    pub edge_tone: f32,
    /// px decay length the Edge Tone correction carries outward (0 = per pixel).
    pub edge_tone_spread: f32,
}

pub fn read_all(param_set: OfxParamSetHandle) -> Values {
    Values {
        self_mul_strength: get_double(param_set, P_SELF_MUL_STRENGTH) as f32,
        self_mul_mode: get_int(param_set, P_SELF_MUL_MODE),
        soft_band_only: get_int(param_set, P_SOFT_BAND_ONLY) != 0,
        band_dilation: get_int(param_set, P_BAND_DILATION),
        pin_hard: get_int(param_set, P_PIN_HARD) != 0,
        pin_low: get_double(param_set, P_PIN_LOW) as f32,
        pin_high: get_double(param_set, P_PIN_HIGH) as f32,
        regularization: get_double(param_set, P_REGULARIZATION) as f32,
        quality: get_int(param_set, P_QUALITY),
        gradient_weight: get_double(param_set, P_GRADIENT_WEIGHT) as f32,
        image_affinity: get_double(param_set, P_IMAGE_AFFINITY) as f32,
        auto_solve: get_int(param_set, P_AUTO_SOLVE) != 0,
        input_colorspace: get_int(param_set, P_INPUT_COLORSPACE),
        roi_padding: get_int(param_set, P_ROI_PADDING),
        output_mode: get_int(param_set, P_OUTPUT_MODE),
        despill_screen: get_int(param_set, P_DESPILL_SCREEN),
        residual_despill: get_double(param_set, P_RESIDUAL_DESPILL) as f32,
        despill_range: get_double(param_set, P_DESPILL_RANGE) as f32,
        despill_softness: get_double(param_set, P_DESPILL_SOFTNESS) as f32,
        screen_colour: get_rgb(param_set, P_SCREEN_COLOUR),
        despill_soft_only: get_int(param_set, P_DESPILL_SOFT_ONLY) != 0,
        despill_matte_thresh: get_double(param_set, P_DESPILL_MATTE_THRESH) as f32,
        edge_tone: get_double(param_set, P_EDGE_TONE) as f32,
        edge_tone_spread: get_double(param_set, P_EDGE_TONE_SPREAD).max(0.0) as f32,
    }
}

#[cfg(test)]
mod tests {
    // The Custom Colour pick reaches us through the variadic C shim. Prove the
    // shim hands a variadic callee three double* in order on this arch (the
    // "despill does nothing" report suspected a 0/default read).
    #[test]
    fn rgb_shim_reads_three_doubles_through_variadic_call() {
        use crate::ofx_api::cw_param_get_rgb;
        extern "C" {
            fn cw_selftest_variadic_rgb(handle: *mut std::ffi::c_void, ...) -> std::os::raw::c_int;
        }
        let (mut r, mut g, mut b) = (0.0f64, 0.0f64, 0.0f64);
        let st = unsafe {
            cw_param_get_rgb(
                cw_selftest_variadic_rgb as *const std::ffi::c_void,
                std::ptr::null_mut(),
                &mut r,
                &mut g,
                &mut b,
            )
        };
        assert_eq!(st, 0);
        assert_eq!((r, g, b), (0.249145508, 0.537597656, 0.720214844));
    }

    // Class-killer guard: Flame truncates OFX string properties at 255 chars —
    // a hint that exceeds it shows up as a tooltip cut mid-word (observed
    // live: 'Image Awareness' + 'RoI Padding' hints). Scan every
    // byte-string literal in this file and fail on any over the limit, so the
    // next param added can't regress silently.
    #[test]
    fn no_string_literal_exceeds_flame_255_char_limit() {
        let src = include_str!("params.rs");
        let bytes = src.as_bytes();
        let mut i = 0;
        let mut offenders = Vec::new();
        while i + 1 < bytes.len() {
            if bytes[i] == b'b' && bytes[i + 1] == b'"' {
                let start = i;
                i += 2;
                let mut len = 0usize;
                let mut preview = String::new();
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' {
                        match bytes.get(i + 1) {
                            // Line continuation: skip newline + leading spaces.
                            Some(b'\n') => {
                                i += 2;
                                while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
                                    i += 1;
                                }
                                continue;
                            }
                            // \0 terminator and other escapes count as 1 byte.
                            Some(_) => {
                                if bytes[i + 1] != b'0' {
                                    len += 1;
                                }
                                i += 2;
                                continue;
                            }
                            None => break,
                        }
                    }
                    len += 1;
                    if preview.len() < 40 {
                        preview.push(bytes[i] as char);
                    }
                    i += 1;
                }
                if len > 255 {
                    let line = src[..start].matches('\n').count() + 1;
                    offenders.push(format!("line {line}: {len} chars — {preview}..."));
                }
            }
            i += 1;
        }
        assert!(
            offenders.is_empty(),
            "string literals exceed Flame's 255-char OFX property limit:\n{}",
            offenders.join("\n")
        );
    }
}
