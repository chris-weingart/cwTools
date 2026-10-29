//! OFX clip/image access. OFX images are BOTTOM-UP (y grows upwards) with an
//! arbitrary pixel bounds rectangle and row stride; the helpers here convert to
//! the TOP-DOWN buffers in [`crate::pixels`] at the boundary.
//!
//! Only 32-bit float images are accepted (every cwTools plugin requests float
//! clip depth; half-float casts are garbage in ACEScg).

use crate::ofx_api::*;
use crate::props::{prop_get_int, prop_get_pointer, prop_get_rect_i, prop_get_string};
use crate::suites::effect;
use std::os::raw::c_char;

pub struct Image {
    pub props: OfxPropertySetHandle,
    pub data: *mut u8,
    pub bounds: OfxRectI,
    pub row_bytes: i32,
    pub n_comps: i32,
}

impl Image {
    /// Pointer to the first float of pixel (x, y) (OFX pixel coordinates).
    ///
    /// # Safety
    /// (x, y) must lie inside `bounds`.
    #[inline]
    pub unsafe fn px(&self, x: i32, y: i32) -> *mut f32 {
        self.data
            .offset((y - self.bounds.y1) as isize * self.row_bytes as isize)
            .offset((x - self.bounds.x1) as isize * self.n_comps as isize * 4) as *mut f32
    }
    pub fn w(&self) -> usize {
        (self.bounds.x2 - self.bounds.x1).max(0) as usize
    }
    pub fn h(&self) -> usize {
        (self.bounds.y2 - self.bounds.y1).max(0) as usize
    }
    #[inline]
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.bounds.x1 && x < self.bounds.x2 && y >= self.bounds.y1 && y < self.bounds.y2
    }
    /// Channel index holding a matte: alpha for RGBA, red (grey) for RGB, 0 for Alpha.
    pub fn matte_channel(&self) -> usize {
        match self.n_comps {
            4 => 3,
            _ => 0,
        }
    }
    /// Read channel `ch` of the rectangle `[x0,x1) x [y0,y1)` (OFX pixel coords)
    /// into a TOP-DOWN buffer (row 0 = y1-1). Pixels outside the image bounds
    /// read as `fill`; non-finite values read as 0.
    pub fn read_channel(&self, ch: usize, x0: i32, y0: i32, x1: i32, y1: i32, fill: f32) -> Vec<f32> {
        let w = (x1 - x0).max(0) as usize;
        let h = (y1 - y0).max(0) as usize;
        let mut out = vec![fill; w * h];
        if ch >= self.n_comps as usize {
            return out;
        }
        for r in 0..h {
            let y = y1 - 1 - r as i32;
            if y < self.bounds.y1 || y >= self.bounds.y2 {
                continue;
            }
            let row = &mut out[r * w..(r + 1) * w];
            let xa = x0.max(self.bounds.x1);
            let xb = x1.min(self.bounds.x2);
            for x in xa..xb {
                let v = unsafe { *self.px(x, y).add(ch) };
                row[(x - x0) as usize] = if v.is_finite() { v } else { 0.0 };
            }
        }
        out
    }
    /// Read RGB of the rectangle into a TOP-DOWN interleaved buffer (3 floats/px).
    pub fn read_rgb(&self, x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<f32> {
        let w = (x1 - x0).max(0) as usize;
        let h = (y1 - y0).max(0) as usize;
        let mut out = vec![0f32; w * h * 3];
        if self.n_comps < 3 {
            return out;
        }
        for r in 0..h {
            let y = y1 - 1 - r as i32;
            if y < self.bounds.y1 || y >= self.bounds.y2 {
                continue;
            }
            let xa = x0.max(self.bounds.x1);
            let xb = x1.min(self.bounds.x2);
            for x in xa..xb {
                let p = unsafe { self.px(x, y) };
                let o = (r * w + (x - x0) as usize) * 3;
                for k in 0..3 {
                    let v = unsafe { *p.add(k) };
                    out[o + k] = if v.is_finite() { v } else { 0.0 };
                }
            }
        }
        out
    }
}

/// Position param value (canonical coords) → top-down frame pixel coords
/// (relative to `bounds`, at render scale `rs`). Values with both |x|,|y| <= 1
/// are taken as NORMALISED (0..1 of the frame, y up): covers hosts that ignore
/// kOfxParamCoordinatesNormalised defaults and lets users type 0..1 values;
/// anything larger is canonical pixels.
pub fn canon_to_frame(x: f64, y: f64, fw: usize, fh: usize, bounds: OfxRectI, rs: (f64, f64)) -> (f64, f64) {
    if x.abs() <= 1.0 && y.abs() <= 1.0 {
        (x * fw as f64, (1.0 - y) * fh as f64)
    } else {
        (x * rs.0 - bounds.x1 as f64, bounds.y2 as f64 - y * rs.1)
    }
}

/// Display-referred 8-bit RGB (top-down, interleaved) of the frame rectangle
/// `[x0,x1) x [y0,y1)` in TOP-DOWN frame coords relative to `img.bounds`,
/// through [`crate::color::to_display`] for input colour space `cs`. Rows are
/// converted in parallel.
pub fn display_rgb_u8(img: &Image, x0: usize, y0: usize, x1: usize, y1: usize, cs: i32) -> Vec<u8> {
    let sb = img.bounds;
    let rgb = img.read_rgb(sb.x1 + x0 as i32, sb.y2 - y1 as i32, sb.x1 + x1 as i32, sb.y2 - y0 as i32);
    let mut o = vec![0u8; rgb.len()];
    let row = (x1 - x0) * 3;
    crate::resample::par_rows(&mut o, row.max(1), |r, orow| {
        let src = &rgb[r * row..r * row + orow.len()];
        for (d, p) in orow.chunks_exact_mut(3).zip(src.chunks_exact(3)) {
            let v = crate::color::to_display(p[0], p[1], p[2], cs);
            d[0] = crate::color::to_u8(v[0]);
            d[1] = crate::color::to_u8(v[1]);
            d[2] = crate::color::to_u8(v[2]);
        }
    });
    o
}

pub fn clip_handle(effect_handle: OfxImageEffectHandle, name: &[u8]) -> Option<OfxImageClipHandle> {
    let mut clip: OfxImageClipHandle = std::ptr::null_mut();
    let mut cp: OfxPropertySetHandle = std::ptr::null_mut();
    let st = unsafe { (effect().clipGetHandle)(effect_handle, name.as_ptr() as *const c_char, &mut clip, &mut cp) };
    if st != kOfxStatOK || clip.is_null() {
        return None;
    }
    Some(clip)
}

/// The clip's frame range (first, last) as the host reports it.
pub fn clip_frame_range(effect_handle: OfxImageEffectHandle, name: &[u8]) -> Option<(f64, f64)> {
    let clip = clip_handle(effect_handle, name)?;
    let mut cp: OfxPropertySetHandle = std::ptr::null_mut();
    if unsafe { (effect().clipGetPropertySet)(clip, &mut cp) } != kOfxStatOK || cp.is_null() {
        return None;
    }
    let a = crate::props::prop_try_double(cp, kOfxImageEffectPropFrameRange, 0)?;
    let b = crate::props::prop_try_double(cp, kOfxImageEffectPropFrameRange, 1)?;
    Some((a.min(b), a.max(b)))
}

/// Every time-related property the host reports on the effect, the clip and `args` (a
/// diagnostic: which of them carry the host's own frame numbers).
pub fn time_probe(effect_handle: OfxImageEffectHandle, clip: &[u8], args: OfxPropertySetHandle) -> String {
    const KEYS: [&[u8]; 8] = [
        b"OfxImageEffectPropFrameRange\0",
        b"OfxImageEffectPropUnmappedFrameRange\0",
        b"OfxImageClipPropUnmappedFrameRange\0",
        b"OfxImageEffectPropFrameRate\0",
        b"OfxImageClipPropUnmappedFrameRate\0",
        b"OfxImageEffectInstancePropEffectDuration\0",
        b"OfxImageEffectPropProjectOffset\0",
        b"OfxPropTime\0",
    ];
    let mut cp: OfxPropertySetHandle = std::ptr::null_mut();
    if let Some(c) = clip_handle(effect_handle, clip) {
        unsafe { (effect().clipGetPropertySet)(c, &mut cp) };
    }
    let mut out = String::new();
    for (what, h) in [("effect", crate::suites::effect_props(effect_handle)), ("clip", cp), ("args", args)] {
        if h.is_null() {
            continue;
        }
        for k in KEYS {
            let vals: Vec<String> = (0..2).filter_map(|i| crate::props::prop_try_double(h, k, i)).map(|v| format!("{v}")).collect();
            if !vals.is_empty() {
                let name = std::str::from_utf8(&k[..k.len() - 1]).unwrap_or("?");
                out += &format!(" {what}.{name}={}", vals.join(","));
            }
        }
    }
    out
}

pub fn clip_connected(effect_handle: OfxImageEffectHandle, name: &[u8]) -> bool {
    let Some(clip) = clip_handle(effect_handle, name) else { return false };
    let mut cp: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe { (effect().clipGetPropertySet)(clip, &mut cp) };
    !cp.is_null() && prop_get_int(cp, kOfxImageClipPropConnected, 0) != 0
}

/// Fetch the clip's image at `time` (float RGBA / RGB / Alpha only). Must be
/// paired with [`release`].
pub fn fetch_image(effect_handle: OfxImageEffectHandle, name: &[u8], time: f64) -> Option<Image> {
    let clip = clip_handle(effect_handle, name)?;
    let mut ip: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        if (effect().clipGetImage)(clip, time, std::ptr::null(), &mut ip) != kOfxStatOK || ip.is_null() {
            return None;
        }
    }
    let data = prop_get_pointer(ip, kOfxImagePropData, 0) as *mut u8;
    let bounds = prop_get_rect_i(ip, kOfxImagePropBounds);
    let row_bytes = prop_get_int(ip, kOfxImagePropRowBytes, 0);
    let comps = prop_get_string(ip, kOfxImageEffectPropComponents, 0);
    let n_comps = match comps.as_str() {
        "OfxImageComponentRGBA" => 4,
        "OfxImageComponentRGB" => 3,
        "OfxImageComponentAlpha" => 1,
        _ => 0,
    };
    let depth = prop_get_string(ip, kOfxImageEffectPropPixelDepth, 0);
    if data.is_null() || n_comps == 0 || (depth != "OfxBitDepthFloat" && !depth.is_empty()) {
        crate::log::log(&format!("unsupported image on clip: comps={comps} depth={depth}"));
        unsafe { (effect().clipReleaseImage)(ip) };
        return None;
    }
    Some(Image { props: ip, data, bounds, row_bytes, n_comps })
}

pub fn release(img: &Image) {
    unsafe { (effect().clipReleaseImage)(img.props) };
}

/// Box-downscale channels `chans` of an OFX image (bottom-up) to a top-down
/// interleaved buffer of size dw×dh.
pub fn downscale_ofx(img: &Image, chans: &[usize], dw: usize, dh: usize) -> Vec<f32> {
    let (sw, sh) = (img.w(), img.h());
    let c = chans.len();
    let mut out = vec![0f32; dw * dh * c];
    let fx = sw as f64 / dw as f64;
    let fy = sh as f64 / dh as f64;
    for oy in 0..dh {
        let r0 = (oy as f64 * fy).floor() as usize;
        let r1 = (((oy + 1) as f64 * fy).ceil() as usize).clamp(r0 + 1, sh);
        for ox in 0..dw {
            let c0 = (ox as f64 * fx).floor() as usize;
            let c1 = (((ox + 1) as f64 * fx).ceil() as usize).clamp(c0 + 1, sw);
            let mut acc = [0f64; 4];
            for r in r0..r1 {
                // top-down row r -> OFX y
                let y = img.bounds.y2 - 1 - r as i32;
                for col in c0..c1 {
                    let x = img.bounds.x1 + col as i32;
                    let p = unsafe { img.px(x, y) };
                    for (k, &ch) in chans.iter().enumerate() {
                        let v = unsafe { *p.add(ch) };
                        acc[k] += if v.is_finite() { v as f64 } else { 0.0 };
                    }
                }
            }
            let n = ((r1 - r0) * (c1 - c0)) as f64;
            for k in 0..c {
                out[(oy * dw + ox) * c + k] = (acc[k] / n) as f32;
            }
        }
    }
    out
}

/// Clip-preference key for clip `name`: `OfxImageClipPropComponents_<name>\0`.
pub fn clip_pref_components_key(name: &str) -> Vec<u8> {
    format!("OfxImageClipPropComponents_{name}\0").into_bytes()
}
/// Clip-preference key for clip `name`: `OfxImageClipPropDepth_<name>\0`.
pub fn clip_pref_depth_key(name: &str) -> Vec<u8> {
    format!("OfxImageClipPropDepth_{name}\0").into_bytes()
}
