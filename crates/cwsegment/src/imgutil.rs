//! Codecs for the remote protocol (JPEG frame up, PNG mask down) + re-exports
//! of the core pixel helpers.
//! All buffers here are TOP-DOWN, row-major (OFX images are bottom-up; render.rs
//! flips at the boundary).

// Buffers, resamplers, hashes and the display transform moved to
// cwtools-core; re-exported here so existing paths keep working.
pub use cwtools_core::color::{rgbf_to_display8, to_display, CS_ACESCCT, CS_ACESCG, CS_LIN709, CS_VIDEO};
pub use cwtools_core::pixels::{area_resize, bilinear_gray, fit_size, fnv1a, frame_key, Gray, Rgb8, RgbF};

pub fn encode_jpeg(img: &Rgb8, quality: u8) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let enc = jpeg_encoder::Encoder::new(&mut out, quality);
    enc.encode(&img.data, img.w as u16, img.h as u16, jpeg_encoder::ColorType::Rgb)
        .map_err(|e| format!("jpeg encode: {e}"))?;
    Ok(out)
}

pub fn encode_png_gray(m: &Gray) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, m.w as u32, m.h as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().map_err(|e| e.to_string())?;
        let bytes: Vec<u8> = m.data.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
        wr.write_image_data(&bytes).map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Decode a PNG to a single-channel [0,1] image (first channel, or alpha if
/// the PNG is gray+alpha / RGBA and `prefer_alpha`).
pub fn decode_png_gray(bytes: &[u8]) -> Result<Gray, String> {
    let mut dec = png::Decoder::new(bytes);
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut rd = dec.read_info().map_err(|e| format!("png: {e}"))?;
    let mut buf = vec![0u8; rd.output_buffer_size()];
    let info = rd.next_frame(&mut buf).map_err(|e| format!("png: {e}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let ch = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => 3,
    };
    let stride = info.line_size;
    let mut data = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            data[y * w + x] = buf[y * stride + x * ch] as f32 / 255.0;
        }
    }
    Ok(Gray { w, h, data })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn png_roundtrip() {
        let m = Gray { w: 3, h: 2, data: vec![0.0, 1.0, 0.5, 1.0, 0.0, 0.25] };
        let b = encode_png_gray(&m).unwrap();
        let d = decode_png_gray(&b).unwrap();
        assert_eq!((d.w, d.h), (3, 2));
        assert!((d.data[2] - 0.5).abs() < 0.01);
    }
}
