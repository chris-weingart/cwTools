//! IEEE half (OfxBitDepthHalf) conversion. cwFill keeps its own copy
//! (crates/cwfill is a separate workspace); keep the two identical.

/// IEEE half → f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            sign
        } else {
            // subnormal: normalise the mantissa
            let (mut e2, mut m2) = (113u32, m);
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            sign | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        sign | 0x7f80_0000 | (m << 13)
    } else {
        sign | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// f32 → IEEE half, round to nearest even; overflow → inf, NaN stays NaN.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let e = ((x >> 23) & 0xff) as i32;
    let m = x & 0x7f_ffff;
    if e == 255 {
        return sign | 0x7c00 | if m != 0 { 0x200 | (m >> 13) as u16 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return sign | 0x7c00;
    }
    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e16) as u32;
        let (q, rem, half) = (m >> shift, m & ((1 << shift) - 1), 1u32 << (shift - 1));
        let r = q + (rem > half || (rem == half && q & 1 == 1)) as u32;
        return sign | r as u16;
    }
    let q = ((e16 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    let r = q + (rem > 0x1000 || (rem == 0x1000 && q & 1 == 1)) as u32; // a carry rolls into the exponent (to inf at the top)
    sign | r as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_roundtrip_is_exact_for_every_finite_half() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f32_to_f16(f) & 0x7c00 == 0x7c00 && f32_to_f16(f) & 0x3ff != 0, "{h:#06x} NaN");
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "{h:#06x} -> {f}");
        }
    }

    #[test]
    fn f32_to_half_rounds_to_nearest_even() {
        for (f, h) in [
            (1.0f32, 0x3c00u16), (-2.0, 0xc000), (0.1, 0x2e66), (65504.0, 0x7bff), (65519.0, 0x7bff),
            (65520.0, 0x7c00), (1e9, 0x7c00), (5.960_464_5e-8, 0x0001), (2.0e-8, 0x0000), (0.0, 0x0000),
            (-0.0, 0x8000), (f32::INFINITY, 0x7c00), (f32::NEG_INFINITY, 0xfc00),
        ] {
            assert_eq!(f32_to_f16(f), h, "{f}");
        }
        // Midpoints between neighbouring halves (exact in f32) go to the even one,
        // normal and subnormal ranges alike.
        for h in (1u16..0x7bff).step_by(7) {
            let mid = (f16_to_f32(h) + f16_to_f32(h + 1)) * 0.5;
            assert_eq!(f32_to_f16(mid), if h & 1 == 0 { h } else { h + 1 }, "mid {h:#06x}");
        }
    }
}
