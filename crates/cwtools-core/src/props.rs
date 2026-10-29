//! Property-suite helpers. Keys and string values are NUL-terminated byte
//! strings (`b"OfxPropLabel\0"`), as the C API expects.

use crate::ofx_api::*;
use crate::suites::prop;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

pub fn prop_set_string(h: OfxPropertySetHandle, key: &[u8], index: c_int, val: &[u8]) {
    unsafe {
        (prop().propSetString)(h, key.as_ptr() as *const c_char, index, val.as_ptr() as *const c_char);
    }
}
pub fn prop_set_int(h: OfxPropertySetHandle, key: &[u8], index: c_int, val: c_int) {
    unsafe {
        (prop().propSetInt)(h, key.as_ptr() as *const c_char, index, val);
    }
}
pub fn prop_set_double(h: OfxPropertySetHandle, key: &[u8], index: c_int, val: f64) {
    unsafe {
        (prop().propSetDouble)(h, key.as_ptr() as *const c_char, index, val);
    }
}
pub fn prop_set_pointer(h: OfxPropertySetHandle, key: &[u8], index: c_int, val: *mut c_void) {
    unsafe {
        (prop().propSetPointer)(h, key.as_ptr() as *const c_char, index, val);
    }
}
pub fn prop_get_int(h: OfxPropertySetHandle, key: &[u8], index: c_int) -> c_int {
    let mut out: c_int = 0;
    unsafe {
        (prop().propGetInt)(h, key.as_ptr() as *const c_char, index, &mut out);
    }
    out
}
/// Like prop_get_int but returns None when the host reports an error.
pub fn prop_try_int(h: OfxPropertySetHandle, key: &[u8], index: c_int) -> Option<c_int> {
    let mut out: c_int = 0;
    let st = unsafe { (prop().propGetInt)(h, key.as_ptr() as *const c_char, index, &mut out) };
    if st == kOfxStatOK {
        Some(out)
    } else {
        None
    }
}
pub fn prop_get_pointer(h: OfxPropertySetHandle, key: &[u8], index: c_int) -> *mut c_void {
    let mut out: *mut c_void = std::ptr::null_mut();
    unsafe {
        (prop().propGetPointer)(h, key.as_ptr() as *const c_char, index, &mut out);
    }
    out
}
pub fn prop_get_string(h: OfxPropertySetHandle, key: &[u8], index: c_int) -> String {
    let mut out: *mut c_char = std::ptr::null_mut();
    unsafe {
        (prop().propGetString)(h, key.as_ptr() as *const c_char, index, &mut out);
        if out.is_null() {
            return String::new();
        }
        CStr::from_ptr(out).to_string_lossy().into_owned()
    }
}
pub fn prop_get_rect_i(h: OfxPropertySetHandle, key: &[u8]) -> OfxRectI {
    OfxRectI {
        x1: prop_get_int(h, key, 0),
        y1: prop_get_int(h, key, 1),
        x2: prop_get_int(h, key, 2),
        y2: prop_get_int(h, key, 3),
    }
}
pub fn prop_get_double(h: OfxPropertySetHandle, key: &[u8], index: c_int) -> f64 {
    let mut out: f64 = 0.0;
    unsafe {
        (prop().propGetDouble)(h, key.as_ptr() as *const c_char, index, &mut out);
    }
    out
}
/// Like prop_get_double but None on host error / missing property.
pub fn prop_try_double(h: OfxPropertySetHandle, key: &[u8], index: c_int) -> Option<f64> {
    let mut out: f64 = 0.0;
    let st = unsafe { (prop().propGetDouble)(h, key.as_ptr() as *const c_char, index, &mut out) };
    if st == kOfxStatOK {
        Some(out)
    } else {
        None
    }
}
