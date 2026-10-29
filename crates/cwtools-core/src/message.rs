//! Node messages.

use crate::ofx_api::*;
use crate::suites::suites;
use std::os::raw::c_char;

/// Post (or clear, with `None`) the node's persistent message. Uses
/// MessageSuite V2's setPersistentMessage only — the V1 `message()` can pop a
/// modal dialog mid-render, which we never want. Always logged as well. The
/// message id is the plugin name ([`crate::log::name`]).
pub fn set_node_message(effect_handle: OfxImageEffectHandle, text: Option<&str>) {
    let s = suites();
    if let Some(t) = text {
        crate::log::log(t);
    }
    if s.message.is_null() || !s.message_v2 {
        return;
    }
    let ms = unsafe { &*s.message };
    unsafe {
        match text {
            Some(t) => {
                let c = std::ffi::CString::new(t.replace('\0', " ")).unwrap_or_default();
                let id = std::ffi::CString::new(crate::log::name()).unwrap_or_default();
                cw_message(
                    ms.setPersistentMessage,
                    effect_handle,
                    kOfxMessageWarning.as_ptr() as *const c_char,
                    id.as_ptr(),
                    c.as_ptr(),
                );
            }
            None => {
                (ms.clearPersistentMessage)(effect_handle);
            }
        }
    }
}

/// A modal OK dialog (MessageSuite `message`). Only from instanceChanged (a button
/// press), never during a render. Logged as well.
pub fn dialog(effect_handle: OfxImageEffectHandle, text: &str) {
    crate::log::log(text);
    let s = suites();
    if s.message.is_null() {
        return;
    }
    let ms = unsafe { &*s.message };
    let c = std::ffi::CString::new(text.replace('\0', " ")).unwrap_or_default();
    let id = std::ffi::CString::new(crate::log::name()).unwrap_or_default();
    unsafe {
        cw_message(ms.message, effect_handle, kOfxMessageMessage.as_ptr() as *const c_char, id.as_ptr(), c.as_ptr());
    }
}
