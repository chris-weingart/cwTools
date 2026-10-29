//! Host suite pointers (fetched once in `OfxActionLoad`) and accessors.

use crate::ofx_api::*;
use crate::props::prop_get_string;
use std::os::raw::c_char;

pub struct Suites {
    pub host: *mut OfxHost,
    pub prop: *const OfxPropertySuiteV1,
    pub effect: *const OfxImageEffectSuiteV1,
    pub param: *const OfxParameterSuiteV1,
    pub message: *const OfxMessageSuiteV2, // may be null
    pub message_v2: bool,
    pub interact: *const OfxInteractSuiteV1, // may be null
    pub progress: *const OfxProgressSuiteV1, // may be null
}

static mut SUITES: Suites = Suites {
    host: std::ptr::null_mut(),
    prop: std::ptr::null(),
    effect: std::ptr::null(),
    param: std::ptr::null(),
    message: std::ptr::null(),
    message_v2: false,
    interact: std::ptr::null(),
    progress: std::ptr::null(),
};

#[inline]
pub fn suites() -> &'static Suites {
    unsafe { &*std::ptr::addr_of!(SUITES) }
}
#[allow(clippy::mut_from_ref)]
pub fn suites_mut() -> &'static mut Suites {
    unsafe { &mut *std::ptr::addr_of_mut!(SUITES) }
}
#[inline]
pub fn prop() -> &'static OfxPropertySuiteV1 {
    unsafe { &*suites().prop }
}
#[inline]
pub fn effect() -> &'static OfxImageEffectSuiteV1 {
    unsafe { &*suites().effect }
}
#[inline]
pub fn param() -> &'static OfxParameterSuiteV1 {
    unsafe { &*suites().param }
}

/// `OfxPlugin.setHost` implementation: remember the host.
pub unsafe extern "C" fn set_host(host: *mut OfxHost) {
    suites_mut().host = host;
}

/// Generic `OfxActionLoad`: fetch property / image-effect / parameter suites
/// (mandatory) and message V2 (else V1) + interact suites (optional).
pub fn load_suites() -> OfxStatus {
    let host = suites().host;
    if host.is_null() {
        return kOfxStatErrBadHandle;
    }
    let host_ref = unsafe { &*host };
    let fetch = match host_ref.fetchSuite {
        Some(f) => f,
        None => return kOfxStatErrMissingHostFeature,
    };
    unsafe {
        let prop = fetch(host_ref.host, kOfxPropertySuite.as_ptr() as *const c_char, 1);
        let eff = fetch(host_ref.host, kOfxImageEffectSuite.as_ptr() as *const c_char, 1);
        let par = fetch(host_ref.host, kOfxParameterSuite.as_ptr() as *const c_char, 1);
        if prop.is_null() || eff.is_null() || par.is_null() {
            return kOfxStatErrMissingHostFeature;
        }
        let s = suites_mut();
        s.prop = prop as *const OfxPropertySuiteV1;
        s.effect = eff as *const OfxImageEffectSuiteV1;
        s.param = par as *const OfxParameterSuiteV1;
        // Optional suites.
        let m2 = fetch(host_ref.host, kOfxMessageSuite.as_ptr() as *const c_char, 2);
        if !m2.is_null() {
            s.message = m2 as *const OfxMessageSuiteV2;
            s.message_v2 = true;
        } else {
            let m1 = fetch(host_ref.host, kOfxMessageSuite.as_ptr() as *const c_char, 1);
            s.message = m1 as *const OfxMessageSuiteV2;
            s.message_v2 = false;
        }
        let it = fetch(host_ref.host, kOfxInteractSuite.as_ptr() as *const c_char, 1);
        s.interact = it as *const OfxInteractSuiteV1;
        s.progress = fetch(host_ref.host, kOfxProgressSuite.as_ptr() as *const c_char, 1) as *const OfxProgressSuiteV1;
        crate::log::log(&format!(
            "load: cwTools {} host={} message_v2={} interact_suite={} progress_suite={}",
            crate::version::label_text(),
            prop_get_string(host_ref.host, kOfxPropName, 0),
            s.message_v2,
            !s.interact.is_null(),
            !s.progress.is_null()
        ));
    }
    kOfxStatOK
}

/// The effect instance's parameter set.
pub fn effect_param_set(effect_handle: OfxImageEffectHandle) -> OfxParamSetHandle {
    let mut ps: OfxParamSetHandle = std::ptr::null_mut();
    unsafe {
        (effect().getParamSet)(effect_handle, &mut ps);
    }
    ps
}

/// The effect instance's property set.
pub fn effect_props(effect_handle: OfxImageEffectHandle) -> OfxPropertySetHandle {
    let mut p: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        (effect().getPropertySet)(effect_handle, &mut p);
    }
    p
}

/// True when the effect's host asks us to abort the current render.
pub fn aborted(effect_handle: OfxImageEffectHandle) -> bool {
    unsafe { (effect().abort)(effect_handle) != 0 }
}

/// Host progress bar for a long render step (OFX progress suite; a no-op when the
/// host has none). Ends when dropped.
pub struct Progress {
    eh: OfxImageEffectHandle,
    on: bool,
}

impl Progress {
    pub fn start(eh: OfxImageEffectHandle, label: &str) -> Progress {
        let p = suites().progress;
        let mut on = false;
        if !p.is_null() {
            let c = std::ffi::CString::new(label.replace('\0', " ")).unwrap_or_default();
            on = unsafe { ((*p).progressStart)(eh as *mut std::os::raw::c_void, c.as_ptr()) } == kOfxStatOK;
        }
        Progress { eh, on }
    }
    /// Report `f` (0..1). false = the user cancelled (progress dialog or the host's abort).
    pub fn update(&self, f: f64) -> bool {
        if unsafe { (effect().abort)(self.eh) } != 0 {
            return false;
        }
        let p = suites().progress;
        if self.on && !p.is_null() {
            return unsafe { ((*p).progressUpdate)(self.eh as *mut std::os::raw::c_void, f.clamp(0.0, 1.0)) } != kOfxStatReplyNo;
        }
        true
    }
    pub fn shown(&self) -> bool {
        self.on
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        let p = suites().progress;
        if self.on && !p.is_null() {
            unsafe { ((*p).progressEnd)(self.eh as *mut std::os::raw::c_void) };
        }
    }
}
