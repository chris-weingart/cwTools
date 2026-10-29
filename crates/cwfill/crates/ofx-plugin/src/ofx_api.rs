//! Hand-written FFI for the subset of the OpenFX 1.4/1.5 C API this plugin
//! uses. Mirrors the abTools-Defocus approach (hand-defined types, exact field
//! order from the OFX headers) rather than depending on the unmaintained
//! `ofx-rs` crate.
//!
//! Nothing here links against an OFX SDK: an `.ofx` plugin is a C-ABI shared
//! library that the host loads. The host hands us its `OfxHost`, and we fetch
//! suites (tables of function pointers) at runtime. So these declarations only
//! have to match the host's ABI, which is the published OFX standard.
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]

use std::os::raw::{c_char, c_double, c_int, c_void};

// ---------------------------------------------------------------------------
// Core scalar types & handles
// ---------------------------------------------------------------------------
pub type OfxStatus = c_int;
pub type OfxTime = c_double;

// All OFX handles are opaque blind pointers.
pub type OfxPropertySetHandle = *mut c_void;
pub type OfxParamSetHandle = *mut c_void;
pub type OfxParamHandle = *mut c_void;
pub type OfxImageEffectHandle = *mut c_void;
pub type OfxImageClipHandle = *mut c_void;
pub type OfxImageMemoryHandle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OfxRectI {
    pub x1: c_int,
    pub y1: c_int,
    pub x2: c_int,
    pub y2: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OfxRectD {
    pub x1: c_double,
    pub y1: c_double,
    pub x2: c_double,
    pub y2: c_double,
}

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------
pub const kOfxStatOK: OfxStatus = 0;
pub const kOfxStatFailed: OfxStatus = 1;
pub const kOfxStatErrFatal: OfxStatus = 2;
pub const kOfxStatErrUnknown: OfxStatus = 3;
pub const kOfxStatErrMissingHostFeature: OfxStatus = 4;
pub const kOfxStatErrUnsupported: OfxStatus = 5;
pub const kOfxStatErrExists: OfxStatus = 6;
pub const kOfxStatErrFormat: OfxStatus = 7;
pub const kOfxStatErrMemory: OfxStatus = 8;
pub const kOfxStatErrBadHandle: OfxStatus = 9;
pub const kOfxStatErrBadIndex: OfxStatus = 10;
pub const kOfxStatErrValue: OfxStatus = 11;
pub const kOfxStatReplyYes: OfxStatus = 12;
pub const kOfxStatReplyNo: OfxStatus = 13;
pub const kOfxStatReplyDefault: OfxStatus = 14;

// ---------------------------------------------------------------------------
// Host & plugin structs
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct OfxHost {
    pub host: OfxPropertySetHandle,
    pub fetchSuite: Option<
        unsafe extern "C" fn(
            host: OfxPropertySetHandle,
            suiteName: *const c_char,
            suiteVersion: c_int,
        ) -> *const c_void,
    >,
}

pub type SetHostFn = unsafe extern "C" fn(host: *mut OfxHost);
pub type MainEntryFn = unsafe extern "C" fn(
    action: *const c_char,
    handle: *const c_void,
    inArgs: OfxPropertySetHandle,
    outArgs: OfxPropertySetHandle,
) -> OfxStatus;

#[repr(C)]
pub struct OfxPlugin {
    pub pluginApi: *const c_char,
    pub apiVersion: c_int,
    pub pluginIdentifier: *const c_char,
    pub pluginVersionMajor: c_uint_,
    pub pluginVersionMinor: c_uint_,
    pub setHost: Option<SetHostFn>,
    pub mainEntry: Option<MainEntryFn>,
}
pub type c_uint_ = std::os::raw::c_uint;

// OfxPlugin holds raw C string pointers to 'static data — safe to share.
unsafe impl Sync for OfxPlugin {}

// ---------------------------------------------------------------------------
// Property suite (fixed-signature — callable directly from Rust)
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct OfxPropertySuiteV1 {
    pub propSetPointer: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut c_void) -> OfxStatus,
    pub propSetString: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *const c_char) -> OfxStatus,
    pub propSetDouble: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, c_double) -> OfxStatus,
    pub propSetInt: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, c_int) -> OfxStatus,
    pub propSetPointerN: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *const *mut c_void) -> OfxStatus,
    pub propSetStringN: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *const *const c_char) -> OfxStatus,
    pub propSetDoubleN: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *const c_double) -> OfxStatus,
    pub propSetIntN: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *const c_int) -> OfxStatus,
    pub propGetPointer: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut *mut c_void) -> OfxStatus,
    pub propGetString: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut *mut c_char) -> OfxStatus,
    pub propGetDouble: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut c_double) -> OfxStatus,
    pub propGetInt: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut c_int) -> OfxStatus,
    pub propGetPointerN: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut *mut c_void) -> OfxStatus,
    pub propGetStringN: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut *mut c_char) -> OfxStatus,
    pub propGetDoubleN: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut c_double) -> OfxStatus,
    pub propGetIntN: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, c_int, *mut c_int) -> OfxStatus,
    pub propReset: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char) -> OfxStatus,
    pub propGetDimension: unsafe extern "C" fn(OfxPropertySetHandle, *const c_char, *mut c_int) -> OfxStatus,
}

// ---------------------------------------------------------------------------
// Image effect suite (fixed-signature — callable directly from Rust)
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct OfxImageEffectSuiteV1 {
    pub getPropertySet: unsafe extern "C" fn(OfxImageEffectHandle, *mut OfxPropertySetHandle) -> OfxStatus,
    pub getParamSet: unsafe extern "C" fn(OfxImageEffectHandle, *mut OfxParamSetHandle) -> OfxStatus,
    pub clipDefine: unsafe extern "C" fn(OfxImageEffectHandle, *const c_char, *mut OfxPropertySetHandle) -> OfxStatus,
    pub clipGetHandle: unsafe extern "C" fn(OfxImageEffectHandle, *const c_char, *mut OfxImageClipHandle, *mut OfxPropertySetHandle) -> OfxStatus,
    pub clipGetPropertySet: unsafe extern "C" fn(OfxImageClipHandle, *mut OfxPropertySetHandle) -> OfxStatus,
    pub clipGetImage: unsafe extern "C" fn(OfxImageClipHandle, OfxTime, *const OfxRectD, *mut OfxPropertySetHandle) -> OfxStatus,
    pub clipReleaseImage: unsafe extern "C" fn(OfxPropertySetHandle) -> OfxStatus,
    pub clipGetRegionOfDefinition: unsafe extern "C" fn(OfxImageClipHandle, OfxTime, *mut OfxRectD) -> OfxStatus,
    pub abort: unsafe extern "C" fn(OfxImageEffectHandle) -> c_int,
    pub imageMemoryAlloc: unsafe extern "C" fn(OfxImageEffectHandle, usize, *mut OfxImageMemoryHandle) -> OfxStatus,
    pub imageMemoryFree: unsafe extern "C" fn(OfxImageMemoryHandle) -> OfxStatus,
    pub imageMemoryLock: unsafe extern "C" fn(OfxImageMemoryHandle, *mut *mut c_void) -> OfxStatus,
    pub imageMemoryUnlock: unsafe extern "C" fn(OfxImageMemoryHandle) -> OfxStatus,
}

// ---------------------------------------------------------------------------
// Parameter suite. paramGetValue/paramSetValue are VARIADIC, so we store them
// as opaque pointers and route through the C shim (variadic_helpers.c). The
// rest are fixed-signature.
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct OfxParameterSuiteV1 {
    pub paramDefine: unsafe extern "C" fn(OfxParamSetHandle, *const c_char, *const c_char, *mut OfxPropertySetHandle) -> OfxStatus,
    pub paramGetHandle: unsafe extern "C" fn(OfxParamSetHandle, *const c_char, *mut OfxParamHandle, *mut OfxPropertySetHandle) -> OfxStatus,
    pub paramSetGetPropertySet: unsafe extern "C" fn(OfxParamSetHandle, *mut OfxPropertySetHandle) -> OfxStatus,
    pub paramGetPropertySet: unsafe extern "C" fn(OfxParamHandle, *mut OfxPropertySetHandle) -> OfxStatus,
    pub paramGetValue: *const c_void,         // variadic
    pub paramGetValueAtTime: *const c_void,   // variadic
    pub paramGetDerivative: *const c_void,    // variadic
    pub paramGetIntegral: *const c_void,      // variadic
    pub paramSetValue: *const c_void,         // variadic
    pub paramSetValueAtTime: *const c_void,   // variadic
    pub paramGetNumKeys: unsafe extern "C" fn(OfxParamHandle, *mut c_uint_) -> OfxStatus,
    pub paramGetKeyTime: unsafe extern "C" fn(OfxParamHandle, c_uint_, *mut OfxTime) -> OfxStatus,
    pub paramGetKeyIndex: unsafe extern "C" fn(OfxParamHandle, OfxTime, c_int, *mut c_int) -> OfxStatus,
    pub paramDeleteKey: unsafe extern "C" fn(OfxParamHandle, OfxTime) -> OfxStatus,
    pub paramDeleteAllKeys: unsafe extern "C" fn(OfxParamHandle) -> OfxStatus,
    pub paramCopy: unsafe extern "C" fn(OfxParamHandle, OfxParamHandle, OfxTime, *const OfxRectD) -> OfxStatus,
    pub paramEditBegin: unsafe extern "C" fn(OfxParamSetHandle, *const c_char) -> OfxStatus,
    pub paramEditEnd: unsafe extern "C" fn(OfxParamSetHandle) -> OfxStatus,
}

// ---------------------------------------------------------------------------
// C shim bindings (variadic_helpers.c)
// ---------------------------------------------------------------------------
extern "C" {
    pub fn cw_param_get_double(f: *const c_void, handle: OfxParamHandle, out: *mut c_double) -> OfxStatus;
    pub fn cw_param_get_int(f: *const c_void, handle: OfxParamHandle, out: *mut c_int) -> OfxStatus;
    pub fn cw_param_get_rgb(
        f: *const c_void,
        handle: OfxParamHandle,
        r: *mut c_double,
        g: *mut c_double,
        b: *mut c_double,
    ) -> OfxStatus;
    pub fn cw_param_get_string(f: *const c_void, handle: OfxParamHandle, out: *mut *mut c_char) -> OfxStatus;
    pub fn cw_param_set_double(f: *const c_void, handle: OfxParamHandle, v: c_double) -> OfxStatus;
    pub fn cw_param_set_int(f: *const c_void, handle: OfxParamHandle, v: c_int) -> OfxStatus;
}

// ---------------------------------------------------------------------------
// String constants (NUL-terminated, as the C API expects). Grouped by use.
// ---------------------------------------------------------------------------
pub const kOfxImageEffectPluginApi: &[u8] = b"OfxImageEffectPluginAPI\0";
pub const kOfxImageEffectPluginApiVersion: c_int = 1;

// Suite names
pub const kOfxPropertySuite: &[u8] = b"OfxPropertySuite\0";
pub const kOfxImageEffectSuite: &[u8] = b"OfxImageEffectSuite\0";
pub const kOfxParameterSuite: &[u8] = b"OfxParameterSuite\0";

// Actions
pub const kOfxActionLoad: &[u8] = b"OfxActionLoad\0";
pub const kOfxActionUnload: &[u8] = b"OfxActionUnload\0";
pub const kOfxActionDescribe: &[u8] = b"OfxActionDescribe\0";
pub const kOfxActionCreateInstance: &[u8] = b"OfxActionCreateInstance\0";
pub const kOfxActionDestroyInstance: &[u8] = b"OfxActionDestroyInstance\0";
pub const kOfxActionInstanceChanged: &[u8] = b"OfxActionInstanceChanged\0";
pub const kOfxPropChangeReason: &[u8] = b"OfxPropChangeReason\0";
pub const kOfxImageEffectActionDescribeInContext: &[u8] = b"OfxImageEffectActionDescribeInContext\0";
pub const kOfxImageEffectActionRender: &[u8] = b"OfxImageEffectActionRender\0";
pub const kOfxImageEffectActionIsIdentity: &[u8] = b"OfxImageEffectActionIsIdentity\0";
pub const kOfxImageEffectActionGetClipPreferences: &[u8] = b"OfxImageEffectActionGetClipPreferences\0";

// Image effect / plugin props
pub const kOfxPropType: &[u8] = b"OfxPropType\0";
pub const kOfxPropName: &[u8] = b"OfxPropName\0";
pub const kOfxPropLabel: &[u8] = b"OfxPropLabel\0";
pub const kOfxPropShortLabel: &[u8] = b"OfxPropShortLabel\0";
pub const kOfxPropLongLabel: &[u8] = b"OfxPropLongLabel\0";
pub const kOfxPropIcon: &[u8] = b"OfxPropIcon\0";
pub const kOfxPropPluginDescription: &[u8] = b"OfxImageEffectPropPluginDescription\0";
pub const kOfxImageEffectPropSupportedContexts: &[u8] = b"OfxImageEffectPropSupportedContexts\0";
pub const kOfxImageEffectPropContext: &[u8] = b"OfxImageEffectPropContext\0";
pub const kOfxImageEffectPropSupportedPixelDepths: &[u8] = b"OfxImageEffectPropSupportedPixelDepths\0";
pub const kOfxImageEffectPropSupportsTiles: &[u8] = b"OfxImageEffectPropSupportsTiles\0";
pub const kOfxImageEffectPluginPropGrouping: &[u8] = b"OfxImageEffectPluginPropGrouping\0";

// Contexts
pub const kOfxImageEffectContextGeneral: &[u8] = b"OfxImageEffectContextGeneral\0";
pub const kOfxImageEffectContextFilter: &[u8] = b"OfxImageEffectContextFilter\0";

// Clip props
pub const kOfxImageEffectPropComponents: &[u8] = b"OfxImageEffectPropComponents\0";
// Clip DESCRIPTOR property: the set of components a clip accepts. Distinct from
// PropComponents above (an IMAGE's actual components). A clip defined without
// this gets no input socket in Flame.
pub const kOfxImageEffectPropSupportedComponents: &[u8] = b"OfxImageEffectPropSupportedComponents\0";
pub const kOfxImageEffectPropPixelDepth: &[u8] = b"OfxImageEffectPropPixelDepth\0";
pub const kOfxImageClipPropConnected: &[u8] = b"OfxImageClipPropConnected\0";
pub const kOfxImageEffectPropSupportsMultipleClipDepths: &[u8] = b"OfxImageEffectPropMultipleClipDepths\0";
pub const kOfxImageClipPropOptional: &[u8] = b"OfxImageClipPropOptional\0";

// Components / pixel depths
pub const kOfxImageComponentRGBA: &[u8] = b"OfxImageComponentRGBA\0";
pub const kOfxImageComponentRGB: &[u8] = b"OfxImageComponentRGB\0";
pub const kOfxImageComponentAlpha: &[u8] = b"OfxImageComponentAlpha\0";
pub const kOfxBitDepthFloat: &[u8] = b"OfxBitDepthFloat\0";
pub const kOfxBitDepthHalf: &[u8] = b"OfxBitDepthHalf\0";

// Image props (in render)
pub const kOfxImagePropData: &[u8] = b"OfxImagePropData\0";
pub const kOfxImagePropBounds: &[u8] = b"OfxImagePropBounds\0";
pub const kOfxImagePropRowBytes: &[u8] = b"OfxImagePropRowBytes\0";
pub const kOfxImageEffectPropRenderWindow: &[u8] = b"OfxImageEffectPropRenderWindow\0";
pub const kOfxPropTime: &[u8] = b"OfxPropTime\0";

// Clip names (our plugin's two inputs + output)
pub const kOfxImageEffectSimpleSourceClipName: &[u8] = b"Source\0";
pub const kOfxImageEffectOutputClipName: &[u8] = b"Output\0";

// Parameter types
pub const kOfxParamTypeDouble: &[u8] = b"OfxParamTypeDouble\0";
pub const kOfxParamTypeInteger: &[u8] = b"OfxParamTypeInteger\0";
pub const kOfxParamTypeBoolean: &[u8] = b"OfxParamTypeBoolean\0";
pub const kOfxParamTypeChoice: &[u8] = b"OfxParamTypeChoice\0";
pub const kOfxParamTypeGroup: &[u8] = b"OfxParamTypeGroup\0";
pub const kOfxParamTypeRGB: &[u8] = b"OfxParamTypeRGB\0";

// Parameter props
pub const kOfxParamPropDefault: &[u8] = b"OfxParamPropDefault\0";
pub const kOfxParamPropMin: &[u8] = b"OfxParamPropMin\0";
pub const kOfxParamPropMax: &[u8] = b"OfxParamPropMax\0";
pub const kOfxParamPropDisplayMin: &[u8] = b"OfxParamPropDisplayMin\0";
pub const kOfxParamPropDisplayMax: &[u8] = b"OfxParamPropDisplayMax\0";
pub const kOfxParamPropParent: &[u8] = b"OfxParamPropParent\0";
pub const kOfxParamPropChoiceOption: &[u8] = b"OfxParamPropChoiceOption\0";
pub const kOfxParamPropHint: &[u8] = b"OfxParamPropHint\0";
pub const kOfxParamPropIncrement: &[u8] = b"OfxParamPropIncrement\0";
pub const kOfxParamPropDigits: &[u8] = b"OfxParamPropDigits\0";
