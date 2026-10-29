//! cwTools suite version. One version for every plugin (workspace
//! `[workspace.package] version`), plus the build id (`git describe`, or the
//! BUILD_ID file in a build copy without .git). Shown in Flame as the read-only
//! "Version" param, in the OFX version props and in the plugin log at load.

use crate::ofx_api::*;
use crate::params::{define, hint, label};
use crate::props::{prop_set_int, prop_set_string};
use std::os::raw::c_int;

pub const SUITE: &str = env!("CARGO_PKG_VERSION");
pub const BUILD_ID: &str = env!("CWTOOLS_BUILD_ID");
pub const MAJOR: u32 = parse_u32(env!("CARGO_PKG_VERSION_MAJOR"));
pub const MINOR: u32 = parse_u32(env!("CARGO_PKG_VERSION_MINOR"));
pub const PATCH: u32 = parse_u32(env!("CARGO_PKG_VERSION_PATCH"));

pub const P_VERSION: &[u8] = b"cwtoolsVersion\0";

const fn parse_u32(s: &str) -> u32 {
    let b = s.as_bytes();
    let mut i = 0;
    let mut v = 0u32;
    while i < b.len() {
        v = v * 10 + (b[i] - b'0') as u32;
        i += 1;
    }
    v
}

/// "0.2.0 (v0.2.0-3-gabc1234-dirty)"; just "0.2.0" when the build id is the tag itself.
pub fn label_text() -> String {
    if BUILD_ID == format!("v{SUITE}") {
        SUITE.to_string()
    } else {
        format!("{SUITE} ({BUILD_ID})")
    }
}

/// Plugin descriptor: OFX version props + version appended to the description.
pub fn describe_props(props: OfxPropertySetHandle, description: &str) {
    prop_set_int(props, kOfxPropVersion, 0, MAJOR as c_int);
    prop_set_int(props, kOfxPropVersion, 1, MINOR as c_int);
    prop_set_int(props, kOfxPropVersion, 2, PATCH as c_int);
    let l = format!("{}\0", label_text());
    prop_set_string(props, kOfxPropVersionLabel, 0, l.as_bytes());
    let d = format!("{description} cwTools {}.\0", label_text());
    prop_set_string(props, kOfxPropPluginDescription, 0, d.as_bytes());
}

/// Read-only "Version" param in `group` (b"\0" = top level). Not persistent: a
/// setup saved with an older build shows the build that is loaded now.
pub fn define_param(ps: OfxParamSetHandle, group: &[u8]) {
    let p = define(ps, kOfxParamTypeString, P_VERSION);
    label(p, b"Version\0");
    hint(p, b"cwTools suite version and build of the loaded plugin (read-only).\0");
    let d = format!("{}\0", label_text());
    prop_set_string(p, kOfxParamPropDefault, 0, d.as_bytes());
    prop_set_string(p, kOfxParamPropStringMode, 0, kOfxParamStringIsLabel);
    prop_set_int(p, kOfxParamPropAnimates, 0, 0);
    prop_set_int(p, kOfxParamPropPersistant, 0, 0);
    prop_set_int(p, kOfxParamPropEnabled, 0, 0);
    prop_set_int(p, kOfxParamPropEvaluateOnChange, 0, 0);
    if group.len() > 1 {
        crate::params::parent(p, group);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn version_parts_match_suite() {
        assert_eq!(format!("{MAJOR}.{MINOR}.{PATCH}"), SUITE);
        assert!(!BUILD_ID.is_empty());
        assert!(label_text().starts_with(SUITE));
    }
}
