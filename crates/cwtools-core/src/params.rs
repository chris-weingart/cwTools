//! Parameter definition + readback.
//!
//! Flame constraints (flame_ofx_host_facts): no Pages (use Groups — the group
//! label becomes a column header), no parametric params, String params never
//! animate, Boolean/Choice animate only since 2025.1, OFX strings truncate at
//! 255 bytes, Double labels clip at ~15 chars.
//!
//! Two layers:
//! - byte-string helpers ([`define`], [`boolp`], [`choice`], ...) taking
//!   NUL-terminated `b"..\0"` literals (cwSegment style);
//! - a table DSL ([`ParamDef`], [`Kind`], [`define_all`], [`validate`]) whose
//!   definitions are plain `&str` and can be checked by unit tests.

use crate::ofx_api::*;
use crate::props::{prop_set_double, prop_set_int, prop_set_string};
use crate::suites::param;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};

/// Flame truncates OFX string properties at this many bytes.
pub const MAX_STRING_BYTES: usize = 255;
/// Double-param labels clip at roughly this many characters in Flame's panel.
pub const MAX_DOUBLE_LABEL: usize = 15;

// ---------------------------------------------------------------------------
// Byte-string helpers
// ---------------------------------------------------------------------------
pub fn define(ps: OfxParamSetHandle, ty: &[u8], name: &[u8]) -> OfxPropertySetHandle {
    let mut props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        (param().paramDefine)(ps, ty.as_ptr() as *const c_char, name.as_ptr() as *const c_char, &mut props);
    }
    props
}
pub fn label(p: OfxPropertySetHandle, t: &[u8]) {
    prop_set_string(p, kOfxPropLabel, 0, t);
}
pub fn hint(p: OfxPropertySetHandle, t: &[u8]) {
    prop_set_string(p, kOfxParamPropHint, 0, t);
}
pub fn parent(p: OfxPropertySetHandle, g: &[u8]) {
    prop_set_string(p, kOfxParamPropParent, 0, g);
}
pub fn boolp(ps: OfxParamSetHandle, name: &[u8], lbl: &[u8], hnt: &[u8], def: bool, g: &[u8]) {
    let p = define(ps, kOfxParamTypeBoolean, name);
    label(p, lbl);
    hint(p, hnt);
    prop_set_int(p, kOfxParamPropDefault, 0, def as c_int);
    parent(p, g);
}
pub fn choice(ps: OfxParamSetHandle, name: &[u8], lbl: &[u8], hnt: &[u8], opts: &[&[u8]], def: c_int, g: &[u8]) {
    let p = define(ps, kOfxParamTypeChoice, name);
    label(p, lbl);
    hint(p, hnt);
    for (i, o) in opts.iter().enumerate() {
        prop_set_string(p, kOfxParamPropChoiceOption, i as c_int, o);
    }
    prop_set_int(p, kOfxParamPropDefault, 0, def);
    parent(p, g);
}
pub fn string(ps: OfxParamSetHandle, name: &[u8], lbl: &[u8], hnt: &[u8], def: &str, g: &[u8]) {
    let p = define(ps, kOfxParamTypeString, name);
    label(p, lbl);
    hint(p, hnt);
    let d = format!("{def}\0");
    prop_set_string(p, kOfxParamPropDefault, 0, d.as_bytes());
    prop_set_string(p, kOfxParamPropStringMode, 0, kOfxParamStringIsSingleLine);
    prop_set_int(p, kOfxParamPropAnimates, 0, 0);
    parent(p, g);
}
pub fn button(ps: OfxParamSetHandle, name: &[u8], lbl: &[u8], hnt: &[u8], g: &[u8]) {
    let p = define(ps, kOfxParamTypePushButton, name);
    label(p, lbl);
    hint(p, hnt);
    parent(p, g);
}
pub fn xy(ps: OfxParamSetHandle, name: &[u8], lbl: &[u8], hnt: &[u8], dx: f64, dy: f64, g: &[u8]) {
    let p = define(ps, kOfxParamTypeDouble2D, name);
    label(p, lbl);
    hint(p, hnt);
    prop_set_string(p, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeXYAbsolute);
    prop_set_string(p, kOfxParamPropDefaultCoordinateSystem, 0, kOfxParamCoordinatesNormalised);
    prop_set_double(p, kOfxParamPropDefault, 0, dx);
    prop_set_double(p, kOfxParamPropDefault, 1, dy);
    parent(p, g);
}
pub fn group(ps: OfxParamSetHandle, name: &[u8], lbl: &[u8]) {
    let g = define(ps, kOfxParamTypeGroup, name);
    label(g, lbl);
}

// ---------------------------------------------------------------------------
// Table DSL
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    /// A group (column header in Flame). `group` of a Group is ignored.
    Group,
    Bool { def: bool },
    Choice { opts: &'static [&'static str], def: i32 },
    /// Animatable double.
    Double { def: f64, min: f64, max: f64, dmin: f64, dmax: f64, digits: i32 },
    Int { def: i32, min: i32, max: i32 },
    /// Single-line string (never animates in Flame).
    Str { def: &'static str },
    Button,
    /// Double2D position (XY absolute; default in normalised 0..1 frame coords).
    Xy { dx: f64, dy: f64 },
    /// One-row read-only text = `label` (Flame shows no label for String params,
    /// so a String gets a Caption row above it).
    Caption,
    /// One blank row (an empty label-mode String: a dark empty field in Flame).
    Spacer,
    /// The read-only Version row ([`crate::version::define_param`]) at this
    /// place in the table (`name` / `label` are ignored).
    Version,
}

#[derive(Clone, Copy, Debug)]
pub struct ParamDef {
    pub name: &'static str,
    pub label: &'static str,
    pub hint: &'static str,
    /// Parent group's `name` ("" = top level).
    pub group: &'static str,
    pub kind: Kind,
}

fn cs(s: &str) -> Vec<u8> {
    CString::new(s.replace('\0', " ")).unwrap_or_default().into_bytes_with_nul()
}

/// Define every parameter of `defs` in order.
pub fn define_all(ps: OfxParamSetHandle, defs: &[ParamDef]) {
    define_all_with(ps, defs, &|_| None);
}

/// [`define_all`], with String defaults known only at load time: `shown(name)`
/// replaces the table default (e.g. the Model Folder / Server URL / ONNX
/// Runtime in effect from cwtools.conf, so a new node shows them instead of an
/// empty field).
pub fn define_all_with(ps: OfxParamSetHandle, defs: &[ParamDef], shown: &dyn Fn(&str) -> Option<String>) {
    for d in defs {
        let (name, lbl, hnt, grp) = (cs(d.name), cs(d.label), cs(d.hint), cs(d.group));
        if matches!(d.kind, Kind::Version) {
            crate::version::define_param(ps, &grp);
            continue;
        }
        let ty: &[u8] = match d.kind {
            Kind::Group => kOfxParamTypeGroup,
            Kind::Bool { .. } => kOfxParamTypeBoolean,
            Kind::Choice { .. } => kOfxParamTypeChoice,
            Kind::Double { .. } => kOfxParamTypeDouble,
            Kind::Int { .. } => kOfxParamTypeInteger,
            Kind::Str { .. } => kOfxParamTypeString,
            Kind::Button => kOfxParamTypePushButton,
            Kind::Xy { .. } => kOfxParamTypeDouble2D,
            Kind::Caption | Kind::Spacer => kOfxParamTypeString,
            Kind::Version => unreachable!(),
        };
        let p = define(ps, ty, &name);
        label(p, &lbl);
        if !d.hint.is_empty() {
            hint(p, &hnt);
        }
        match d.kind {
            Kind::Group => {}
            Kind::Bool { def } => prop_set_int(p, kOfxParamPropDefault, 0, def as c_int),
            Kind::Choice { opts, def } => {
                for (i, o) in opts.iter().enumerate() {
                    prop_set_string(p, kOfxParamPropChoiceOption, i as c_int, &cs(o));
                }
                prop_set_int(p, kOfxParamPropDefault, 0, def);
            }
            Kind::Double { def, min, max, dmin, dmax, digits } => {
                prop_set_double(p, kOfxParamPropDefault, 0, def);
                prop_set_double(p, kOfxParamPropMin, 0, min);
                prop_set_double(p, kOfxParamPropMax, 0, max);
                prop_set_double(p, kOfxParamPropDisplayMin, 0, dmin);
                prop_set_double(p, kOfxParamPropDisplayMax, 0, dmax);
                prop_set_int(p, kOfxParamPropDigits, 0, digits);
                // Without an increment Flame steps (and in effect rounds) by 1: a 0..1 field
                // could only be 0 or 1 (Mid Bias, cwMatte Clip/Gamma).
                prop_set_double(p, kOfxParamPropIncrement, 0, increment_for(digits));
                prop_set_int(p, kOfxParamPropAnimates, 0, 1);
            }
            Kind::Int { def, min, max } => {
                prop_set_int(p, kOfxParamPropDefault, 0, def);
                prop_set_int(p, kOfxParamPropMin, 0, min);
                prop_set_int(p, kOfxParamPropMax, 0, max);
                prop_set_int(p, kOfxParamPropDisplayMin, 0, min);
                prop_set_int(p, kOfxParamPropDisplayMax, 0, max);
            }
            Kind::Str { def } => {
                let def = shown(d.name).unwrap_or_else(|| def.to_string());
                prop_set_string(p, kOfxParamPropDefault, 0, &cs(&def));
                prop_set_string(p, kOfxParamPropStringMode, 0, kOfxParamStringIsSingleLine);
                prop_set_int(p, kOfxParamPropAnimates, 0, 0);
            }
            Kind::Button | Kind::Version => {}
            Kind::Caption | Kind::Spacer => {
                let text = if matches!(d.kind, Kind::Caption) { cs(d.label) } else { cs("") };
                prop_set_string(p, kOfxParamPropDefault, 0, &text);
                prop_set_string(p, kOfxParamPropStringMode, 0, kOfxParamStringIsLabel);
                prop_set_int(p, kOfxParamPropAnimates, 0, 0);
                prop_set_int(p, kOfxParamPropPersistant, 0, 0);
                prop_set_int(p, kOfxParamPropEvaluateOnChange, 0, 0);
            }
            Kind::Xy { dx, dy } => {
                prop_set_string(p, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeXYAbsolute);
                prop_set_string(p, kOfxParamPropDefaultCoordinateSystem, 0, kOfxParamCoordinatesNormalised);
                prop_set_double(p, kOfxParamPropDefault, 0, dx);
                prop_set_double(p, kOfxParamPropDefault, 1, dy);
            }
        }
        if !matches!(d.kind, Kind::Group) && !d.group.is_empty() {
            parent(p, &grp);
        }
    }
}

/// Check a definition table against Flame's limits. Used by unit tests.
pub fn validate(defs: &[ParamDef]) -> Result<(), String> {
    let mut names = std::collections::HashSet::new();
    let groups: std::collections::HashSet<&str> =
        defs.iter().filter(|d| matches!(d.kind, Kind::Group)).map(|d| d.name).collect();
    for (i, d) in defs.iter().enumerate() {
        if d.name.is_empty() || !d.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("bad param name {:?}", d.name));
        }
        if !names.insert(d.name) {
            return Err(format!("duplicate param name {}", d.name));
        }
        if d.label.is_empty() {
            return Err(format!("{}: empty label", d.name));
        }
        for (what, s) in [("label", d.label), ("hint", d.hint), ("default", match d.kind {
            Kind::Str { def } => def,
            _ => "",
        })] {
            if s.len() > MAX_STRING_BYTES {
                return Err(format!("{}: {what} is {} bytes (> {MAX_STRING_BYTES})", d.name, s.len()));
            }
        }
        match d.kind {
            Kind::Group => {}
            _ => {
                if !d.group.is_empty() {
                    if !groups.contains(d.group) {
                        return Err(format!("{}: unknown group {}", d.name, d.group));
                    }
                    // Group must be defined before its children.
                    let gi = defs.iter().position(|g| g.name == d.group).unwrap_or(usize::MAX);
                    if gi > i {
                        return Err(format!("{}: group {} defined after it", d.name, d.group));
                    }
                }
            }
        }
        match d.kind {
            Kind::Double { def, min, max, dmin, dmax, .. } => {
                if d.label.chars().count() > MAX_DOUBLE_LABEL {
                    return Err(format!("{}: double label {:?} > {MAX_DOUBLE_LABEL} chars", d.name, d.label));
                }
                if !(min <= def && def <= max && min <= dmin && dmin <= dmax && dmax <= max) {
                    return Err(format!("{}: inconsistent range", d.name));
                }
            }
            Kind::Choice { opts, def } => {
                if opts.is_empty() || def < 0 || def as usize >= opts.len() {
                    return Err(format!("{}: bad choice default", d.name));
                }
                for o in opts {
                    if o.len() > MAX_STRING_BYTES {
                        return Err(format!("{}: option too long", d.name));
                    }
                }
            }
            Kind::Int { def, min, max } => {
                if !(min <= def && def <= max) {
                    return Err(format!("{}: inconsistent int range", d.name));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Guard for hand-written byte-string params: every `b"..."` literal in `src`
/// must fit Flame's 255-byte limit (line continuations collapse).
pub fn check_byte_literals(src: &str) -> Result<(), String> {
    let mut i = 0;
    let b = src.as_bytes();
    while let Some(off) = src[i..].find("b\"") {
        let start = i + off + 2;
        let mut j = start;
        let mut len = 0;
        while j < b.len() && b[j] != b'"' {
            if b[j] == b'\\' && j + 1 < b.len() && b[j + 1] == b'\n' {
                // line continuation: skip newline + leading whitespace
                j += 2;
                while j < b.len() && (b[j] == b' ' || b[j] == b'\n') {
                    j += 1;
                }
                continue;
            }
            len += 1;
            j += 1;
        }
        if len > MAX_STRING_BYTES {
            return Err(format!("literal over {MAX_STRING_BYTES} bytes at offset {start}: {len}"));
        }
        i = j + 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Readback (names are NUL-terminated byte strings)
// ---------------------------------------------------------------------------
pub fn handle(ps: OfxParamSetHandle, name: &[u8]) -> OfxParamHandle {
    let mut h: OfxParamHandle = std::ptr::null_mut();
    unsafe {
        (param().paramGetHandle)(ps, name.as_ptr() as *const c_char, &mut h, std::ptr::null_mut());
    }
    h
}
pub fn get_int(ps: OfxParamSetHandle, name: &[u8], t: f64) -> c_int {
    let h = handle(ps, name);
    let mut v: c_int = 0;
    if !h.is_null() {
        unsafe { cw_param_get_int_at(param().paramGetValueAtTime, h, t, &mut v) };
    }
    v
}
pub fn get_bool(ps: OfxParamSetHandle, name: &[u8], t: f64) -> bool {
    get_int(ps, name, t) != 0
}
pub fn get_double(ps: OfxParamSetHandle, name: &[u8], t: f64) -> f64 {
    let h = handle(ps, name);
    let mut v = 0.0;
    if !h.is_null() {
        unsafe { cw_param_get_double_at(param().paramGetValueAtTime, h, t, &mut v) };
    }
    v
}
pub fn get_xy(ps: OfxParamSetHandle, name: &[u8], t: f64) -> (f64, f64) {
    let h = handle(ps, name);
    let (mut x, mut y) = (0.0, 0.0);
    if !h.is_null() {
        unsafe { cw_param_get_double2_at(param().paramGetValueAtTime, h, t, &mut x, &mut y) };
    }
    (x, y)
}
pub fn get_string(ps: OfxParamSetHandle, name: &[u8]) -> String {
    let h = handle(ps, name);
    let mut p: *mut c_char = std::ptr::null_mut();
    if !h.is_null() {
        unsafe { cw_param_get_string(param().paramGetValue, h, &mut p) };
    }
    if p.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().trim().to_string() }
}
pub fn set_bool(ps: OfxParamSetHandle, name: &[u8], v: bool) {
    let h = handle(ps, name);
    if !h.is_null() {
        unsafe { cw_param_set_int(param().paramSetValue, h, v as c_int) };
    }
}
pub fn set_int(ps: OfxParamSetHandle, name: &[u8], v: c_int) {
    let h = handle(ps, name);
    if !h.is_null() {
        unsafe { cw_param_set_int(param().paramSetValue, h, v) };
    }
}
pub fn set_string(ps: OfxParamSetHandle, name: &[u8], v: &str) {
    let h = handle(ps, name);
    if !h.is_null() {
        let c = CString::new(v.replace('\0', " ")).unwrap_or_default();
        unsafe { cw_param_set_string(param().paramSetValue, h, c.as_ptr()) };
    }
}
/// At instance creation: fill each EMPTY String param with the value it would
/// resolve to (cwtools.conf / auto-detect), so nodes from older setups show the
/// real server / folder / library too. A value already set is left alone.
pub fn fill_empty_strings(ps: OfxParamSetHandle, fields: &[(&[u8], String)]) {
    for (name, value) in fields {
        if !value.is_empty() && get_string(ps, name).trim().is_empty() {
            set_string(ps, name, value);
        }
    }
}
/// Multi-line, non-animating string param (display / notes).
pub fn string_multiline(ps: OfxParamSetHandle, name: &[u8], lbl: &[u8], hnt: &[u8], def: &str, g: &[u8]) {
    let p = define(ps, kOfxParamTypeString, name);
    label(p, lbl);
    hint(p, hnt);
    let d = format!("{def}\0");
    prop_set_string(p, kOfxParamPropDefault, 0, d.as_bytes());
    prop_set_string(p, kOfxParamPropStringMode, 0, kOfxParamStringIsMultiLine);
    prop_set_int(p, kOfxParamPropAnimates, 0, 0);
    parent(p, g);
}
/// Integer param with range (display range = range).
pub fn int(ps: OfxParamSetHandle, name: &[u8], lbl: &[u8], hnt: &[u8], def: c_int, min: c_int, max: c_int, g: &[u8]) {
    let p = define(ps, kOfxParamTypeInteger, name);
    label(p, lbl);
    hint(p, hnt);
    prop_set_int(p, kOfxParamPropDefault, 0, def);
    prop_set_int(p, kOfxParamPropMin, 0, min);
    prop_set_int(p, kOfxParamPropMax, 0, max);
    prop_set_int(p, kOfxParamPropDisplayMin, 0, min);
    prop_set_int(p, kOfxParamPropDisplayMax, 0, max);
    parent(p, g);
}
pub fn set_xy(ps: OfxParamSetHandle, name: &[u8], x: f64, y: f64) {
    let h = handle(ps, name);
    if !h.is_null() {
        unsafe { cw_param_set_double2(param().paramSetValue, h, x, y) };
    }
}

/// `&str` param name → NUL-terminated bytes for the readback functions.
pub fn n(name: &str) -> Vec<u8> {
    cs(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    const OK: &[ParamDef] = &[
        ParamDef { name: "grp", label: "Group", hint: "", group: "", kind: Kind::Group },
        ParamDef {
            name: "a",
            label: "Fifteen chars..",
            hint: "h",
            group: "grp",
            kind: Kind::Double { def: 1.0, min: 0.0, max: 2.0, dmin: 0.0, dmax: 2.0, digits: 2 },
        },
        ParamDef { name: "c", label: "C", hint: "", group: "grp", kind: Kind::Choice { opts: &["x", "y"], def: 1 } },
    ];
    #[test]
    fn validate_accepts_good_table() {
        validate(OK).unwrap();
    }
    #[test]
    fn validate_rejects_long_double_label() {
        let mut v = OK.to_vec();
        v[1].label = "Sixteen chars..."; // 16
        assert!(validate(&v).is_err());
    }
    #[test]
    fn validate_rejects_long_hint_and_bad_group() {
        let long: &'static str = Box::leak("x".repeat(256).into_boxed_str());
        let mut v = OK.to_vec();
        v[2].hint = long;
        assert!(validate(&v).is_err());
        let mut v = OK.to_vec();
        v[2].group = "nope";
        assert!(validate(&v).is_err());
        let mut v = OK.to_vec();
        v[2].kind = Kind::Choice { opts: &["x"], def: 1 };
        assert!(validate(&v).is_err());
    }
    #[test]
    fn byte_literal_check() {
        assert!(check_byte_literals("b\"short\\0\"").is_ok());
        let long = format!("b\"{}\"", "y".repeat(300));
        assert!(check_byte_literals(&long).is_err());
    }
}

/// Drag / step increment for a Double shown with `digits` decimals: 1 for whole
/// numbers, 0.1 for one decimal, 0.01 for two or more.
pub fn increment_for(digits: i32) -> f64 {
    10f64.powi(-digits.clamp(0, 2))
}

/// Hide a described param (OfxParamPropSecret): Flame gives it no row; the
/// plugin can still read and set it.
pub fn set_secret(ps: OfxParamSetHandle, name: &[u8]) {
    let mut h: OfxParamHandle = std::ptr::null_mut();
    let mut props: OfxPropertySetHandle = std::ptr::null_mut();
    unsafe {
        (param().paramGetHandle)(ps, name.as_ptr() as *const c_char, &mut h, &mut props);
    }
    if !props.is_null() {
        crate::props::prop_set_int(props, b"OfxParamPropSecret\0", 0, 1);
    }
}

// ---------------------------------------------------------------------------
// Flame panel grid (cwProbe layout probes, Flame 2027.2)
// ---------------------------------------------------------------------------
/// Rows per column in Flame's OFX panel (fixed; same on Linux and retina Mac).
pub const GRID_ROWS: usize = 5;
/// Columns per page before Flame starts the next page tab.
pub const GRID_COLS: usize = 6;

/// Rows one param takes: a Choice is a label row + a value row (never split
/// across columns), so is a 2D point (cwMatte ROI corners, seen in Flame
/// 2026-10-21); everything else is one row; a String shows no label.
pub fn grid_rows(k: &Kind) -> usize {
    match k {
        Kind::Group => 0,
        Kind::Choice { .. } | Kind::Xy { .. } => 2,
        _ => 1,
    }
}

/// One simulated column: its header (the group label) and row labels.
#[derive(Debug, Clone, PartialEq)]
pub struct GridColumn {
    pub header: String,
    pub rows: Vec<String>,
}

/// Lay `defs` out the way Flame does: every group starts a new column, a column
/// holds [`GRID_ROWS`] rows, a param that does not fit continues in a new column
/// under the same header. Params are placed in declaration order. Top-level
/// params go under "Controls".
pub fn grid(defs: &[ParamDef]) -> Vec<GridColumn> {
    let mut cols: Vec<GridColumn> = Vec::new();
    let mut used = GRID_ROWS;
    let mut cur_group = "\u{0}";
    for d in defs {
        if matches!(d.kind, Kind::Group) {
            continue;
        }
        let header = if d.group.is_empty() {
            "Controls".to_string()
        } else {
            defs.iter().find(|g| g.name == d.group && matches!(g.kind, Kind::Group)).map(|g| g.label.to_string()).unwrap_or_default()
        };
        let n = grid_rows(&d.kind);
        if d.group != cur_group || used + n > GRID_ROWS {
            cols.push(GridColumn { header, rows: Vec::new() });
            used = 0;
            cur_group = d.group;
        }
        let c = cols.last_mut().unwrap();
        c.rows.push(if matches!(d.kind, Kind::Spacer) { String::new() } else { d.label.to_string() });
        used += n;
    }
    cols
}

/// [`grid`] split into pages of [`GRID_COLS`] columns.
pub fn grid_pages(defs: &[ParamDef]) -> Vec<Vec<GridColumn>> {
    grid(defs).chunks(GRID_COLS).map(|c| c.to_vec()).collect()
}

/// Text dump of the simulated panel (for test failure messages / docs).
pub fn grid_text(defs: &[ParamDef]) -> String {
    let mut s = String::new();
    for (i, page) in grid_pages(defs).iter().enumerate() {
        s += &format!("Page {}\n", i + 1);
        for c in page {
            s += &format!("  [{}] {}\n", c.header, c.rows.join(" / "));
        }
    }
    s
}

#[cfg(test)]
mod grid_tests {
    use super::*;
    const G: &str = "g";
    fn d(name: &'static str, group: &'static str, kind: Kind) -> ParamDef {
        ParamDef { name, label: name, hint: "", group, kind }
    }
    #[test]
    fn matches_the_flame_probe() {
        // cwProbe Layout None: Main = Model / Device / Release (one full column),
        // A = 6 doubles (5 + 1), B = 10 bools, C = 3 choices (2 + 2 | 2).
        let ch = Kind::Choice { opts: &["a", "b"], def: 0 };
        let db = Kind::Double { def: 0.0, min: 0.0, max: 1.0, dmin: 0.0, dmax: 1.0, digits: 2 };
        let mut v = vec![
            ParamDef { name: G, label: "Main", hint: "", group: "", kind: Kind::Group },
            d("Model", G, ch), d("Device", G, ch), d("Release", G, Kind::Button),
            ParamDef { name: "a", label: "A", hint: "", group: "", kind: Kind::Group },
        ];
        for n in ["A1", "A2", "A3", "A4", "A5", "A6"] { v.push(d(n, "a", db)); }
        v.push(ParamDef { name: "c", label: "C", hint: "", group: "", kind: Kind::Group });
        for n in ["C1", "C2", "C3"] { v.push(d(n, "c", ch)); }
        let g = grid(&v);
        let shape: Vec<(String, usize)> = g.iter().map(|c| (c.header.clone(), c.rows.len())).collect();
        assert_eq!(shape, vec![("Main".into(), 3), ("A".into(), 5), ("A".into(), 1), ("C".into(), 2), ("C".into(), 1)]);
    }
}

/// One-row read-only text (a String's visible name; Flame shows no String
/// labels). Same as [`Kind::Caption`] for hand-written param code.
pub fn caption(ps: OfxParamSetHandle, name: &[u8], text: &[u8], g: &[u8]) {
    let p = define(ps, kOfxParamTypeString, name);
    label(p, text);
    prop_set_string(p, kOfxParamPropDefault, 0, text);
    prop_set_string(p, kOfxParamPropStringMode, 0, kOfxParamStringIsLabel);
    prop_set_int(p, kOfxParamPropAnimates, 0, 0);
    prop_set_int(p, kOfxParamPropPersistant, 0, 0);
    prop_set_int(p, kOfxParamPropEvaluateOnChange, 0, 0);
    parent(p, g);
}
