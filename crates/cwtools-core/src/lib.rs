//! cwtools-core — shared plumbing for the cwTools OpenFX plugins (cwSegment,
//! cwRefine, ...) for Autodesk Flame.
//!
//! Each plugin is its own `.ofx` cdylib that links this crate statically, so
//! every static in here (suites, log name, model cache) exists once *per
//! plugin*, shared by all of that plugin's node instances.
//!
//! - [`ofx_api`]  hand-written OFX C ABI (structs, constants, C variadic shim)
//! - [`suites`]   host suite pointers + the generic `OfxActionLoad`
//! - [`props`]    property-suite helpers
//! - [`params`]   param definition helpers, a table DSL, readback, guard checks
//! - [`message`]  node (persistent) messages
//! - [`log`]      file logger (`/tmp/<Plugin>.log`, `<PLUGIN>_VERBOSE`)
//! - [`image`]    clip/image fetch, pixel access, release
//! - [`color`]    input colour space → display-referred conversion
//! - [`pixels`]   top-down image buffers, resampling, hashing
//! - [`resample`] Pillow-exact Image.resize (BILINEAR / LANCZOS; u8 + float) + row-parallel helper
//! - [`config`]   install-wide settings (`cwtools.conf`: model folders, servers, onnxruntime)
//! - [`ort_find`] locating onnxruntime 1.22 (loaded copy / running Flame / newest lib64)
//! - [`ort_rt`]   onnxruntime init (Flame's bundled 1.22, auto-detected) + process-wide session cache
//! - [`http`]     shared keep-alive HTTP agent, timeouts, error formatting
//! - [`guard`]    panic guard for every OFX entry point
//! - [`contour`]  marching-squares boundary loops + exact EDT (spline trimap)
//! - [`version`]  suite version + build id (OFX version props, Version param, load log)
//! - [`overlay_gl`] runtime-resolved legacy GL + recording canvas for interacts

#![allow(non_snake_case)]

pub mod color;
pub mod config;
pub mod contour;
pub mod guard;
pub mod http;
pub mod image;
pub mod log;
pub mod message;
pub mod ofx_api;
pub mod overlay_gl;
pub mod ort_find;
pub mod ort_rt;
pub mod params;
pub mod pixels;
pub mod resample;
pub mod props;
pub mod suites;
pub mod version;

pub use guard::guarded;
pub use message::set_node_message;
pub use props::*;
pub use suites::{effect, param, prop, suites, suites_mut, Suites};
