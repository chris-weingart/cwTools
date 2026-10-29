//! Overlay drawing for OFX interacts: legacy OpenGL resolved at runtime with
//! dlsym(RTLD_DEFAULT) (no link-time GL dependency), plus a recording canvas
//! that writes the same primitives to a text file for headless inspection.
//!
//! Dump format (one primitive per line):
//!   `strip r g b width x0 y0 x1 y1 ...`  (GL_LINE_STRIP)
//!   `loop  r g b width x0 y0 ...`         (GL_LINE_LOOP)
//!   `lines r g b width x0 y0 x1 y1 ...`   (GL_LINES pairs)
//! Coordinates are whatever the caller passes (OFX canonical for overlays).

use std::ffi::c_void;
use std::io::Write;
use std::os::raw::c_char;
use std::sync::OnceLock;

extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
#[cfg(target_os = "macos")]
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;
#[cfg(not(target_os = "macos"))]
const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

pub const GL_LINES: u32 = 0x0001;
pub const GL_LINE_LOOP: u32 = 0x0002;
pub const GL_LINE_STRIP: u32 = 0x0003;

pub struct Gl {
    pub begin: unsafe extern "C" fn(u32),
    pub end: unsafe extern "C" fn(),
    pub vertex2d: unsafe extern "C" fn(f64, f64),
    pub color3f: unsafe extern "C" fn(f32, f32, f32),
    pub line_width: unsafe extern "C" fn(f32),
    pub get_error: Option<unsafe extern "C" fn() -> u32>,
}
unsafe impl Send for Gl {}
unsafe impl Sync for Gl {}

/// The process's legacy GL entry points, or None (logged once by the caller).
pub fn gl() -> Option<&'static Gl> {
    static GL: OnceLock<Option<Gl>> = OnceLock::new();
    GL.get_or_init(|| unsafe {
        let s = |n: &[u8]| dlsym(RTLD_DEFAULT, n.as_ptr() as *const c_char);
        let (b, e, v, c, l, ge) =
            (s(b"glBegin\0"), s(b"glEnd\0"), s(b"glVertex2d\0"), s(b"glColor3f\0"), s(b"glLineWidth\0"), s(b"glGetError\0"));
        if b.is_null() || e.is_null() || v.is_null() || c.is_null() || l.is_null() {
            return None;
        }
        Some(Gl {
            begin: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(u32)>(b),
            end: std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(e),
            vertex2d: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(f64, f64)>(v),
            color3f: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(f32, f32, f32)>(c),
            line_width: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(f32)>(l),
            get_error: if ge.is_null() { None } else { Some(std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> u32>(ge)) },
        })
    })
    .as_ref()
}

/// Something overlay geometry can be drawn onto.
pub trait Canvas {
    fn color(&mut self, r: f32, g: f32, b: f32);
    fn width(&mut self, w: f32);
    /// `mode` = GL_LINES / GL_LINE_LOOP / GL_LINE_STRIP.
    fn prim(&mut self, mode: u32, pts: &[(f64, f64)]);
}

pub struct GlCanvas {
    gl: &'static Gl,
    pub vertices: usize,
    pub prims: usize,
}

impl GlCanvas {
    pub fn new(gl: &'static Gl) -> Self {
        GlCanvas { gl, vertices: 0, prims: 0 }
    }
    /// glGetError after drawing (0 = GL_NO_ERROR; None = symbol missing).
    pub fn error(&self) -> Option<u32> {
        self.gl.get_error.map(|f| unsafe { f() })
    }
}

impl Canvas for GlCanvas {
    fn color(&mut self, r: f32, g: f32, b: f32) {
        unsafe { (self.gl.color3f)(r, g, b) }
    }
    fn width(&mut self, w: f32) {
        unsafe { (self.gl.line_width)(w) }
    }
    fn prim(&mut self, mode: u32, pts: &[(f64, f64)]) {
        if pts.is_empty() {
            return;
        }
        unsafe {
            (self.gl.begin)(mode);
            for &(x, y) in pts {
                (self.gl.vertex2d)(x, y);
            }
            (self.gl.end)();
        }
        self.vertices += pts.len();
        self.prims += 1;
    }
}

/// Records primitives as text (see module doc).
#[derive(Default)]
pub struct RecordCanvas {
    pub text: String,
    rgb: (f32, f32, f32),
    w: f32,
    pub vertices: usize,
    pub prims: usize,
}

impl RecordCanvas {
    pub fn new() -> Self {
        RecordCanvas { w: 1.0, ..Default::default() }
    }
    /// Write the recording to `path` (overwrites).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::File::create(path)?.write_all(self.text.as_bytes())
    }
}

impl Canvas for RecordCanvas {
    fn color(&mut self, r: f32, g: f32, b: f32) {
        self.rgb = (r, g, b);
    }
    fn width(&mut self, w: f32) {
        self.w = w;
    }
    fn prim(&mut self, mode: u32, pts: &[(f64, f64)]) {
        if pts.is_empty() {
            return;
        }
        let tag = match mode {
            GL_LINES => "lines",
            GL_LINE_LOOP => "loop",
            _ => "strip",
        };
        let (r, g, b) = self.rgb;
        self.text.push_str(&format!("{tag} {r:.3} {g:.3} {b:.3} {:.2}", self.w));
        for &(x, y) in pts {
            self.text.push_str(&format!(" {x:.2} {y:.2}"));
        }
        self.text.push('\n');
        self.vertices += pts.len();
        self.prims += 1;
    }
}
