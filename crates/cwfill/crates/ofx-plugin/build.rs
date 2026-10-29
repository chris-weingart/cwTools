// Compiles the C shim for OFX's *variadic* parameter get/set functions.
// Rust cannot call variadic C function pointers, so the shim wraps each
// concrete arity/type we use behind a fixed-signature function.
fn main() {
    println!("cargo:rerun-if-changed=src/variadic_helpers.c");
    cc::Build::new()
        .file("src/variadic_helpers.c")
        .warnings(true)
        .compile("ofx_variadic_helpers");
}
