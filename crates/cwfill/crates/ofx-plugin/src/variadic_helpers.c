/* C shim for OFX's variadic parameter functions.
 *
 * OfxParameterSuiteV1::paramGetValue / paramSetValue are declared variadic
 * in the OFX C API:
 *     OfxStatus (*paramGetValue)(OfxParamHandle paramHandle, ...);
 *     OfxStatus (*paramSetValue)(OfxParamHandle paramHandle, ...);
 * The same function pointer serves every parameter type; the trailing args
 * differ by type (double*, int*, char**, or several of them for N-D params).
 *
 * Rust has no stable way to call a variadic C function pointer, so for each
 * concrete shape we use we expose a fixed-signature wrapper. Rust passes the
 * suite's variadic fn pointer (as void*) plus the param handle; we cast and
 * call with the right trailing argument.
 */

typedef int OfxStatus;

/* paramGetValue(handle, T* out) */
typedef OfxStatus (*GetFn)(void *handle, ...);
/* paramSetValue(handle, T in)   */
typedef OfxStatus (*SetFn)(void *handle, ...);

OfxStatus cw_param_get_double(void *fn, void *handle, double *out) {
    return ((GetFn)fn)(handle, out);
}

OfxStatus cw_param_get_int(void *fn, void *handle, int *out) {
    return ((GetFn)fn)(handle, out);
}

/* paramGetValue(handle, double* r, double* g, double* b) for an RGB param */
OfxStatus cw_param_get_rgb(void *fn, void *handle, double *r, double *g, double *b) {
    return ((GetFn)fn)(handle, r, g, b);
}

OfxStatus cw_param_get_string(void *fn, void *handle, char **out) {
    return ((GetFn)fn)(handle, out);
}

OfxStatus cw_param_set_double(void *fn, void *handle, double v) {
    return ((SetFn)fn)(handle, v);
}

OfxStatus cw_param_set_int(void *fn, void *handle, int v) {
    return ((SetFn)fn)(handle, v);
}

/* Test-only stand-in for a host's variadic paramGetValue on an RGB param:
 * reads three double* from the va_list exactly as an OFX host does. Lets
 * `cargo test` prove the cw_param_get_rgb shim's calling convention on each
 * arch (arm64 macOS passes variadics on the stack, x86-64 in registers). */
#include <stdarg.h>
OfxStatus cw_selftest_variadic_rgb(void *handle, ...) {
    (void)handle;
    va_list ap;
    va_start(ap, handle);
    double *r = va_arg(ap, double *);
    double *g = va_arg(ap, double *);
    double *b = va_arg(ap, double *);
    va_end(ap);
    *r = 0.249145508; *g = 0.537597656; *b = 0.720214844;
    return 0;
}
