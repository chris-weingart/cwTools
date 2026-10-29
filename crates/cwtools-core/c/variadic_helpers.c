/* C shim for OFX's variadic functions (paramGetValue/SetValue[AtTime],
 * message suite). Rust cannot call variadic C fn pointers, so each concrete
 * shape gets a fixed-signature wrapper. Mirrors cwFill. */
typedef int OfxStatus;
typedef OfxStatus (*VarFn)(void *handle, ...);
typedef OfxStatus (*VarTFn)(void *handle, double t, ...);
typedef OfxStatus (*MsgFn)(void *handle, const char *ty, const char *id, const char *fmt, ...);

OfxStatus cw_param_get_double(void *fn, void *h, double *out) { return ((VarFn)fn)(h, out); }
OfxStatus cw_param_get_int(void *fn, void *h, int *out) { return ((VarFn)fn)(h, out); }
OfxStatus cw_param_get_rgb(void *fn, void *h, double *r, double *g, double *b) { return ((VarFn)fn)(h, r, g, b); }
OfxStatus cw_param_get_string(void *fn, void *h, char **out) { return ((VarFn)fn)(h, out); }
OfxStatus cw_param_set_double(void *fn, void *h, double v) { return ((VarFn)fn)(h, v); }
OfxStatus cw_param_set_int(void *fn, void *h, int v) { return ((VarFn)fn)(h, v); }
OfxStatus cw_param_set_string(void *fn, void *h, const char *v) { return ((VarFn)fn)(h, v); }

OfxStatus cw_param_get_double_at(void *fn, void *h, double t, double *out) { return ((VarTFn)fn)(h, t, out); }
OfxStatus cw_param_get_int_at(void *fn, void *h, double t, int *out) { return ((VarTFn)fn)(h, t, out); }
OfxStatus cw_param_get_double2_at(void *fn, void *h, double t, double *x, double *y) { return ((VarTFn)fn)(h, t, x, y); }
OfxStatus cw_param_set_double2(void *fn, void *h, double x, double y) { return ((VarFn)fn)(h, x, y); }
OfxStatus cw_param_set_double2_at(void *fn, void *h, double t, double x, double y) { return ((VarTFn)fn)(h, t, x, y); }
OfxStatus cw_param_set_double_at(void *fn, void *h, double t, double v) { return ((VarTFn)fn)(h, t, v); }
OfxStatus cw_param_set_int_at(void *fn, void *h, double t, int v) { return ((VarTFn)fn)(h, t, v); }

OfxStatus cw_message(void *fn, void *h, const char *ty, const char *id, const char *text) {
    return ((MsgFn)fn)(h, ty, id, "%s", text);
}
