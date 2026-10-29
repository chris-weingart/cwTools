/* Minimal fake OFX host: dlopen the built .ofx, then drive
 *   OfxGetNumberOfPlugins / OfxGetPlugin / setHost
 *   -> OfxActionLoad -> OfxActionDescribe -> DescribeInContext
 *
 * Implements stub suites with the STANDARD OFX struct layouts, so it
 * cross-checks cwFill's hand-written FFI (ofx_api.rs): a field-order
 * mismatch shows up as a crash or wrong dispatch here. No Flame needed.
 *
 * build:  cc -o /tmp/cwfill_host scripts/test_host.c -ldl
 * run:    /tmp/cwfill_host dist/cwFill.ofx.bundle/Contents/MacOS/cwFill.ofx
 */
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef int OfxStatus;
typedef void *Handle;

/* ---- Property suite (fixed signatures, standard OFX order) ---- */
static OfxStatus pSetPointer(Handle p, const char *k, int i, void *v) { (void)p;(void)k;(void)i;(void)v; return 0; }
static OfxStatus pSetString(Handle p, const char *k, int i, const char *v) {
    printf("    propSetString  %-44s [%d] = %s\n", k, i, v); return 0;
}
static OfxStatus pSetDouble(Handle p, const char *k, int i, double v) { (void)p; printf("    propSetDouble  %-44s [%d] = %g\n", k, i, v); return 0; }
static OfxStatus pSetInt(Handle p, const char *k, int i, int v) { (void)p; printf("    propSetInt     %-44s [%d] = %d\n", k, i, v); return 0; }
static OfxStatus pSetPointerN(Handle p,const char*k,int i,void*const*v){(void)p;(void)k;(void)i;(void)v;return 0;}
static OfxStatus pSetStringN(Handle p,const char*k,int i,const char*const*v){(void)p;(void)k;(void)i;(void)v;return 0;}
static OfxStatus pSetDoubleN(Handle p,const char*k,int i,const double*v){(void)p;(void)k;(void)i;(void)v;return 0;}
static OfxStatus pSetIntN(Handle p,const char*k,int i,const int*v){(void)p;(void)k;(void)i;(void)v;return 0;}
static OfxStatus pGetPointer(Handle p,const char*k,int i,void**v){(void)p;(void)k;(void)i;if(v)*v=0;return 0;}
static OfxStatus pGetString(Handle p,const char*k,int i,char**v){(void)p;(void)k;(void)i;if(v)*v=0;return 0;}
static OfxStatus pGetDouble(Handle p,const char*k,int i,double*v){(void)p;(void)k;(void)i;if(v)*v=0;return 0;}
static OfxStatus pGetInt(Handle p,const char*k,int i,int*v){(void)p;(void)k;(void)i;if(v)*v=0;return 0;}
static OfxStatus pGetPointerN(Handle p,const char*k,int i,void**v){(void)p;(void)k;(void)i;(void)v;return 0;}
static OfxStatus pGetStringN(Handle p,const char*k,int i,char**v){(void)p;(void)k;(void)i;(void)v;return 0;}
static OfxStatus pGetDoubleN(Handle p,const char*k,int i,double*v){(void)p;(void)k;(void)i;(void)v;return 0;}
static OfxStatus pGetIntN(Handle p,const char*k,int i,int*v){(void)p;(void)k;(void)i;(void)v;return 0;}
static OfxStatus pReset(Handle p,const char*k){(void)p;(void)k;return 0;}
static OfxStatus pGetDimension(Handle p,const char*k,int*v){(void)p;(void)k;if(v)*v=0;return 0;}

struct PropSuite {
    void *fns[18];
} propSuite = {{
    pSetPointer,pSetString,pSetDouble,pSetInt,pSetPointerN,pSetStringN,pSetDoubleN,pSetIntN,
    pGetPointer,pGetString,pGetDouble,pGetInt,pGetPointerN,pGetStringN,pGetDoubleN,pGetIntN,
    pReset,pGetDimension
}};

/* ---- Image effect suite ---- */
static int dummyProps = 1, dummyParamSet = 2, dummyClip = 3;
static OfxStatus eGetPropertySet(Handle e,Handle*o){(void)e;if(o)*o=&dummyProps;return 0;}
static OfxStatus eGetParamSet(Handle e,Handle*o){(void)e;if(o)*o=&dummyParamSet;return 0;}
static OfxStatus eClipDefine(Handle e,const char*n,Handle*o){(void)e;printf("    clipDefine     %s\n",n);if(o)*o=&dummyProps;return 0;}
static OfxStatus eClipGetHandle(Handle e,const char*n,Handle*c,Handle*p){(void)e;(void)n;if(c)*c=&dummyClip;if(p)*p=&dummyProps;return 0;}
static OfxStatus eClipGetPropertySet(Handle c,Handle*o){(void)c;if(o)*o=&dummyProps;return 0;}
static OfxStatus eClipGetImage(Handle c,double t,const void*r,Handle*o){(void)c;(void)t;(void)r;if(o)*o=0;return 1;}
static OfxStatus eClipReleaseImage(Handle i){(void)i;return 0;}
static OfxStatus eClipGetRoD(Handle c,double t,void*b){(void)c;(void)t;(void)b;return 0;}
static int eAbort(Handle e){(void)e;return 0;}
static OfxStatus eMemAlloc(Handle e,size_t n,Handle*o){(void)e;(void)n;if(o)*o=0;return 0;}
static OfxStatus eMemFree(Handle m){(void)m;return 0;}
static OfxStatus eMemLock(Handle m,void**o){(void)m;if(o)*o=0;return 0;}
static OfxStatus eMemUnlock(Handle m){(void)m;return 0;}

struct EffSuite { void *fns[13]; } effSuite = {{
    eGetPropertySet,eGetParamSet,eClipDefine,eClipGetHandle,eClipGetPropertySet,
    eClipGetImage,eClipReleaseImage,eClipGetRoD,eAbort,eMemAlloc,eMemFree,eMemLock,eMemUnlock
}};

/* ---- Parameter suite (paramGetValue/SetValue are variadic) ---- */
static int dummyParam = 7;
static OfxStatus prmDefine(Handle ps,const char*ty,const char*nm,Handle*o){(void)ps;printf("    paramDefine    %-22s %s\n",ty,nm);if(o)*o=&dummyProps;return 0;}
static OfxStatus prmGetHandle(Handle ps,const char*nm,Handle*h,Handle*p){(void)ps;(void)nm;if(h)*h=&dummyParam;if(p)*p=&dummyProps;return 0;}
static OfxStatus prmSetGetPropertySet(Handle ps,Handle*o){(void)ps;if(o)*o=&dummyProps;return 0;}
static OfxStatus prmGetPropertySet(Handle p,Handle*o){(void)p;if(o)*o=&dummyProps;return 0;}
static OfxStatus prmGetValue(Handle p,...){(void)p;return 0;}
static OfxStatus prmGetValueAtTime(Handle p,double t,...){(void)p;(void)t;return 0;}
static OfxStatus prmGetDerivative(Handle p,double t,...){(void)p;(void)t;return 0;}
static OfxStatus prmGetIntegral(Handle p,double t1,double t2,...){(void)p;(void)t1;(void)t2;return 0;}
static OfxStatus prmSetValue(Handle p,...){(void)p;return 0;}
static OfxStatus prmSetValueAtTime(Handle p,double t,...){(void)p;(void)t;return 0;}
static OfxStatus prmGetNumKeys(Handle p,unsigned*o){(void)p;if(o)*o=0;return 0;}
static OfxStatus prmGetKeyTime(Handle p,unsigned i,double*o){(void)p;(void)i;if(o)*o=0;return 0;}
static OfxStatus prmGetKeyIndex(Handle p,double t,int d,int*o){(void)p;(void)t;(void)d;if(o)*o=0;return 0;}
static OfxStatus prmDeleteKey(Handle p,double t){(void)p;(void)t;return 0;}
static OfxStatus prmDeleteAllKeys(Handle p){(void)p;return 0;}
static OfxStatus prmCopy(Handle a,Handle b,double t,const void*r){(void)a;(void)b;(void)t;(void)r;return 0;}
static OfxStatus prmEditBegin(Handle ps,const char*n){(void)ps;(void)n;return 0;}
static OfxStatus prmEditEnd(Handle ps){(void)ps;return 0;}

struct PrmSuite { void *fns[20]; } prmSuite = {{
    prmDefine,prmGetHandle,prmSetGetPropertySet,prmGetPropertySet,
    prmGetValue,prmGetValueAtTime,prmGetDerivative,prmGetIntegral,prmSetValue,prmSetValueAtTime,
    prmGetNumKeys,prmGetKeyTime,prmGetKeyIndex,prmDeleteKey,prmDeleteAllKeys,prmCopy,
    prmEditBegin,prmEditEnd
}};

static const void *fetchSuite(Handle host, const char *name, int ver) {
    (void)host; (void)ver;
    if (!strcmp(name, "OfxPropertySuite"))    return &propSuite;
    if (!strcmp(name, "OfxImageEffectSuite")) return &effSuite;
    if (!strcmp(name, "OfxParameterSuite"))   return &prmSuite;
    return 0;
}

struct OfxHost { Handle host; const void *(*fetchSuite)(Handle, const char *, int); };

struct OfxPlugin {
    const char *pluginApi; int apiVersion; const char *pluginIdentifier;
    unsigned versionMajor, versionMinor;
    void (*setHost)(struct OfxHost *);
    OfxStatus (*mainEntry)(const char *, const void *, Handle, Handle);
};

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <plugin.ofx>\n", argv[0]); return 2; }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen FAILED: %s\n", dlerror()); return 1; }
    printf("[ok] dlopen\n");

    int (*getN)(void) = dlsym(lib, "OfxGetNumberOfPlugins");
    struct OfxPlugin *(*getP)(int) = dlsym(lib, "OfxGetPlugin");
    if (!getN || !getP) { fprintf(stderr, "missing entry symbols\n"); return 1; }
    printf("[ok] symbols: OfxGetNumberOfPlugins, OfxGetPlugin\n");

    int n = getN();
    printf("[ok] OfxGetNumberOfPlugins() = %d\n", n);
    if (n != 1) { fprintf(stderr, "expected 1 plugin\n"); return 1; }

    struct OfxPlugin *p = getP(0);
    if (!p) { fprintf(stderr, "OfxGetPlugin(0) returned NULL\n"); return 1; }
    printf("[ok] plugin api=%s v%u.%u id=%s\n", p->pluginApi, p->versionMajor, p->versionMinor, p->pluginIdentifier);
    if (strcmp(p->pluginApi, "OfxImageEffectPluginAPI") != 0) { fprintf(stderr, "bad api\n"); return 1; }

    struct OfxHost host = { (Handle)&host, fetchSuite };
    p->setHost(&host);
    printf("[ok] setHost\n");

    OfxStatus st = p->mainEntry("OfxActionLoad", 0, 0, 0);
    printf("[%s] OfxActionLoad -> %d\n", st == 0 ? "ok" : "FAIL", st);
    if (st != 0) return 1;

    int effect = 99;
    printf("  -- Describe --\n");
    st = p->mainEntry("OfxActionDescribe", &effect, 0, 0);
    printf("[%s] OfxActionDescribe -> %d\n", st == 0 ? "ok" : "FAIL", st);
    if (st != 0) return 1;

    printf("  -- DescribeInContext --\n");
    st = p->mainEntry("OfxImageEffectActionDescribeInContext", &effect, 0, 0);
    printf("[%s] DescribeInContext -> %d\n", st == 0 ? "ok" : "FAIL", st);
    if (st != 0) return 1;

    printf("\nALL ACTIONS OK\n");
    return 0;
}
