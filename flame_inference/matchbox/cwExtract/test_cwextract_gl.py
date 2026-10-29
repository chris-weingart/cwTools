#!/usr/bin/env python3
"""Run the cwExtract Matchbox in a headless OpenGL 4.3 context (moderngl + EGL)
the way Flame does, on synthetic cwSegment Object IDs maps, and compare every
mode against a numpy reference.

Host emulation as in ../cwTrimap/test_cwtrimap_gl.py: input sampler idTex
(socket 1, Front), AdskUniformBlock (binding 1), UniformBlock (binding 2),
std140 forced for hand packing, textures uploaded bottom-up. Also checks that
the decode survives float16 and 8-bit ID maps and a differently sized input.

    python test_cwextract_gl.py
"""
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VS = """#version 430
in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }
"""
ID_SCALE = 50.0  # cwSegment params::ID_SCALE


class Extract:
    def __init__(self, ctx):
        self.ctx = ctx
        root = ET.parse(HERE / "cwExtract.xml").getroot()
        sh = root.find("Shader")
        src = (HERE / "cwExtract.1.glsl").read_text()
        src = re.sub(r"layout\(\s*binding\s*=\s*(\d+)\s*\)\s*uniform\s+(\w+Block)",
                     r"layout( std140, binding = \1 ) uniform \2", src)
        self.blocks = {}
        for bname, body in re.findall(r"uniform\s+(\w+Block)\s*\{([^}]*)\}", src):
            body = "\n".join(ln.split("//")[0] for ln in body.splitlines())
            self.blocks[bname] = [tuple(x.split()) for x in (l.strip() for l in body.split(";")) if x]
        self.defaults = {u.get("Name"): u.get("Default") for u in sh.findall("Uniform") if u.get("Default") is not None}
        self.prog = ctx.program(vertex_shader=VS, fragment_shader=src)
        self.vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())

    def run(self, ids, out_wh=None, dtype="f4", **kw):
        ctx = self.ctx
        H, W = ids.shape
        OW, OH = out_wh or (W, H)
        vals = {k: (d == "True") if d in ("True", "False") else float(d) for k, d in self.defaults.items()}
        vals.update(kw)
        a = np.ascontiguousarray(np.repeat(ids[::-1, :, None], 4, 2), np.float32)
        if dtype == "f1":
            data = np.round(np.clip(a, 0, 1) * 255).astype(np.uint8).tobytes()
        elif dtype == "f2":
            data = a.astype(np.float16).tobytes()
        else:
            data = a.tobytes()
        tex = ctx.texture((W, H), 4, data, dtype=dtype)
        tex.filter = (ctx.NEAREST, ctx.NEAREST)  # XML: GL_NEAREST
        out = ctx.texture((OW, OH), 4, dtype="f4")
        fbo = ctx.framebuffer([out])
        fbo.use()
        for bname, members in self.blocks.items():
            buf = b""
            for typ, m in members:
                x = OW if m == "adsk_result_w" else OH if m == "adsk_result_h" else vals[m]
                buf += struct.pack("f", float(x)) if typ == "float" else struct.pack("i", int(x))
            binding = 1 if bname == "AdskUniformBlock" else 2
            self.prog[bname].binding = binding
            ctx.buffer(buf).bind_to_uniform_block(binding)
        tex.use(self.prog["idTex"].value)
        ctx.vertex_array(self.prog, [(self.vbo, "2f", "pos")]).render(mode=ctx.TRIANGLE_STRIP)
        ctx.finish()
        return np.frombuffer(out.read(), np.float32).reshape(OH, OW, 4)[::-1]


def reference(lab, mode=0, objR=1, objG=2, objB=3, objMatte=4, objOne=1, rangeFrom=1, rangeTo=50, invertOut=False):
    def is_(k):
        return ((lab == k) & (k > 0)).astype(np.float32)
    if mode == 1:
        o = np.repeat(is_(objOne)[..., None], 4, 2)
    elif mode == 2:
        a, b = sorted((rangeFrom, rangeTo))
        o = np.repeat(((lab > 0) & (lab >= a) & (lab <= b)).astype(np.float32)[..., None], 4, 2)
    else:
        o = np.stack([is_(objR), is_(objG), is_(objB), is_(objMatte)], 2)
    return 1.0 - o if invertOut else o


def main():
    import moderngl
    ctx = moderngl.create_standalone_context(backend="egl", require=430)
    print("[gl]", ctx.info["GL_RENDERER"], ctx.info["GL_VERSION"])
    mb = Extract(ctx)
    rng = np.random.default_rng(7)
    H, W = 217, 389
    lab = np.zeros((H, W), np.int32)
    # 50 objects: blocks plus a few one-pixel ones, object 50 included, background left
    for k in range(1, 51):
        y, x = rng.integers(0, H - 20), rng.integers(0, W - 20)
        lab[y:y + rng.integers(1, 20), x:x + rng.integers(1, 20)] = k
    lab[0, 0], lab[H - 1, W - 1], lab[5, W - 1] = 50, 1, 37
    ids = (lab / ID_SCALE).astype(np.float32)  # what cwSegment writes
    ok = True
    cases = [("RGB+Matte defaults", {}),
             ("RGB+Matte 50/37/0/1", dict(objR=50, objG=37, objB=0, objMatte=1)),
             ("One Object 50", dict(mode=1, objOne=50)),
             ("One Object 0 (none)", dict(mode=1, objOne=0)),
             ("Range 10..20", dict(mode=2, rangeFrom=10, rangeTo=20)),
             ("Range reversed 20..10", dict(mode=2, rangeFrom=20, rangeTo=10)),
             ("Range 1..50 invert", dict(mode=2, invertOut=True)),
             ("RGB+Matte invert", dict(invertOut=True))]
    for dtype in ("f4", "f2", "f1"):
        for label, kw in cases:
            got = mb.run(ids, dtype=dtype, **kw)
            want = reference(lab, **kw)
            bad = int((np.abs(got - want) > 0).any(2).sum())
            ok &= bad == 0
            print(f"[gl] {dtype} {label:26s} mismatched px={bad} {'OK' if bad == 0 else 'FAIL'}")
    # drift tolerance: +-0.009 (under half a step of 0.02) must still decode exactly
    noisy = ids + rng.uniform(-0.009, 0.009, ids.shape).astype(np.float32) * (lab > 0)
    bad = int((np.abs(mb.run(noisy, mode=2, rangeFrom=1, rangeTo=50) - reference(lab, mode=2)) > 0).any(2).sum())
    ok &= bad == 0
    print(f"[gl] f4 drift +-0.009             mismatched px={bad} {'OK' if bad == 0 else 'FAIL'}")
    # input at half size: nearest sample, never a blend of two IDs
    small = ids[::2, ::2]
    got = mb.run(small, out_wh=(W, H), mode=2, rangeFrom=1, rangeTo=50)
    vals = np.unique(got[..., 0])
    ok &= set(vals.tolist()) <= {0.0, 1.0}
    print(f"[gl] half-size input values={vals.tolist()} {'OK' if set(vals.tolist()) <= {0.0, 1.0} else 'FAIL'}")
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
