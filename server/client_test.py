#!/usr/bin/env python3
"""API tests for cwtools-infer (run anywhere with numpy + zstandard + pillow,
e.g. on the server: venv/bin/python server/client_test.py http://127.0.0.1:8201).

Checks: /health, /models, a lut8-webp and a lut8-grad-zstd round trip vs the
canary fingerprint, u8 output mode, sha256 mismatch -> 409, bad input name ->
400, corrupted crc -> 400, unknown model -> 404, 3 parallel requests queue,
LRU cap (3 models -> at most MAX_SESSIONS loaded), /release.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canary  # noqa: E402
import cwtensor as ct  # noqa: E402

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8201").rstrip("/")
FAIL = []


def get(path):
    with urllib.request.urlopen(URL + path, timeout=60) as r:
        return json.loads(r.read())


def post(body: bytes, path="/infer"):
    req = urllib.request.Request(URL + path, data=body, method="POST", headers={"Content-Type": ct.CONTENT_TYPE})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not ok:
        FAIL.append(name)


def request(desc, feeds, sha=None, codec="auto", extra=None, corrupt_crc=False, rename=None):
    b = ct.Builder()
    ins = []
    for k, a in feeds.items():
        c = codec
        if c == "auto":
            c = "lut8-webp" if a.ndim == 4 and a.shape[1] >= 3 else "xor-shuf-zstd"
        try:
            d = b.tensor(k, a, c)
        except ct.CodecError:
            d = b.tensor(k, a, "xor-shuf-zstd")
        if corrupt_crc and "crc32" in d:
            d["crc32"] ^= 1
        if rename:
            d["name"] = rename
        ins.append(d)
    h = {"model": desc["name"], "sha256": sha or desc["sha256"], "inputs": ins, **(extra or {})}
    return b.pack(h)


def main():
    h = get("/health")
    check("health ok", h.get("ok") and h.get("api") == 1, f"vram {h.get('vram')}")
    ms = {m["name"]: m for m in get("/models")["models"]}
    # vitmatte_core is required; the BiRefNet checks (LRU) are skipped when those models
    # are not installed (e.g. a server set up with the cwRefine core only).
    lru = [n for n in ("BiRefNet-general-bb_swin_v1_tiny-epoch_232_gs", "BiRefNet_dynamic-general-epoch_174_1536_gs_fp16") if n in ms]
    check("models listed", "vitmatte_core" in ms, f"{len(ms)} models")
    if "vitmatte_core" not in ms:
        print(f"FAILED: vitmatte_core not on the server ({sorted(ms)})")
        sys.exit(1)
    vm = ms["vitmatte_core"]
    feeds = canary.inputs_for(vm)
    ref = np.load(Path(__file__).resolve().parent.parent / "canary" / f"vitmatte_core.{vm['sha256'][:16]}.npz") \
        if (Path(__file__).resolve().parent.parent / "canary").is_dir() else None

    for codec in ("lut8-webp", "lut8-grad-zstd"):
        st, body = post(request(vm, feeds, codec=codec))
        ok = st == 200
        info = ""
        if ok:
            hd, area = ct.unpack(body)
            a = ct.decode_tensor(hd["outputs"][0], area)
            if ref is not None:
                d = canary.compare(a, ref["blocks_alphas"], ref["samples_alphas"])
                ok = d < 1e-4
                info = f"canary diff {d:.2g}, timings {hd['timings_ms']}"
        check(f"vitmatte_core {codec} round trip", ok, info or str(body[:200]))

    st, body = post(request(vm, feeds, extra={"output_quantize": "u8_trunc255"}))
    if st == 200:
        hd, area = ct.unpack(body)
        u = ct.decode_tensor(hd["outputs"][0], area)
        st2, body2 = post(request(vm, feeds))
        hd2, area2 = ct.unpack(body2)
        f = ct.decode_tensor(hd2["outputs"][0], area2)
        exp = np.clip(f * np.float32(255), 0, 255).astype(np.uint8)
        check("u8_trunc255 = trunc(float * 255)", u.dtype == np.uint8 and np.array_equal(u, exp),
              f"{len(body)} vs {len(body2)} bytes")
    else:
        check("u8_trunc255", False, str(body[:200]))

    st, body = post(request(vm, feeds, sha="0" * 64))
    check("sha mismatch -> 409", st == 409 and b"sha_mismatch" in body, body[:120].decode())
    st, body = post(request(vm, feeds, rename="nope"))
    check("bad input name -> 400", st == 400 and b"bad_inputs" in body, body[:120].decode())
    st, body = post(request(vm, feeds, codec="lut8-grad-zstd", corrupt_crc=True))
    check("crc mismatch -> 400", st == 400 and b"crc32" in body, body[:120].decode())
    st, body = post(request({**vm, "name": "no_such_model"}, feeds))
    check("unknown model -> 404", st == 404, body[:120].decode())

    # 3 parallel requests: serialised on the GPU, later ones report queue time
    res = []

    def one():
        s, b = post(request(vm, feeds))
        res.append(ct.unpack(b)[0]["timings_ms"] if s == 200 else {"status": s})
    th = [threading.Thread(target=one) for _ in range(3)]
    [t.start() for t in th]
    [t.join() for t in th]
    q = sorted(r.get("queue", -1) for r in res)
    check("parallel requests queue (one at a time)", len(res) == 3 and q[-1] > 0, f"queue ms {q}")

    # LRU: touch 3 models -> at most max_sessions loaded
    if len(lru) < 2:
        print(f"SKIP LRU check (needs both BiRefNet general models, have {lru})")
    for name in lru:
        m = ms[name]
        st, body = post(request(m, canary.inputs_for(m)))
        check(f"{name} infer", st == 200, ct.unpack(body)[0]["timings_ms"] if st == 200 else body[:200].decode())
    h = get("/health")
    cap = h["settings"]["max_sessions"]
    check("LRU cap", len(h["sessions"]) <= cap, f"{[s['model'] for s in h['sessions']]} cap {cap}, vram {h['vram']}")
    st, body = post(b"", "/release")
    h = get("/health")
    check("release", st == 200 and not h["sessions"], f"{body.decode()} vram after {h['vram']}")
    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
