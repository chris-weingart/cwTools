#!/usr/bin/env python3
"""Build canary references for the cwtools-infer server (run on the server
machine, in the server venv, while nothing else needs that GPU memory for a minute).

For each model: sha256-verified bytes (page cache evicted) -> CUDA EP session
with the family's EP options -> canary inputs (canary.py) -> output
fingerprints. A second, fresh CUDA session gives the run-to-run spread; a CPU
EP session is an independent sanity check that the CUDA reference is not
itself garbage (it must agree within a loose bound).

usage: make_canary.py [model names...]     (default: every servable model
                                            without a reference)
       make_canary.py --all                (rebuild every reference)
       make_canary.py --no-cpu ...         (skip the slow CPU cross-check)
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("CWINFER_LOG_DIR", "/tmp")
sys.argv, _args = sys.argv[:1], sys.argv[1:]
import cwinfer_server as S  # noqa: E402
import canary  # noqa: E402

import onnxruntime as ort  # noqa: E402


def session(raw, m, ep, cpu=False):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if cpu:
        return ort.InferenceSession(raw, so, providers=["CPUExecutionProvider"])
    cuda = {"device_id": 0, "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": S._SEARCH[ep["conv_search"]],
            "cudnn_conv_use_max_workspace": "1" if ep["conv_max_workspace"] else "0",
            "use_tf32": "1" if ep["precision"] == "tf32" else "0"}
    s = ort.InferenceSession(raw, so, providers=[("CUDAExecutionProvider", cuda)])
    assert s.get_providers()[0] == "CUDAExecutionProvider", s.get_providers()
    return s


def main():
    all_ = "--all" in _args
    cpu = "--no-cpu" not in _args
    names = [a for a in _args if not a.startswith("--")]
    S.REGISTRY.scan(force=True)
    S.CANARY.mkdir(parents=True, exist_ok=True)
    todo = [S.REGISTRY.models[n] for n in names] if names else [
        m for m in S.REGISTRY.models.values() if m.sha and (all_ or m.canary_path() is None)]
    for m in todo:
        t0 = time.time()
        desc = S.REGISTRY.describe(m)
        raw = S.read_verified(m.path, m.sha)
        S.check_initializers_finite(raw, m.name)
        ep = m.ep_defaults()
        try:
            feeds = canary.inputs_for(desc)
        except (ValueError, KeyError) as exc:
            # e.g. the SAM3 decoder / detector / tracker (prompt, text and memory inputs): served
            # without a canary (sha256 + finite weights still checked at every load)
            print(json.dumps({"model": m.name, "skipped": f"no canary recipe ({exc})"}), flush=True)
            continue
        s1 = session(raw, m, ep)
        o1 = s1.run(None, feeds)
        o1b = s1.run(None, feeds)
        del s1
        s2 = session(raw, m, ep)
        o2 = s2.run(None, feeds)
        del s2
        names_out = [o["name"] for o in desc["outputs"]]
        meta = {"model": m.name, "sha256": m.sha, "ep": ep, "onnxruntime": ort.__version__,
                "created": datetime.datetime.now().isoformat(timespec="seconds"), "outputs": {}}
        arrays = {}
        tol = 0.0
        oc = None
        if cpu:
            sc = session(raw, m, ep, cpu=True)
            oc = sc.run(None, feeds)
            del sc
        for k, n in enumerate(names_out):
            a = o1[k]
            if not np.isfinite(a).all():
                raise SystemExit(f"{m.name}: CUDA canary output {n} non-finite (corrupt weights or memory fault?)")
            fp = canary.fingerprint(a)
            d_rep = max(canary.compare(o1b[k], fp["blocks"], fp["samples"]),
                        canary.compare(o2[k], fp["blocks"], fp["samples"]))
            scale = max(1.0, float(np.quantile(np.abs(a), 0.99)))
            t = max(10 * d_rep, 2e-3 * scale)
            info = {"d_rep": d_rep, "scale": scale, "tol": t, "mean": float(a.mean()), "std": float(a.std())}
            if oc is not None:
                d_cpu = float(np.abs(oc[k].astype(np.float64) - a.astype(np.float64)).max())
                info["d_cpu_max"] = d_cpu
                info["d_cpu_mean"] = float(np.abs(oc[k].astype(np.float64) - a).mean())
                if not (info["d_cpu_mean"] < 0.01 * scale):
                    raise SystemExit(f"{m.name}: CUDA vs CPU disagree (mean {info['d_cpu_mean']:.3g}); not trusting")
            meta["outputs"][n] = info
            tol = max(tol, t)
            arrays[f"blocks_{n}"] = fp["blocks"]
            arrays[f"samples_{n}"] = fp["samples"]
        meta["tol"] = tol
        out = S.CANARY / f"{m.name}.{m.sha[:16]}.npz"
        np.savez(out, meta=np.array(json.dumps(meta)), **arrays)
        print(json.dumps({"model": m.name, "file": out.name, "secs": round(time.time() - t0, 1), **meta["outputs"]}),
              flush=True)
        del raw


if __name__ == "__main__":
    main()
