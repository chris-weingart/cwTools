"""Canary inputs + compact output fingerprints for the integrity check.

The inputs are regenerated deterministically from code (no stored tensors);
the stored reference per model (canary/<name>.<sha16>.npz) holds only output
fingerprints: 32x32 block means + 4096 sampled values per output. A flipped
weight or activation bit that matters shows up in both.
"""
from __future__ import annotations

import numpy as np

DYN = 512  # size used for dynamic H / W
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def synth_rgb(h: int, w: int) -> np.ndarray:
    """Deterministic display-referred test picture (u8 HxWx3): gradient
    background, textured noise, a warm disc with a frayed edge."""
    rng = np.random.default_rng(1234)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([40 + 120 * y / h, 60 + 80 * x / w, 150 - 60 * y / h], -1)
    bg += rng.normal(0, 6, (h, w, 3)).astype(np.float32)
    cy, cx, r = h * 0.55, w * 0.5, min(h, w) * 0.28
    d = np.hypot(y - cy, x - cx)
    edge = r + 6 * np.sin(np.arctan2(y - cy, x - cx) * 40)
    a = np.clip((edge - d) / 3 + 0.5, 0, 1)[..., None]
    fg = np.stack([210 + 0 * y, 150 + 30 * np.sin(x / 9), 110 + 0 * y], -1)
    img = bg * (1 - a) + fg * a
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def synth_disc(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(y - h * 0.55, x - w * 0.5)
    return (d < min(h, w) * 0.28).astype(np.float32)


def _dim(s, default=DYN):
    return s if isinstance(s, int) and s > 0 else default


def inputs_for(desc: dict) -> dict:
    """Feeds for a model description from the registry (/models entry)."""
    feeds = {}
    sidecar_inputs = ((desc.get("sidecar") or {}).get("ModelDescription") or {}).get("Inputs") or []
    norm = next((i.get("Normalization") for i in sidecar_inputs if i.get("Normalization")), None)
    for inp in desc["inputs"]:
        shape = [_dim(s, 1) if k < 2 else _dim(s) for k, s in enumerate(inp["shape"])]
        n, c, h, w = shape
        rgb = synth_rgb(h, w)
        if c == 4:  # ViTMatte core pixel_values: HF norm RGB + trimap
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dd = np.hypot(yy - h * 0.55, xx - w * 0.5) - min(h, w) * 0.28
            tri = np.where(dd < -20, 255, np.where(dd > 20, 0, 128)).astype(np.uint8)
            pv = np.empty((1, 4, h, w), np.float32)
            for ch in range(3):
                pv[0, ch] = ((rgb[..., ch].astype(np.float64) / 255).astype(np.float32) - 0.5) / 0.5
            pv[0, 3] = (tri.astype(np.float64) / 255).astype(np.float32)
            feeds[inp["name"]] = pv
        elif c == 3:
            t = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            if norm:
                m = np.array(norm["Mean"], np.float32)[None, :, None, None]
                s = np.array(norm["StandardDeviation"], np.float32)[None, :, None, None]
                t = (t - m) / s
            feeds[inp["name"]] = np.ascontiguousarray(t, np.float32)
        elif c == 1:
            name = inp["name"].lower()
            feeds[inp["name"]] = (np.zeros((1, 1, h, w), np.float32) if "trimap" in name
                                  else synth_disc(h, w)[None, None])
        else:
            raise ValueError(f"no canary recipe for input {inp}")
        if inp["dtype"] == "float16":
            feeds[inp["name"]] = feeds[inp["name"]].astype(np.float16)
    return feeds


def fingerprint(a: np.ndarray) -> dict:
    a = np.asarray(a, np.float64)
    flat = a.reshape(-1)
    h, w = a.shape[-2], a.shape[-1]
    p = a.reshape(-1, h, w)[0]
    bh, bw = max(1, h // 32), max(1, w // 32)
    blocks = p[:bh * 32, :bw * 32].reshape(32, bh, 32, bw).mean(axis=(1, 3)) if h >= 32 and w >= 32 else p
    idx = np.random.default_rng(99).integers(0, flat.size, 4096)
    return {"blocks": blocks, "samples": flat[idx], "idx": idx}


def compare(a: np.ndarray, ref_blocks: np.ndarray, ref_samples: np.ndarray) -> float:
    fp = fingerprint(a)
    return float(max(np.abs(fp["blocks"] - ref_blocks).max(), np.abs(fp["samples"] - ref_samples).max()))
