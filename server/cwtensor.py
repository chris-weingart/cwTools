"""CWT1 tensor container: the wire format of the cwTools remote inference
server (see API.md). Shared by the server and the Python test client.

Container (request and response bodies, Content-Type application/x-cwtensor):

    b"CWT1" | u32 LE header length | header JSON (UTF-8) | blob area

The header lists tensors; each tensor names its blobs as [offset, length]
inside the blob area. All codecs are lossless; zstd frames carry their
content checksum (xxh64), so a corrupted transfer fails to decode instead of
producing different numbers.

Codecs (`codec` field of a tensor):
  raw            little-endian bytes as-is
  zstd           zstd(raw)
  xor-shuf-zstd  float tensors: every element XOR its left neighbour along the
                 last axis (bit pattern), then byte-shuffle (all byte-0s, all
                 byte-1s, ...), then zstd. Smooth float fields compress 2-4x.
  grad-zstd      uint8 tensors: 2-D gradient filter over the last two axes
                 (mod 256), then zstd.
  lut8-grad-zstd tensors whose every channel (axis 1) takes <= 256 distinct
                 values (e.g. a normalised uint8 image): blob `lut` = C x 256
                 values of the tensor dtype (raw), blob `data` = uint8 indices,
                 2-D gradient filtered per H x W plane (dx then dy, mod 256),
                 then zstd. Decoding is LUT[idx]: bit-exact, by construction.
"""
from __future__ import annotations

import io
import json
import struct
import zlib

import numpy as np
import zstandard

MAGIC = b"CWT1"
CONTENT_TYPE = "application/x-cwtensor"

DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "uint8": np.uint8,
    "int64": np.int64,
    "int32": np.int32,
    "bool": np.bool_,
}
_UINT_OF = {4: np.uint32, 2: np.uint16, 1: np.uint8, 8: np.uint64}


class CodecError(ValueError):
    pass


# ───────────────────────────────────────────────────────────── container ──
def pack(header: dict, blobs: list[bytes]) -> bytes:
    """Build a container. `header` must already reference blobs by index
    via [offset, length] (use `Builder` for convenience)."""
    hj = json.dumps(header, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<I", len(hj)) + hj + b"".join(blobs)


def unpack(body: bytes) -> tuple[dict, memoryview]:
    if len(body) < 8 or body[:4] != MAGIC:
        raise CodecError("not a CWT1 container (bad magic)")
    (hl,) = struct.unpack_from("<I", body, 4)
    if 8 + hl > len(body):
        raise CodecError("truncated header")
    try:
        header = json.loads(bytes(body[8:8 + hl]))
    except Exception as exc:  # noqa: BLE001 — client error, reported
        raise CodecError(f"header JSON: {exc}") from exc
    return header, memoryview(body)[8 + hl:]


class Builder:
    """Accumulates blobs and tensor descriptors for one container."""

    def __init__(self, level: int = 3):
        self.blobs: list[bytes] = []
        self.off = 0
        self.level = level

    def blob(self, b: bytes) -> list[int]:
        ref = [self.off, len(b)]
        self.blobs.append(b)
        self.off += len(b)
        return ref

    def tensor(self, name: str, arr: np.ndarray, codec: str = "auto") -> dict:
        arr = np.ascontiguousarray(arr)
        dt = _dtype_name(arr.dtype)
        if codec == "auto":
            codec = "xor-shuf-zstd" if arr.dtype.kind == "f" else "zstd"
        d = {"name": name, "dtype": dt, "shape": list(arr.shape), "codec": codec}
        if codec == "raw":
            d["data"] = self.blob(_le(arr).tobytes())
        elif codec == "zstd":
            d["data"] = self.blob(_zc(self.level).compress(_le(arr).tobytes()))
        elif codec == "xor-shuf-zstd":
            d["data"] = self.blob(_zc(self.level).compress(_xor_shuf(arr)))
        elif codec == "grad-zstd":
            if arr.dtype != np.uint8 or arr.ndim < 2:
                raise CodecError("grad-zstd is for uint8 tensors with H, W")
            d["data"] = self.blob(_zc(self.level).compress(_grad_enc(arr).tobytes()))
        elif codec == "lut8-grad-zstd":
            lut, idx = lut8_split(arr)
            d["lut"] = self.blob(_le(lut).tobytes())
            d["crc32"] = zlib.crc32(idx.tobytes())
            if arr.ndim >= 3 and arr.shape[1] >= 3:
                x = idx.copy()
                x[:, 0] -= idx[:, 1]
                x[:, 2] -= idx[:, 1]
                d["xform"] = "gdiff"
            else:
                x = idx
            d["data"] = self.blob(_zc(self.level).compress(_grad_enc(x).tobytes()))
        elif codec == "lut8-webp":
            from PIL import Image
            lut, idx = lut8_split(arr)
            if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[1] < 3:
                raise CodecError("lut8-webp needs shape [1, C>=3, H, W]")
            d["lut"] = self.blob(_le(lut).tobytes())
            d["crc32"] = zlib.crc32(idx.tobytes())
            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(idx[0, :3].transpose(1, 2, 0)), "RGB").save(
                buf, "WEBP", lossless=True, quality=0, method=0)
            d["webp"] = self.blob(buf.getvalue())
            if arr.shape[1] > 3:
                d["data"] = self.blob(_zc(self.level).compress(_grad_enc(idx[:, 3:]).tobytes()))
        else:
            raise CodecError(f"unknown codec {codec!r}")
        return d

    def pack(self, header: dict) -> bytes:
        return pack(header, self.blobs)


def decode_tensor(d: dict, area: memoryview, max_bytes: int = 1 << 31) -> np.ndarray:
    try:
        dtype = np.dtype(DTYPES[d["dtype"]]).newbyteorder("<")
        shape = tuple(int(s) for s in d["shape"])
        codec = d.get("codec", "raw")
    except KeyError as exc:
        raise CodecError(f"tensor {d.get('name')!r}: missing/unknown field {exc}") from exc
    if any(s < 0 for s in shape):
        raise CodecError(f"tensor {d.get('name')!r}: negative dim")
    n = int(np.prod(shape, dtype=np.int64))
    nbytes = n * dtype.itemsize
    if nbytes > max_bytes:
        raise CodecError(f"tensor {d.get('name')!r}: {nbytes} bytes exceeds limit")

    def blob(key):
        o, ln = d[key]
        if o < 0 or ln < 0 or o + ln > len(area):
            raise CodecError(f"tensor {d.get('name')!r}: blob {key} out of range")
        return area[o:o + ln]

    def unz(b, size):
        try:
            out = zstandard.ZstdDecompressor().decompress(bytes(b), max_output_size=size)
        except zstandard.ZstdError as exc:
            raise CodecError(f"tensor {d.get('name')!r}: zstd: {exc}") from exc
        if len(out) != size:
            raise CodecError(f"tensor {d.get('name')!r}: decoded {len(out)} bytes, expected {size}")
        return out

    if codec == "raw":
        b = bytes(blob("data"))
        if len(b) != nbytes:
            raise CodecError(f"tensor {d.get('name')!r}: {len(b)} bytes, expected {nbytes}")
        a = np.frombuffer(b, dtype).reshape(shape)
    elif codec == "zstd":
        a = np.frombuffer(unz(blob("data"), nbytes), dtype).reshape(shape)
    elif codec == "xor-shuf-zstd":
        a = _xor_unshuf(unz(blob("data"), nbytes), dtype, shape)
    elif codec == "grad-zstd":
        if dtype != np.uint8 or len(shape) < 2:
            raise CodecError("grad-zstd is for uint8 tensors with H, W")
        a = _grad_dec(np.frombuffer(unz(blob("data"), nbytes), np.uint8).reshape(shape))
    elif codec in ("lut8-grad-zstd", "lut8-webp"):
        if len(shape) < 3:
            raise CodecError("lut8 needs a tensor with a channel axis 1 and H, W")
        c, hh, ww = shape[1], shape[-2], shape[-1]
        lb = bytes(blob("lut"))
        if len(lb) != c * 256 * dtype.itemsize:
            raise CodecError(f"tensor {d.get('name')!r}: lut has {len(lb)} bytes, expected {c * 256 * dtype.itemsize}")
        lut = np.frombuffer(lb, dtype).reshape(c, 256)
        if codec == "lut8-webp":
            if len(shape) != 4 or shape[0] != 1 or c < 3:
                raise CodecError("lut8-webp needs shape [1, C>=3, H, W]")
            idx = np.empty(shape, np.uint8)
            idx[0, :3] = _webp_rgb(bytes(blob("webp")), ww, hh).transpose(2, 0, 1)
            if c > 3:
                rest = (1, c - 3, hh, ww)
                idx[:, 3:] = _grad_dec(np.frombuffer(unz(blob("data"), int(np.prod(rest))), np.uint8).reshape(rest))
        else:
            idx = _grad_dec(np.frombuffer(unz(blob("data"), n), np.uint8).reshape(shape))
            if d.get("xform") == "gdiff" and c >= 3:
                idx = idx.copy()
                idx[:, 0] += idx[:, 1]
                idx[:, 2] += idx[:, 1]
            elif d.get("xform") not in (None, "none"):
                raise CodecError(f"tensor {d.get('name')!r}: unknown xform {d.get('xform')!r}")
        if "crc32" in d and zlib.crc32(idx.tobytes()) != int(d["crc32"]):
            raise CodecError(f"tensor {d.get('name')!r}: index crc32 mismatch (corrupt transfer or decode)")
        a = np.empty(shape, dtype)
        for ch in range(c):
            a[:, ch] = lut[ch][idx[:, ch]]
    else:
        raise CodecError(f"tensor {d.get('name')!r}: unknown codec {codec!r}")
    return a.astype(dtype.newbyteorder("="), copy=False)


# ──────────────────────────────────────────────────────────────── codecs ──
def _dtype_name(dt) -> str:
    for k, v in DTYPES.items():
        if np.dtype(v) == np.dtype(dt).newbyteorder("="):
            return k
    raise CodecError(f"unsupported dtype {dt}")


def _le(a: np.ndarray) -> np.ndarray:
    return a.astype(a.dtype.newbyteorder("<"), copy=False)


def _zc(level):
    return zstandard.ZstdCompressor(level=level, write_checksum=True, write_content_size=True)


def _xor_shuf(a: np.ndarray) -> bytes:
    isz = a.dtype.itemsize
    u = _le(a).view(np.dtype(_UINT_OF[isz]).newbyteorder("<")).reshape(-1, a.shape[-1] if a.ndim else 1)
    x = u.copy()
    x[:, 1:] ^= u[:, :-1]
    b = np.frombuffer(x.tobytes(), np.uint8).reshape(-1, isz)
    return np.ascontiguousarray(b.T).tobytes()


def _xor_unshuf(raw: bytes, dtype, shape) -> np.ndarray:
    isz = dtype.itemsize
    n = len(raw) // isz
    b = np.frombuffer(raw, np.uint8).reshape(isz, n)
    u = np.frombuffer(np.ascontiguousarray(b.T).tobytes(), np.dtype(_UINT_OF[isz]).newbyteorder("<"))
    last = shape[-1] if len(shape) else 1
    u = np.bitwise_xor.accumulate(u.reshape(-1, last), axis=1)
    return u.view(dtype).reshape(shape)


def _webp_rgb(b: bytes, w: int, h: int) -> np.ndarray:
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(b))
        if im.format != "WEBP":
            raise CodecError(f"lut8-webp blob is {im.format}, not WEBP")
        im.load()
    except CodecError:
        raise
    except Exception as exc:  # noqa: BLE001 — client data error
        raise CodecError(f"webp decode: {exc}") from exc
    if im.size != (w, h) or im.mode != "RGB":
        raise CodecError(f"webp is {im.mode} {im.size}, expected RGB {(w, h)}")
    return np.asarray(im)


def lut8_split(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per channel (axis 1): LUT of the distinct values (<= 256) + uint8 indices."""
    if a.ndim < 3:
        raise CodecError("lut8 needs a channel axis 1")
    c = a.shape[1]
    lut = np.zeros((c, 256), a.dtype)
    idx = np.empty(a.shape, np.uint8)
    for ch in range(c):
        plane = a[:, ch]
        bits = plane.view(_UINT_OF[a.dtype.itemsize])
        vals, inv = np.unique(bits, return_inverse=True)
        if len(vals) > 256:
            raise CodecError(f"channel {ch} has {len(vals)} distinct values (> 256): use xor-shuf-zstd")
        lut[ch, :len(vals)] = vals.view(a.dtype)
        idx[:, ch] = inv.reshape(plane.shape).astype(np.uint8)
    return lut, idx


def _grad_enc(idx: np.ndarray) -> np.ndarray:
    """2-D gradient filter over the last two axes, mod 256 (decode = 2 cumsums)."""
    d = idx.copy()
    d[..., :, 1:] = idx[..., :, 1:] - idx[..., :, :-1]
    e = d.copy()
    e[..., 1:, :] = d[..., 1:, :] - d[..., :-1, :]
    return e


def _grad_dec(e: np.ndarray) -> np.ndarray:
    d = np.cumsum(e, axis=-2, dtype=np.uint8)
    return np.cumsum(d, axis=-1, dtype=np.uint8)
