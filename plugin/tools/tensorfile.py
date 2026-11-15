"""
The container both the exported model and the golden vectors use: a small
JSON header followed by raw little-endian float32 tensors. The full
specification is in plugin/tools/README.md; in short:

  offset  size  content
  0       8     magic, the ASCII bytes "ENMTENSR"
  8       4     uint32 LE: container version (1)
  12      4     uint32 LE: header length H in bytes
  16      H     header: UTF-8 JSON object, padded with spaces so that
                16 + H is a multiple of 64
  16 + H  ...   data: each tensor at data_start + its "offset" (a multiple
                of 64), 4 * prod(shape) bytes, zero bytes between tensors

The header object has at least "kind" (what the file holds), "kind_version"
and "tensors": [{"name", "shape", "dtype": "f32", "offset"}], in the order
the tensors are stored. Everything else in the header depends on "kind".
"""
import json
import struct

import numpy as np

MAGIC = b"ENMTENSR"
VERSION = 1
ALIGN = 64


def _pad(n):
    return -n % ALIGN


def write(path, header, tensors):
    """header: dict (without "tensors"); tensors: list of (name, array).
    Arrays are stored as C-order little-endian float32."""
    entries, data = [], b""
    for name, array in tensors:
        array = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
        if not np.all(np.isfinite(array)):
            raise ValueError(f"tensor {name} has non-finite values")
        data += b"\0" * _pad(len(data))  # padding between tensors, none after the last
        entries.append({"name": name, "shape": list(array.shape), "dtype": "f32", "offset": len(data)})
        data += array.tobytes()
    names = [e["name"] for e in entries]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor names")
    text = json.dumps({**header, "tensors": entries}, indent=1).encode("utf-8")
    text += b" " * _pad(16 + len(text))
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(text)) + text + data)


def read(path):
    """-> (header dict, {name: float32 array}), checking the layout."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 16:
        raise ValueError(f"{path}: too short for a tensor file")
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: not a tensor file (magic {raw[:8]!r})")
    version, length = struct.unpack("<II", raw[8:16])
    if version != VERSION:
        raise ValueError(f"{path}: container version {version}, this reader knows {VERSION}")
    start = 16 + length
    if start % ALIGN:
        raise ValueError(f"{path}: data doesn't start on a {ALIGN}-byte boundary")
    header = json.loads(raw[16:start].decode("utf-8"))
    tensors = {}
    for e in header["tensors"]:
        if e["dtype"] != "f32" or e["offset"] % ALIGN:
            raise ValueError(f"{path}: bad tensor entry {e}")
        n = int(np.prod(e["shape"], dtype=np.int64))
        begin = start + e["offset"]
        if begin + 4 * n > len(raw):
            raise ValueError(f"{path}: tensor {e['name']} runs past the end of the file")
        tensors[e["name"]] = np.frombuffer(raw, dtype="<f4", count=n, offset=begin).reshape(e["shape"]).copy()
    return header, tensors
