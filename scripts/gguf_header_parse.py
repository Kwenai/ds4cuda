#!/usr/bin/env python3
"""
Minimal GGUF v3 header parser.
Reads header + metadata KV + tensor info section ONLY.
Never touches the tensor data section.
"""
import struct
import sys
import os
import json
from collections import defaultdict

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# GGUF metadata value types
GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16 = 0, 1, 2, 3
GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL = 4, 5, 6, 7
GGUF_STRING, GGUF_ARRAY = 8, 9
GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64 = 10, 11, 12

SCALAR = {
    GGUF_UINT8: ("B", 1), GGUF_INT8: ("b", 1),
    GGUF_UINT16: ("<H", 2), GGUF_INT16: ("<h", 2),
    GGUF_UINT32: ("<I", 4), GGUF_INT32: ("<i", 4), GGUF_FLOAT32: ("<f", 4),
    GGUF_BOOL: ("B", 1),
    GGUF_UINT64: ("<Q", 8), GGUF_INT64: ("<q", 8), GGUF_FLOAT64: ("<d", 8),
}

# (name, block_elems, block_bytes) — same as ds4.c gguf_types[] table
TENSOR_TYPES = {
    0:  ("f32",      1,   4),
    1:  ("f16",      1,   2),
    2:  ("q4_0",    32,  18),
    3:  ("q4_1",    32,  20),
    6:  ("q5_0",    32,  22),
    7:  ("q5_1",    32,  24),
    8:  ("q8_0",    32,  34),
    9:  ("q8_1",    32,  40),
    10: ("q2_k",   256,  84),
    11: ("q3_k",   256, 110),
    12: ("q4_k",   256, 144),
    13: ("q5_k",   256, 176),
    14: ("q6_k",   256, 210),
    15: ("q8_k",   256, 292),
    16: ("iq2_xxs",256,  66),
    17: ("iq2_xs", 256,  74),
    18: ("iq3_xxs",256,  98),
    19: ("iq1_s",  256, 110),
    20: ("iq4_nl", 256,  50),
    21: ("iq3_s",  256, 110),
    22: ("iq2_s",  256,  82),
    23: ("iq4_xs", 256, 136),
    24: ("i8",       1,   1),
    25: ("i16",      1,   2),
    26: ("i32",      1,   4),
    27: ("i64",      1,   8),
    28: ("f64",      1,   8),
    29: ("iq1_m",  256,  56),
    30: ("bf16",     1,   2),
}


class Cursor:
    __slots__ = ("buf", "pos")

    def __init__(self, buf, pos=0):
        self.buf = buf
        self.pos = pos

    def take(self, n):
        v = self.buf[self.pos:self.pos + n]
        self.pos += n
        return v

    def u32(self):
        v = struct.unpack_from("<I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def u64(self):
        v = struct.unpack_from("<Q", self.buf, self.pos)[0]
        self.pos += 8
        return v

    def scalar(self, t):
        fmt, sz = SCALAR[t]
        if fmt.startswith("<"):
            v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        else:
            v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += sz
        return v

    def string(self):
        n = self.u64()
        s = bytes(self.buf[self.pos:self.pos + n]).decode("utf-8", errors="replace")
        self.pos += n
        return s

    def value(self, t):
        if t in SCALAR:
            return self.scalar(t)
        if t == GGUF_STRING:
            return self.string()
        if t == GGUF_ARRAY:
            inner = self.u32()
            ln = self.u64()
            # avoid materializing huge arrays in full; only first/last
            if inner in SCALAR:
                fmt, sz = SCALAR[inner]
                # Just return type+len summary plus a small head sample
                head = []
                for _ in range(min(ln, 8)):
                    head.append(self.scalar(inner))
                if ln > 8:
                    skip_bytes = (ln - 8) * sz
                    self.pos += skip_bytes
                return {"type": "array", "inner": inner, "len": ln, "head": head}
            else:
                items = []
                for _ in range(ln):
                    items.append(self.value(inner))
                return items
        raise ValueError(f"unknown gguf metadata type {t}")


def align_up(x, a):
    return (x + a - 1) // a * a


def parse_header(path, max_bytes=64 * 1024 * 1024):
    """Read the first max_bytes of file (sufficient for header + KV + tensor info)."""
    with open(path, "rb") as f:
        # Need to read header+kv+tensor info; for 1300+ tensors with long names this is well under 1 MB
        head = f.read(max_bytes)
    cur = Cursor(head)
    magic = cur.u32()
    if magic != GGUF_MAGIC:
        raise SystemExit(f"bad magic {magic:#x}")
    version = cur.u32()
    n_tensors = cur.u64()
    n_kv = cur.u64()
    if version != 3:
        raise SystemExit(f"only GGUF v3 supported (got {version})")

    alignment = 32
    kv_pairs = []
    for _ in range(n_kv):
        key = cur.string()
        t = cur.u32()
        v = cur.value(t)
        if key == "general.alignment" and t == GGUF_UINT32:
            alignment = v
        kv_pairs.append((key, t, v))

    tensors = []
    for _ in range(n_tensors):
        name = cur.string()
        ndim = cur.u32()
        dims = [cur.u64() for _ in range(ndim)]
        ttype = cur.u32()
        rel_offset = cur.u64()
        tensors.append({
            "name": name,
            "ndim": ndim,
            "dims": dims,
            "type": ttype,
            "rel_offset": rel_offset,
        })

    tensor_data_pos = align_up(cur.pos, alignment)

    # compute bytes / elements / abs_offset
    for t in tensors:
        info = TENSOR_TYPES.get(t["type"])
        if info is None:
            t["bytes"] = None
            t["elements"] = None
            continue
        name, be, bb = info
        elems = 1
        for d in t["dims"]:
            elems *= d
        if elems == 0:
            t["bytes"] = 0
        else:
            blocks = (elems + be - 1) // be
            t["bytes"] = blocks * bb
        t["elements"] = elems
        t["abs_offset"] = tensor_data_pos + t["rel_offset"]

    return {
        "version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "alignment": alignment,
        "tensor_data_pos": tensor_data_pos,
        "kv": kv_pairs,
        "tensors": tensors,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DS4CUDA_GGUF")
    if not path:
        sys.stderr.write(
            "usage: gguf_header_parse.py <gguf-path> [out-dir]\n"
            "  or set DS4CUDA_GGUF env var. See CONTRIBUTING.md.\n"
        )
        sys.exit(2)
    if len(sys.argv) > 2:
        out_dir = sys.argv[2]
    else:
        # Default: current working directory.
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    file_size = os.path.getsize(path)
    print(f"file: {path}")
    print(f"file size: {file_size} bytes ({file_size / 1024**3:.3f} GiB)")

    h = parse_header(path)
    print(f"GGUF v{h['version']} alignment={h['alignment']}")
    print(f"n_kv={h['n_kv']} n_tensors={h['n_tensors']} tensor_data_pos={h['tensor_data_pos']}")

    # KV summary
    kv_path = os.path.join(out_dir, "kv.txt")
    with open(kv_path, "w") as f:
        for key, t, v in h["kv"]:
            if isinstance(v, dict) and v.get("type") == "array":
                f.write(f"{key}\t(type={t})\tarray<inner={v['inner']}, len={v['len']}> head={v['head']}\n")
            elif isinstance(v, list):
                f.write(f"{key}\t(type={t})\tlist[{len(v)}] head={v[:3]}\n")
            else:
                f.write(f"{key}\t(type={t})\t{v!r}\n")
    print(f"wrote {kv_path}")

    # Tensor table
    tt_path = os.path.join(out_dir, "tensors.tsv")
    with open(tt_path, "w") as f:
        f.write("name\ttype_id\ttype_name\tndim\tdims\telements\tbytes\tabs_offset\n")
        for t in h["tensors"]:
            tn = TENSOR_TYPES.get(t["type"], ("?", 0, 0))[0]
            f.write(f"{t['name']}\t{t['type']}\t{tn}\t{t['ndim']}\t{','.join(str(d) for d in t['dims'])}\t{t['elements']}\t{t['bytes']}\t{t['abs_offset']}\n")
    print(f"wrote {tt_path}")

    # Aggregate stats
    by_type_count = defaultdict(int)
    by_type_bytes = defaultdict(int)
    total_bytes = 0
    total_elems = 0
    for t in h["tensors"]:
        tn = TENSOR_TYPES.get(t["type"], ("?", 0, 0))[0]
        by_type_count[tn] += 1
        by_type_bytes[tn] += t["bytes"] or 0
        total_bytes += t["bytes"] or 0
        total_elems += t["elements"] or 0

    print("\n--- type histogram ---")
    for tn in sorted(by_type_count):
        b = by_type_bytes[tn]
        print(f"  {tn:<10s} count={by_type_count[tn]:>5d}  bytes={b:>15d}  ({b/1024**3:.3f} GiB)")

    print(f"\n  total tensor bytes: {total_bytes} ({total_bytes/1024**3:.3f} GiB)")
    print(f"  total elements:     {total_elems}  ({total_elems/1e9:.3f} B)")
    print(f"  file size:          {file_size}")
    print(f"  header + alignment: {h['tensor_data_pos']}")
    print(f"  data span:          {file_size - h['tensor_data_pos']}")

    # Categorize: routed MoE experts vs main body Q8_0 vs other
    routed_bytes = 0
    main_q8_bytes = 0
    other_bytes = 0
    for t in h["tensors"]:
        nm = t["name"]
        tn = TENSOR_TYPES.get(t["type"], ("?", 0, 0))[0]
        b = t["bytes"] or 0
        if "ffn_gate_exps" in nm or "ffn_up_exps" in nm or "ffn_down_exps" in nm:
            routed_bytes += b
        elif tn == "q8_0":
            main_q8_bytes += b
        else:
            other_bytes += b
    print("\n--- categorized ---")
    print(f"  routed MoE experts: {routed_bytes} ({routed_bytes/1024**3:.3f} GiB)")
    print(f"  main-body Q8_0:     {main_q8_bytes} ({main_q8_bytes/1024**3:.3f} GiB)")
    print(f"  other (F16/F32/I32): {other_bytes} ({other_bytes/1024**3:.3f} GiB)")
    print(f"  sum:                {routed_bytes+main_q8_bytes+other_bytes}")


if __name__ == "__main__":
    main()
