#!/usr/bin/env python3
"""Inspect a CPU prefill logits dump produced by ds4_native.

Used to verify the shape and basic statistics of the float32 vocab logits
written by ``DS4_CPU_DUMP_PREFILL_LOGITS=path/to/file.bin ds4_native ...``.

Usage:
    uv run python scripts/inspect_prefill_logits.py /path/to/prefill_hello.bin

The dump is a raw little-endian float32 buffer of length DS4_N_VOCAB (129280).
No header, no metadata. See ds4.c::write_f32_binary_file (line 546) and the
DS4_CPU_DUMP_PREFILL_LOGITS hook in generate_raw_swa_cpu (ds4.c line 14455).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

DS4_N_VOCAB = 129280
EXPECTED_BYTES = DS4_N_VOCAB * 4  # float32


def inspect(path: str, top_k: int = 5) -> int:
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    size = os.path.getsize(path)
    print(f"path:         {path}")
    print(f"file size:    {size} bytes")
    print(f"expected:     {EXPECTED_BYTES} bytes ({DS4_N_VOCAB} * float32)")
    if size != EXPECTED_BYTES:
        print(f"ERROR: size mismatch (got {size}, expected {EXPECTED_BYTES})",
              file=sys.stderr)
        return 1

    a = np.fromfile(path, dtype=np.float32)
    if a.shape != (DS4_N_VOCAB,):
        print(f"ERROR: shape mismatch: {a.shape}", file=sys.stderr)
        return 1

    has_nan = bool(np.isnan(a).any())
    has_inf = bool(np.isinf(a).any())
    print(f"shape:        {a.shape}")
    print(f"dtype:        {a.dtype}")
    print(f"min:          {float(a.min()):.6f}")
    print(f"max:          {float(a.max()):.6f}")
    print(f"mean:         {float(a.mean()):.6f}")
    print(f"std:          {float(a.std()):.6f}")
    print(f"has nan:      {has_nan}")
    print(f"has inf:      {has_inf}")

    # Argmax and top-K (descending).
    order = np.argsort(a)[::-1]
    top_idx = order[:top_k]
    top_val = a[top_idx]
    print(f"top-{top_k} indices:  {top_idx.tolist()}")
    print(f"top-{top_k} values:   {[round(float(v), 4) for v in top_val]}")
    print(f"argmax:       {int(top_idx[0])}")
    print(f"argmax value: {float(top_val[0]):.6f}")

    # Softmax probability of argmax (numerically stable).
    shifted = a - a.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    print(f"top-{top_k} probs:    {[round(float(p), 4) for p in probs[top_idx]]}")
    print(f"argmax prob:  {float(probs[top_idx[0]]):.6f}")

    if has_nan or has_inf:
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="Path to the prefill logits dump (float32, no header)")
    ap.add_argument("-k", "--top-k", type=int, default=5,
                    help="How many top entries to print (default 5)")
    args = ap.parse_args()
    return inspect(args.path, top_k=args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
