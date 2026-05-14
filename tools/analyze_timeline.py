#!/usr/bin/env python3
"""Summarize ds4cuda forward_timeline.csv into a performance tree."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable


BYTE_FIELDS = ("weight_bytes", "input_bytes", "output_bytes", "scratch_bytes")


def as_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def as_int(row: Dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "") or 0))
    except ValueError:
        return 0


def add_row(acc: Dict[str, float], row: Dict[str, str]) -> None:
    acc["ms"] += as_float(row, "ms")
    acc["count"] += 1
    acc["kernels"] += as_int(row, "kernels")
    for key in BYTE_FIELDS:
        acc[key] += as_int(row, key)


def fmt_pct(ms: float, total: float) -> str:
    return f"{(100.0 * ms / total) if total > 0 else 0.0:6.2f}%"


def fmt_gbps(bytes_total: float, ms: float) -> str:
    if ms <= 0:
        return "   0.00"
    return f"{bytes_total / (ms * 1.0e6):7.2f}"


def print_group(title: str, groups: Dict[str, Dict[str, float]], total_ms: float, top: int) -> None:
    print(f"\n{title}")
    print("name,count,kernels,total_ms,pct,est_GBps,weight_GiB,io_MiB,scratch_MiB")
    ranked = sorted(groups.items(), key=lambda kv: kv[1]["ms"], reverse=True)
    for name, acc in ranked[:top]:
        io_bytes = acc["input_bytes"] + acc["output_bytes"]
        total_bytes = acc["weight_bytes"] + io_bytes + acc["scratch_bytes"]
        print(
            f"{name},{int(acc['count'])},{int(acc['kernels'])},"
            f"{acc['ms']:.3f},{fmt_pct(acc['ms'], total_ms)},"
            f"{fmt_gbps(total_bytes, acc['ms'])},"
            f"{acc['weight_bytes'] / (1024 ** 3):.3f},"
            f"{io_bytes / (1024 ** 2):.3f},"
            f"{acc['scratch_bytes'] / (1024 ** 2):.3f}"
        )


def print_fault_tree(
    by_category: Dict[str, Dict[str, float]],
    by_stage: Dict[str, Dict[str, float]],
    total_ms: float,
    total_kernels: int,
    n_tokens: int,
    top: int,
) -> None:
    print("\nFault Tree")
    print("kind,name,total_ms,pct,kernels,est_GBps,next_action")

    categories = sorted(by_category.items(), key=lambda kv: kv[1]["ms"], reverse=True)
    for name, acc in categories:
        pct = (100.0 * acc["ms"] / total_ms) if total_ms > 0 else 0.0
        if pct < 5.0:
            continue
        total_bytes = (
            acc["weight_bytes"] + acc["input_bytes"] +
            acc["output_bytes"] + acc["scratch_bytes"]
        )
        if name in {"q8", "moe", "compressor"}:
            action = "profile_hot_kernel_or_fuse_launches"
        elif name in {"hc", "router", "norm", "rope", "elementwise"}:
            action = "check_launch_count_and_fusion"
        else:
            action = "inspect_stage_timeline"
        print(
            f"primary_branch,{name},{acc['ms']:.3f},{pct:.2f},"
            f"{int(acc['kernels'])},{fmt_gbps(total_bytes, acc['ms'])},{action}"
        )

    stages = sorted(by_stage.items(), key=lambda kv: kv[1]["ms"], reverse=True)
    for name, acc in stages[:top]:
        pct = (100.0 * acc["ms"] / total_ms) if total_ms > 0 else 0.0
        total_bytes = (
            acc["weight_bytes"] + acc["input_bytes"] +
            acc["output_bytes"] + acc["scratch_bytes"]
        )
        action = "optimize_or_microbench"
        if pct < 2.0:
            action = "defer"
        print(
            f"leaf_stage,{name},{acc['ms']:.3f},{pct:.2f},"
            f"{int(acc['kernels'])},{fmt_gbps(total_bytes, acc['ms'])},{action}"
        )

    kernels_per_token = (float(total_kernels) / float(n_tokens)) if n_tokens > 0 else 0.0
    launch_action = "ok"
    if kernels_per_token > 1500.0:
        launch_action = "launch_fragmentation_high_reduce_small_stages_or_graph"
    elif kernels_per_token > 800.0:
        launch_action = "launch_fragmentation_watch"
    print(
        f"launch_fragmentation,kernels_per_token,{kernels_per_token:.3f},"
        f"0.00,{total_kernels},0.00,{launch_action}"
    )


def load_rows(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def analyze(rows: Iterable[Dict[str, str]], top: int) -> None:
    rows = list(rows)
    total_ms = sum(as_float(r, "ms") for r in rows)
    total_kernels = sum(as_int(r, "kernels") for r in rows)
    tokens = sorted({as_int(r, "token") for r in rows})
    layers = sorted({as_int(r, "layer") for r in rows if as_int(r, "layer") >= 0})

    print("Performance Tree")
    print(
        f"records={len(rows)} tokens={len(tokens)} layers={len(layers)} "
        f"total_ms={total_ms:.3f} kernels={total_kernels}"
    )

    by_category: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_stage: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_layer: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in rows:
        add_row(by_category[row.get("category", "") or "unknown"], row)
        add_row(by_stage[row.get("stage", "") or "unknown"], row)
        add_row(by_layer[str(as_int(row, "layer"))], row)

    print_group("By Category", by_category, total_ms, top)
    print_group("Top Stages", by_stage, total_ms, top)
    print_group("By Layer", by_layer, total_ms, top)
    print_fault_tree(by_category, by_stage, total_ms, total_kernels, len(tokens), top)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    rows = load_rows(args.csv_path)
    analyze(rows, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
