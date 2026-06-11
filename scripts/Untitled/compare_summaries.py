#!/usr/bin/env python3
"""Quick side-by-side comparison of two faithfulness summary files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_summary(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        m = re.search(r"Accuracy \(full CoT\)\s*=\s*([0-9.]+)", s)
        if m:
            metrics["Accuracy"] = m.group(1)
            continue
        m = re.search(r"BIFR \(Biased-Input Flip Rate\)\s*=\s*([0-9.]+)", s)
        if m:
            metrics["BIFR"] = m.group(1)
            continue
        m = re.search(r"PS \(Paraphrase Stability\)\s*=\s*([0-9.]+)", s)
        if m:
            metrics["PS"] = m.group(1)
            continue
        m = re.search(r"EAR_(\d+)%\s*=\s*([0-9.]+)", s)
        if m:
            metrics[f"EAR_{m.group(1)}%"] = m.group(2)
            continue
        m = re.search(r"empty@(\d+)%\s*=\s*([0-9.]+)", s)
        if m:
            metrics[f"empty@{m.group(1)}%"] = m.group(2)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two faithfulness summary files.")
    p.add_argument("before", type=Path, help="Earlier summary (e.g. original pilot)")
    p.add_argument("after", type=Path, help="Later summary (e.g. rerun with fix)")
    args = p.parse_args()

    before = parse_summary(args.before)
    after = parse_summary(args.after)
    keys = sorted(set(before) | set(after), key=lambda k: (k.split("_")[0], k))

    print(f"{'Metric':<28} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 60)
    for key in keys:
        b = before.get(key, "-")
        a = after.get(key, "-")
        if b != "-" and a != "-":
            delta = float(a) - float(b)
            delta_s = f"{delta:+.3f}"
        else:
            delta_s = "-"
        print(f"{key:<28} {b:>10} {a:>10} {delta_s:>10}")


if __name__ == "__main__":
    main()
