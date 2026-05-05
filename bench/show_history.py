"""
Print the rolling performance history from bench/history.jsonl as a
compact table (useful for tracking iteration history).
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    default_path = os.path.join(os.path.dirname(__file__), "history.jsonl")
    ap.add_argument("--path", default=default_path)
    ap.add_argument("--latest", type=int, default=0,
                    help="only show the last N runs (0 = all)")
    ap.add_argument("--config", default=None,
                    help="filter to a substring of the config label, e.g. 'lambda/2'")
    args = ap.parse_args()

    with open(args.path) as f:
        rows = [json.loads(L) for L in f if L.strip()]
    if args.latest:
        rows = rows[-args.latest:]

    print(f"{'when (UTC)':17s}  {'git':9s}  {'subject':28s}  {'config':28s} "
          f"{'dt':4s} {'grid':10s}  {'best ms':>8s}  {'med ms':>8s}  "
          f"{'std ms':>8s}  {'fps':>7s}  {'GoP/s':>6s}  {'note':s}")
    print("-" * 160)

    for entry in rows:
        ts = entry.get("timestamp_utc", "")[:16].replace("T", " ")
        sha = entry.get("git_sha", "")[:7]
        dirty = "*" if entry.get("git_dirty") else ""
        subject = (entry.get("git_subject") or
                   entry.get("kernel_version", "?"))[:28]
        note = entry.get("notes", "") or ""
        for r in entry.get("results", []):
            label = r.get("label", "")
            if args.config and args.config not in label:
                continue
            grid = f"{r['nx']}x{r['nz']}"
            print(f"{ts:17s}  {sha+dirty:9s}  {subject:28s}  {label:28s} "
                  f"{r['iq_dtype']:4s} {grid:10s}  {r['best_ms']:8.2f}  "
                  f"{r['median_ms']:8.2f}  {r['std_ms']:8.2f}  "
                  f"{r['fps']:7.0f}  {r['pixel_channel_gops']:6.2f}  {note}")


if __name__ == "__main__":
    main()
