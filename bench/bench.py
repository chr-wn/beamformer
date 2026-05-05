"""
Reproducible speed benchmark for the MLX delay-and-sum beamformer.

For each config, run W warmups (10) and R timed dispatches (30), each followed by
``mx.synchronize()`` to measure the kernel run time.

See main for usage flags
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beamformer.data import load_h5
from beamformer.params import BeamformParams, Grid, default_grid_for, GridParams
from beamformer.mlx_kernel import MlxBeamformer, FRAMES_PER_THREAD


# ----------------------------------------------------------------------
# Result dataclass + history schema


@dataclass
class BenchResult:
    label: str
    nx: int
    nz: int
    grid_dx_mm: float
    grid_dz_mm: float
    iq_dtype: str
    f_number: float
    frames: int
    channels: int
    samples: int
    warmup: int
    runs: int
    best_ms: float
    median_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    std_ms: float
    fps: float
    mpix_per_s: float
    pixel_channel_gops: float
    per_frame_us: float


@dataclass
class HistoryEntry:
    timestamp_utc: str
    git_sha: str
    git_dirty: bool
    git_subject: str
    frames_per_thread: int
    machine: str
    cpu: str
    macos: str
    mlx_version: str
    notes: str
    results: list[dict] = field(default_factory=list)


# ----------------------------------------------------------------------
# Timing primitives


def _sync() -> None:
    """Hard GPU barrier so the timer captures real completion."""
    mx.synchronize()


def time_kernel(bm: MlxBeamformer, iq_mx: mx.array,
                runs: int, warmup: int, label: str = "") -> dict:
    """Run ``warmup`` warmups + ``runs`` timed dispatches, return stats."""
    # Warmup
    sys.stdout.write(f"  warmup  ")
    sys.stdout.flush()
    for _ in range(warmup):
        out = bm.run(iq_mx); mx.eval(out); _sync()
        sys.stdout.write(".")
        sys.stdout.flush()
    sys.stdout.write("\n")
    # Timed runs
    ts: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        out = bm.run(iq_mx); mx.eval(out); _sync()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
        ms = (t1 - t0) * 1e3
        sys.stdout.write(f"  run {i+1:2d}/{runs}  {ms:7.2f} ms\n")
        sys.stdout.flush()
    arr = np.asarray(ts)
    best = float(arr.min())
    sys.stdout.write(f"  -> best {best*1e3:.2f} ms, median {float(np.median(arr))*1e3:.2f} ms\n\n")
    sys.stdout.flush()
    return dict(
        best=best,
        median=float(np.median(arr)),
        p10=float(np.percentile(arr, 10)),
        p90=float(np.percentile(arr, 90)),
        mean=float(arr.mean()),
        std=float(arr.std()),
    )


def upload_iq(iq: np.ndarray, iq_dtype: str) -> mx.array:
    F, C, S = iq.shape
    if iq_dtype == "fp32":
        x = mx.array(np.ascontiguousarray(iq).view(np.float32))
    elif iq_dtype == "fp16":
        x = mx.array(np.stack([iq.real, iq.imag], axis=-1)
                       .astype(np.float16).reshape(F, C, S * 2))
    else:
        raise ValueError(f"unknown iq_dtype {iq_dtype}")
    mx.eval(x); _sync()
    return x


# ----------------------------------------------------------------------
# One configuration


def run_one(label: str, iq: np.ndarray, acq, gp: GridParams, bf: BeamformParams,
            iq_dtype: str, runs: int, warmup: int) -> BenchResult:
    g = Grid.from_params(gp)
    bm = MlxBeamformer(acq=acq, grid=g, bf=bf, iq_dtype=iq_dtype)
    iq_mx = upload_iq(iq, iq_dtype)
    F, C, S = iq.shape
    nx, nz = gp.nx, gp.nz
    st = time_kernel(bm, iq_mx, runs=runs, warmup=warmup)
    return BenchResult(
        label=label, nx=nx, nz=nz,
        grid_dx_mm=gp.grid_x_spacing_m * 1e3,
        grid_dz_mm=gp.grid_z_spacing_m * 1e3,
        iq_dtype=iq_dtype, f_number=bf.f_number,
        frames=F, channels=C, samples=S,
        warmup=warmup, runs=runs,
        best_ms=st["best"] * 1e3, median_ms=st["median"] * 1e3,
        p10_ms=st["p10"] * 1e3, p90_ms=st["p90"] * 1e3,
        mean_ms=st["mean"] * 1e3, std_ms=st["std"] * 1e3,
        fps=F / st["best"],
        mpix_per_s=F * nx * nz / st["best"] / 1e6,
        pixel_channel_gops=F * nx * nz * C / st["best"] / 1e9,
        per_frame_us=st["best"] / F * 1e6,
    )


# ----------------------------------------------------------------------
# Environment capture


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args],
                             capture_output=True, text=True, check=True,
                             cwd=os.path.dirname(__file__))
        return out.stdout.strip()
    except Exception:
        return ""


def env_info() -> dict:
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    subject = _git("log", "-1", "--pretty=%s")
    mac = ""
    cpu = ""
    try:
        mac = subprocess.run(["sw_vers", "-productVersion"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    try:
        import importlib.metadata as md
        mlx_v = md.version("mlx")
    except Exception:
        mlx_v = ""
    return dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=sha, git_dirty=dirty, git_subject=subject,
        frames_per_thread=FRAMES_PER_THREAD,
        machine=platform.machine(),
        cpu=cpu, macos=mac, mlx_version=mlx_v,
    )


# ----------------------------------------------------------------------
# Main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "data.h5"))
    ap.add_argument("--c", type=float, default=1540.0)
    ap.add_argument("--fnumber", type=float, default=1.5)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--quick", action="store_true",
                    help="one config (lambda/2 fp32) + fewer runs, for smoke checks")
    ap.add_argument("--json", default=None,
                    help="also write the latest table to this file")
    ap.add_argument("--no-history", action="store_true",
                    help="don't append to bench/history.jsonl")
    ap.add_argument("--notes", default="",
                    help="optional free-form note attached to the history row")
    args = ap.parse_args()

    runs = 8 if args.quick else args.runs
    warmup = 4 if args.quick else args.warmup

    iq, acq = load_h5(args.data)
    F, C, S = iq.shape
    bf = BeamformParams(speed_of_sound=args.c, f_number=args.fnumber)

    info = env_info()
    print(f"git:       {info['git_sha'][:8]}{'*' if info['git_dirty'] else ''}  "
          f"{info['git_subject'][:60]}")
    print(f"device:    {mx.default_device()}    {info['cpu']}    "
          f"macOS {info['macos']}    MLX {info['mlx_version']}")
    print(f"IQ stack:  {F} frames x {C} channels x {S} samples  "
          f"({iq.nbytes/1e6:.1f} MB)")
    print(f"settings:  c={args.c} m/s, f#={args.fnumber}, "
          f"f0={acq.tx_freq_hz/1e6:.1f} MHz, fs={acq.sampling_rate_hz/1e6:.1f} MHz")
    print(f"timing:    {warmup} warmup + {runs} timed dispatches per config "
          f"(mx.synchronize after each)")
    print()

    if args.quick:
        configs = [(2.0, "lambda/2 (standard)")]
        dtypes = ("fp32",)
    else:
        configs = [
            (2.0, "lambda/2 (standard)"),
            (4.0, "lambda/4 (oversampled)"),
            (8.0, "lambda/8 (very dense)"),
        ]
        dtypes = ("fp32", "fp16")

    rows: list[BenchResult] = []
    n_configs = len(configs) * len(dtypes)
    config_i = 0
    for ppwl, label_pp in configs:
        gp = default_grid_for(acq, S, C, speed_of_sound=args.c,
                              pixels_per_wavelength=ppwl)
        for dt in dtypes:
            config_i += 1
            label = f"{label_pp} / {dt}"
            print(f"[{config_i}/{n_configs}] {label}  grid {gp.nx}x{gp.nz}")
            r = run_one(label, iq, acq, gp, bf, dt,
                        runs=runs, warmup=warmup)
            rows.append(r)

    # pretty table :)
    hdr = (f"{'config':28s} {'grid':10s} {'dt':4s}  "
           f"{'best':>7s}  {'median':>7s}  {'p10':>7s}  {'p90':>7s}  "
           f"{'std':>6s}  {'fps':>7s}  {'us/fr':>7s}  {'GoP/s':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r.label:28s} {r.nx}x{r.nz:<6d} {r.iq_dtype:4s}  "
              f"{r.best_ms:6.2f}m  {r.median_ms:6.2f}m  "
              f"{r.p10_ms:6.2f}m  {r.p90_ms:6.2f}m  "
              f"{r.std_ms:5.2f}m  {r.fps:7.0f}  {r.per_frame_us:6.1f}u  "
              f"{r.pixel_channel_gops:6.2f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        print(f"\nlatest table -> {args.json}")

    if not args.no_history:
        hist_path = os.path.join(os.path.dirname(__file__), "history.jsonl")
        entry = dict(info)
        entry["notes"] = args.notes
        entry["results"] = [asdict(r) for r in rows]
        with open(hist_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"appended to {hist_path}")


if __name__ == "__main__":
    main()
