"""Visualize raw IQ data in xt-plot."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beamformer.data import load_h5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "data.h5"))
    ap.add_argument("--frames", default="0,2,5",
                    help="comma-separated frame indices to plot")
    ap.add_argument("--png", default="docs/raw_iq.png")
    ap.add_argument("--c", type=float, default=1540.0,
                    help="speed of sound for the depth axis label (m/s)")
    args = ap.parse_args()

    iq, acq = load_h5(args.data)
    F, C, S = iq.shape

    fs = acq.sampling_rate_hz
    pitch_mm = acq.pitch_m * 1e3
    c = args.c
    # depth from sample index = c * t / 2 (round-trip)
    t = np.arange(S) / fs
    depth_mm = c * t / 2.0 * 1e3   # mm
    chan_x_mm = (np.arange(C) - (C - 1) / 2.0) * pitch_mm

    frames = [int(s) for s in args.frames.split(",") if s.strip()]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(frames), figsize=(4.5 * len(frames), 6), dpi=130)
    if len(frames) == 1:
        axs = [axs]
    for ax, fi in zip(axs, frames):
        env = np.abs(iq[fi]).astype(np.float32)
        peak = env.max()
        norm = env / max(peak, 1e-9)
        db = 20 * np.log10(np.maximum(norm, 1e-6))
        im = ax.imshow(
            db.T, vmin=-50, vmax=0, cmap="viridis", aspect="auto",
            extent=[chan_x_mm[0] - pitch_mm / 2,
                    chan_x_mm[-1] + pitch_mm / 2,
                    depth_mm[-1], depth_mm[0]],
        )
        ax.set_xlabel("element x (mm)")
        ax.set_ylabel(f"depth (mm)  [c={c:.0f} m/s]")
        ax.set_title(f"frame {fi}: |IQ|  (50 dB, peak={peak:.1f})")
    fig.suptitle("Raw IQ data — point reflectors look like hyperbolas, planes look horizontal")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)
    fig.savefig(args.png)
    plt.close(fig)
    print(f"wrote {args.png}")


if __name__ == "__main__":
    main()