"""
Find optimal speed of sound c (see 2.E of paper)
Also creates png of metric vs c plus B-mode thumbnails at three c values.
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beamformer.data import load_h5
from beamformer.params import BeamformParams, Grid, default_grid_for
from beamformer.mlx_kernel import MlxBeamformer


def metrics(env: np.ndarray) -> dict:
    """Sharpness metrics on the magnitude image."""
    e = env.astype(np.float64)
    e2 = e * e
    e4 = e2 * e2
    K = float(e4.mean() / max(e2.mean() ** 2, 1e-30) - 1.0)
    p = e2 / max(e2.sum(), 1e-30)
    H = float(-(p[p > 0] * np.log(p[p > 0])).sum())  # entropy
    Linf = float(e.max())
    return dict(K=K, neg_H=-H, Linf=Linf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "data.h5"))
    ap.add_argument("--frames", default="0,2",
                    help="comma-separated frame indices to score (avg).")
    ap.add_argument("--c-min", type=float, default=1400.0)
    ap.add_argument("--c-max", type=float, default=1700.0)
    ap.add_argument("--steps", type=int, default=31)
    ap.add_argument("--fnumber", type=float, default=1.5)
    ap.add_argument("--ppwl", type=float, default=4.0,
                    help="grid pixels per wavelength (uses --c-min for lambda)")
    ap.add_argument("--png", default="docs/c_sweep.png")
    args = ap.parse_args()

    iq, acq = load_h5(args.data)
    F, C, S = iq.shape
    frames = [int(s) for s in args.frames.split(",") if s.strip()]
    iq_sel = np.ascontiguousarray(iq[frames])

    cs = np.linspace(args.c_min, args.c_max, args.steps)

    # fixed grid, not biased by varying pixel size. Choose grid based on the lower bound (= densest).
    gp = default_grid_for(acq, S, C, speed_of_sound=float(args.c_min),
                          pixels_per_wavelength=args.ppwl)
    g = Grid.from_params(gp)
    print(f"fixed grid: nx={gp.nx} nz={gp.nz} dx=dz={gp.grid_x_spacing_m*1e3:.3f} mm")

    rows = []
    print(f"sweeping c over [{cs[0]:.1f}, {cs[-1]:.1f}] m/s, {len(cs)} steps, "
          f"frames={frames}, F#={args.fnumber}")
    print(f"{'c (m/s)':>8s} {'kurtosis':>10s} {'-entropy':>10s} {'max|x|':>12s}")
    for c in cs:
        bf = BeamformParams(speed_of_sound=float(c), f_number=args.fnumber)
        bm = MlxBeamformer(acq=acq, grid=g, bf=bf)
        out = bm.run(iq_sel); mx.eval(out)
        env = np.abs(np.asarray(out)).astype(np.float64)
        # Average metrics across the selected frames.
        m = {k: float(np.mean([metrics(env[i])[k] for i in range(env.shape[0])]))
             for k in ("K", "neg_H", "Linf")}
        rows.append((float(c), m["K"], m["neg_H"], m["Linf"]))
        print(f"{c:8.1f} {m['K']:10.3f} {m['neg_H']:10.3f} {m['Linf']:12.2f}")

    rows = np.array(rows)
    c_vals, K, negH, Linf = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    c_K = float(c_vals[int(np.argmax(K))])
    c_H = float(c_vals[int(np.argmax(negH))])
    c_L = float(c_vals[int(np.argmax(Linf))])
    print()
    print(f"argmax kurtosis : c = {c_K:.1f} m/s")
    print(f"argmax neg-H    : c = {c_H:.1f} m/s")
    print(f"argmax max|x|   : c = {c_L:.1f} m/s")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(13, 4), dpi=130)
        for a, y, ylab, c_opt in zip(
            ax,
            [K, negH, Linf],
            ["kurtosis (higher = sharper)", "-entropy (higher = sharper)",
             "max|x| (higher = better focus)"],
            [c_K, c_H, c_L],
        ):
            a.plot(c_vals, y, "-o", ms=3)
            a.axvline(c_opt, color="r", ls="--", lw=1, label=f"argmax: {c_opt:.0f}")
            a.set_xlabel("c (m/s)")
            a.set_ylabel(ylab)
            a.legend()
            a.grid(alpha=0.3)
        fig.suptitle("Sharpness vs speed of sound — image is sharpest at the medium's true c")
        fig.tight_layout()
        fig.savefig(args.png)
        plt.close(fig)
        print(f"wrote {args.png}")
    except Exception as e:
        print(f"(no matplotlib: {e})")


if __name__ == "__main__":
    main()
