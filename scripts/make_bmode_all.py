"""Beamform all frames in one GPU dispatch and write a PNG per frame."""

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
from beamformer.visualize import save_bmode_png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "data.h5"))
    ap.add_argument("--out-dir", default="docs/frames")
    ap.add_argument("--c", type=float, default=1540.0)
    ap.add_argument("--fnumber", type=float, default=1.5)
    ap.add_argument("--ppwl", type=float, default=4.0)
    ap.add_argument("--dynamic-range", type=float, default=55.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="frame stride; e.g. 3 keeps only every 3rd frame")
    ap.add_argument("--rename-sequential", action="store_true",
                    help="number outputs 0000, 0001, ... instead of by raw idx "
                         "(useful with --stride for ffmpeg %04d pattern).")
    ap.add_argument("--global-norm", action="store_true",
                    help="Use one peak across all frames (recommended for movie). "
                         "Otherwise each PNG is normalized to its own peak.")
    args = ap.parse_args()

    iq, acq = load_h5(args.data)
    F = iq.shape[0]
    stop = args.stop if args.stop is not None else F
    iq = iq[args.start:stop:args.stride]
    F_use = iq.shape[0]
    raw_indices = list(range(args.start, stop, args.stride))[:F_use]

    gp = default_grid_for(acq, iq.shape[2], iq.shape[1],
                          speed_of_sound=args.c, pixels_per_wavelength=args.ppwl)
    g = Grid.from_params(gp)
    bf = BeamformParams(speed_of_sound=args.c, f_number=args.fnumber)
    print(f"grid: nx={gp.nx} nz={gp.nz}, frames={args.start}..{args.start+F_use-1}")

    bm = MlxBeamformer(acq=acq, grid=g, bf=bf)
    out = bm.run(iq); mx.eval(out)
    arr = np.asarray(out)
    env = np.abs(arr)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.global_norm:
        peak = float(env.max())
        print(f"global peak = {peak:.1f}")
    for i in range(F_use):
        raw_idx = raw_indices[i]
        out_idx = i if args.rename_sequential else raw_idx
        frame_idx = raw_idx
        if args.global_norm:
            # Scale this frame so save_bmode_png sees the global peak.
            scaled = arr[i] * (peak / max(np.abs(arr[i]).max(), 1e-9)) if False else arr[i]
            # Easier path: bypass save_bmode_png's per-image norm by rendering manually
            normed = env[i] / peak
            db = 20 * np.log10(np.maximum(normed, 1e-12))
            img = (db + args.dynamic_range) / args.dynamic_range
            img = np.clip(img, 0, 1).T
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4, 5), dpi=120)
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="equal",
                          extent=[g.x[0]*1e3, g.x[-1]*1e3, g.z[-1]*1e3, g.z[0]*1e3])
                ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
                ax.set_title(f"frame {frame_idx}  ({args.dynamic_range:.0f} dB, global)")
                fig.tight_layout()
                fig.savefig(os.path.join(args.out_dir, f"frame_{out_idx:04d}.png"))
                plt.close(fig)
            except Exception:
                pass
        else:
            save_bmode_png(arr[i], os.path.join(args.out_dir, f"frame_{out_idx:04d}.png"),
                           dynamic_range_db=args.dynamic_range, x=g.x, z=g.z)

    print(f"wrote {F_use} PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
