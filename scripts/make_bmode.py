"""Render frame 0 to a PNG B-mode image for visual sanity check."""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

# Allow running without install: add src/ to sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beamformer.data import load_h5
from beamformer.params import BeamformParams, Grid, default_grid_for, GridParams
from beamformer.mlx_kernel import beamform_mlx
from beamformer.visualize import save_bmode_png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "data.h5"))
    ap.add_argument("--out", default="docs/bmode.png")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--c", type=float, default=1540.0)
    ap.add_argument("--fnumber", type=float, default=1.5)
    ap.add_argument("--ppwl", type=float, default=4.0,
                    help="grid pixels per wavelength")
    ap.add_argument("--dynamic-range", type=float, default=50.0)
    args = ap.parse_args()

    iq, acq = load_h5(args.data)
    gp = default_grid_for(acq, iq.shape[2], iq.shape[1],
                          speed_of_sound=args.c,
                          pixels_per_wavelength=args.ppwl)
    g = Grid.from_params(gp)
    bf = BeamformParams(speed_of_sound=args.c, f_number=args.fnumber)

    print(f"grid: nx={gp.nx} nz={gp.nz}  dx={gp.grid_x_spacing_m*1e3:.3f}mm dz={gp.grid_z_spacing_m*1e3:.3f}mm")
    out = beamform_mlx(iq[args.frame:args.frame+1], acq, g, bf)
    print(f"|out| min/mean/max: {np.abs(out).min():.3f} / {np.abs(out).mean():.3f} / {np.abs(out).max():.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_bmode_png(out[0], args.out,
                   dynamic_range_db=args.dynamic_range,
                   x=g.x, z=g.z)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
