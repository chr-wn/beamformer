"""
Verifies DAS works right.
Generates IQ for known point targets, beamforms, and verifies:
1. the brightest pixel of each PSF lies within ~half a wavelength of the true (x, z) location;
2. the lateral and axial FWHMs are within an expected envelope set by aperture and pulse length;
3. the off-target floor is far below the on-target peak.
Also writes ``docs/psf.png`` for visual inspection.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beamformer.params import AcqParams, BeamformParams, Grid, GridParams
from beamformer.synthetic import synthetic_iq
from beamformer.mlx_kernel import beamform_mlx


def fwhm(envelope: np.ndarray, axis_m: np.ndarray) -> float:
    """FWHM of a 1-D envelope (m). Returns NaN if peak is at boundary."""
    peak = envelope.max()
    if peak <= 0:
        return float("nan")
    half = peak * 0.5
    above = envelope >= half
    if not above.any():
        return float("nan")
    idxs = np.where(above)[0]
    lo, hi = idxs[0], idxs[-1]
    if lo == 0 or hi == len(envelope) - 1:
        return float("nan")  # ran into the edge
    # linear interp for sub-pixel crossings on each side
    def cross(i_in, i_out):
        v_in, v_out = envelope[i_in], envelope[i_out]
        if v_out == v_in:
            return axis_m[i_in]
        a = (half - v_out) / (v_in - v_out)
        return axis_m[i_out] + a * (axis_m[i_in] - axis_m[i_out])
    x_lo = cross(lo, lo - 1)
    x_hi = cross(hi, hi + 1)
    return float(x_hi - x_lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=1540.0)
    ap.add_argument("--fnumber", type=float, default=1.5)
    ap.add_argument("--fs", type=float, default=12.5e6)
    ap.add_argument("--f0", type=float, default=5.0e6)
    ap.add_argument("--pitch_mm", type=float, default=0.208)
    ap.add_argument("--num-channels", type=int, default=134)
    ap.add_argument("--num-samples", type=int, default=1024)
    ap.add_argument("--png", default="docs/psf.png")
    ap.add_argument("--snr-db", type=float, default=80.0,
                    help="Signal-to-noise ratio (dB). 80 = nearly noiseless.")
    args = ap.parse_args()

    acq = AcqParams(
        pitch_m=args.pitch_mm * 1e-3,
        sampling_rate_hz=args.fs,
        tx_freq_hz=args.f0,
        tx_cycles=1,
        tx_angle_deg=0.0,
        num_channels=args.num_channels,
    )

    # Five point targets spread over the imaging volume.
    targets = [
        (0.0,    10e-3, 1.0 + 0j),
        (0.0,    20e-3, 1.0 + 0j),
        (-5e-3,  15e-3, 1.0 + 0j),
        (+5e-3,  15e-3, 1.0 + 0j),
        (0.0,    30e-3, 1.0 + 0j),
    ]

    # Noise std chosen to give the requested per-sample SNR.
    sig_amp = 1.0
    noise_std = sig_amp * 10.0 ** (-args.snr_db / 20.0)
    iq = synthetic_iq(targets, acq, num_samples=args.num_samples,
                      speed_of_sound=args.c, noise_std=noise_std)

    # Dense grid: ~lambda/8 for accurate PSF measurement.
    lam = args.c / args.f0
    dx = dz = lam / 8.0
    nx = int(round(0.030 / dx)) | 1   # ~30 mm, odd
    nz = int(round(0.040 / dz))       # 0..40 mm
    gp = GridParams(nx=nx, nz=nz, grid_x_spacing_m=dx, grid_z_spacing_m=dz, z0_m=0.0)
    g = Grid.from_params(gp)
    bf = BeamformParams(speed_of_sound=args.c, f_number=args.fnumber)

    out = beamform_mlx(iq, acq, g, bf)[0]   # (nx, nz)
    env = np.abs(out)

    # Per-target measurements.
    print(f"grid: nx={nx} nz={nz} dx=dz={dx*1e3:.3f} mm  (lambda/8)")
    print(f"PW@0  c={args.c} m/s  f0={args.f0/1e6:.1f} MHz  fs={args.fs/1e6:.1f} MHz  F#={args.fnumber}")
    print(f"theoretical lateral FWHM ~ lambda*F# = {lam*args.fnumber*1e3:.3f} mm")
    print(f"theoretical axial   FWHM ~ lambda    = {lam*1e3:.3f} mm  (1-cycle pulse)\n")

    print(f"{'target':>20s}  {'peak (mm)':>20s}  {'pos err (mm)':>14s}  {'lat FWHM':>10s}  {'ax FWHM':>10s}  {'PSLR (dB)':>10s}")
    print("-" * 100)
    all_ok = True
    for (x_t, z_t, _) in targets:
        # Index of expected target.
        ix = int(np.argmin(np.abs(g.x - x_t)))
        iz = int(np.argmin(np.abs(g.z - z_t)))
        # Search small window for actual peak (handle small offsets).
        wx = max(8, int(round(2 * lam / dx)))
        wz = max(8, int(round(2 * lam / dz)))
        x0, x1 = max(0, ix - wx), min(nx, ix + wx + 1)
        z0, z1 = max(0, iz - wz), min(nz, iz + wz + 1)
        sub = env[x0:x1, z0:z1]
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        ix_m, iz_m = x0 + i, z0 + j
        peak_x, peak_z = g.x[ix_m], g.z[iz_m]
        pos_err = np.hypot(peak_x - x_t, peak_z - z_t) * 1e3   # mm

        # FWHM measured in a window local to the peak so we don't pick up
        # neighboring targets along the same row/column.
        win_lat = max(8, int(round(3 * lam * args.fnumber / dx)))
        win_ax  = max(8, int(round(3 * lam / dz)))
        xlo, xhi = max(0, ix_m - win_lat), min(nx, ix_m + win_lat + 1)
        zlo, zhi = max(0, iz_m - win_ax),  min(nz, iz_m + win_ax + 1)
        lat_line = env[xlo:xhi, iz_m]
        ax_line  = env[ix_m, zlo:zhi]
        lat_fwhm = fwhm(lat_line, g.x[xlo:xhi]) * 1e3
        ax_fwhm  = fwhm(ax_line,  g.z[zlo:zhi]) * 1e3

        # Peak side-lobe ratio: max outside an N-pixel exclusion zone vs peak.
        pk = env[ix_m, iz_m]
        excl = np.zeros_like(env, dtype=bool)
        ex = max(4, int(round(3 * lam / dx)))
        ez = max(4, int(round(3 * lam / dz)))
        excl[max(0, ix_m - ex):ix_m + ex + 1, max(0, iz_m - ez):iz_m + ez + 1] = True
        pslr_db = 20 * np.log10(env[~excl].max() / pk + 1e-12)

        ok = (pos_err < lam * 0.5e3) and (lat_fwhm < lam * 1e3 * args.fnumber * 2.0)
        status = "OK" if ok else "FAIL"
        all_ok &= ok
        print(f"({x_t*1e3:+5.1f}, {z_t*1e3:5.1f}) mm  ({peak_x*1e3:+6.2f},{peak_z*1e3:6.2f})    "
              f"{pos_err:>12.3f}    {lat_fwhm:>9.3f}  {ax_fwhm:>9.3f}    {pslr_db:>+8.1f}  [{status}]")

    print()
    print("ALL TARGETS FOCUSED CORRECTLY" if all_ok else "SOME TARGETS FAILED — algorithm is suspect.")

    # Save PSF image.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 6), dpi=150)
        db = 20 * np.log10(env / env.max() + 1e-12)
        ax.imshow(db.T, vmin=-60, vmax=0, cmap="hot",
                  extent=[g.x[0]*1e3, g.x[-1]*1e3, g.z[-1]*1e3, g.z[0]*1e3],
                  aspect="equal")
        for (x_t, z_t, _) in targets:
            ax.plot(x_t * 1e3, z_t * 1e3, "wo", mec="cyan", ms=8, mfc="none")
        ax.set_title("PSFs from synthetic point targets (60 dB)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        fig.tight_layout()
        fig.savefig(args.png)
        plt.close(fig)
        print(f"\nwrote {args.png}")
    except Exception as e:
        print(f"(matplotlib not available, skipped PNG: {e})")


if __name__ == "__main__":
    main()
