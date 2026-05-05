"""
B-mode rendering: log-compressed envelope of the beamformed IQ.
"""

from __future__ import annotations
import numpy as np


def b_mode(iq_bf: np.ndarray, dynamic_range_db: float = 50.0) -> np.ndarray:
    """Convert beamformed complex IQ to a normalized B-mode image in [0, 1].

    Parameters
    ----------
    iq_bf : array of complex (..., nx, nz)
    dynamic_range_db : display dynamic range (dB).

    Returns
    -------
    img : float32 in [0, 1] with same shape as the magnitude of ``iq_bf``.
    """
    env = np.abs(iq_bf).astype(np.float32)
    peak = env.max()
    if peak <= 0:
        return np.zeros_like(env)
    env_n = env / peak
    db = 20.0 * np.log10(np.maximum(env_n, 1e-12))
    img = (db + dynamic_range_db) / dynamic_range_db
    return np.clip(img, 0.0, 1.0)


def save_bmode_png(iq_bf_2d: np.ndarray, path: str,
                   dynamic_range_db: float = 50.0,
                   x: np.ndarray | None = None,
                   z: np.ndarray | None = None) -> None:
    """Save a B-mode image PNG. iq_bf_2d is (nx, nz) complex.

    The image is rendered with z increasing downward (depth) and x left-right.
    Falls back to a header-only PIL save if matplotlib is unavailable.
    """
    img = b_mode(iq_bf_2d, dynamic_range_db).T  # transpose: rows = z, cols = x
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        extent = None
        if x is not None and z is not None:
            extent = [x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3]
        fig, ax = plt.subplots(figsize=(4, 5), dpi=150)
        ax.imshow(img, cmap="gray", extent=extent, aspect="equal", vmin=0, vmax=1)
        if extent is not None:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("z (mm)")
        ax.set_title(f"B-mode  ({dynamic_range_db:.0f} dB)")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        # Fallback: write a raw 8-bit PNG via stdlib + numpy.
        import struct, zlib
        u8 = (img * 255).astype(np.uint8)
        h, w = u8.shape
        raw = b"".join(b"\x00" + bytes(row) for row in u8)
        comp = zlib.compress(raw, 9)
        def chunk(t, d):
            crc = zlib.crc32(t + d) & 0xffffffff
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", crc)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
        with open(path, "wb") as f:
            f.write(sig)
            f.write(chunk(b"IHDR", ihdr))
            f.write(chunk(b"IDAT", comp))
            f.write(chunk(b"IEND", b""))
