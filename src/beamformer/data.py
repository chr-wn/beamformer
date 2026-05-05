"""
Load IQ data and probe parameters.
"""

from __future__ import annotations
import h5py
import numpy as np
from .params import AcqParams

def load_h5(path: str) -> tuple[np.ndarray, AcqParams]:
    with h5py.File(path, "r") as f:
        iq = f["iq_data"][...]  # (frames,channels,samples) complex64
        attrs = dict(f.attrs)
    if iq.dtype != np.complex64:
        iq = iq.astype(np.complex64)
    iq = np.ascontiguousarray(iq)
    acq = AcqParams(
        pitch_m=float(attrs["pitch_m"]),
        sampling_rate_hz=float(attrs["sampling_rate_hz"]),
        tx_freq_hz=float(attrs["tx_freq_hz"]),
        tx_cycles=int(attrs["tx_cycles"]),
        tx_angle_deg=float(attrs.get("tx_angle_deg", 0.0)),
        num_channels=iq.shape[1],
    )
    return iq, acq
