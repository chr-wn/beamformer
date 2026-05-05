"""
Generate synthetic IQ data for point targets.
"""

from __future__ import annotations
from typing import Iterable
import numpy as np
from .params import AcqParams

    
def synthetic_iq(targets: Iterable[tuple[float, float, complex]],
                 acq: AcqParams,
                 num_samples: int,
                 num_frames: int = 1,
                 speed_of_sound: float = 1540.0,
                 noise_std: float = 0.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate (frames, channels, samples) complex64 IQ for point targets.

    Parameters
    ----------
    targets : iterable of (x_m, z_m, amplitude)
    acq : AcqParams
    num_samples : fast-time samples per frame
    num_frames : duplicate the same noiseless field this many times
    speed_of_sound : m/s
    noise_std : std-dev of complex Gaussian noise added per sample
    rng : optional numpy Generator for reproducible noise

    Returns
    -------
    iq : (F, C, S) complex64
    """
    if rng is None:
        rng = np.random.default_rng(0)
    C = acq.num_channels
    if C is None:
        raise ValueError("AcqParams.num_channels must be set")
    F = int(num_frames)
    S = int(num_samples)
    elem_x = acq.element_x(C)         # (C,)
    fs = acq.sampling_rate_hz
    f0 = acq.tx_freq_hz
    c = float(speed_of_sound)

    # Time vector (s).
    t = (np.arange(S) / fs).astype(np.float64)  # (S,)

    # Pulse envelope: Hann window of full duration = tx_cycles / f0.
    # env(u) = 0.5 * (1 + cos(pi * u / half)) for |u| <= half, else 0
    half_pulse = 0.5 * acq.tx_cycles / f0  # half-width in seconds

    iq = np.zeros((C, S), dtype=np.complex128)
    for x_t, z_t, amp in targets:
        # Round-trip delay per element.
        tau = (z_t + np.sqrt((x_t - elem_x) ** 2 + z_t ** 2)) / c  # (C,)
        # Compute envelope and contribution per channel.
        for n in range(C):
            u = t - tau[n]                                # (S,) seconds offset
            inside = np.abs(u) <= half_pulse
            if not inside.any():
                continue
            env = np.zeros(S, dtype=np.float64)
            env[inside] = 0.5 * (1.0 + np.cos(np.pi * u[inside] / half_pulse))
            iq[n] += amp * np.exp(-1j * 2 * np.pi * f0 * tau[n]) * env

    iq = iq.astype(np.complex64)
    # Broadcast to F frames; add per-frame independent noise if requested.
    out = np.broadcast_to(iq[None, ...], (F, C, S)).astype(np.complex64).copy()
    if noise_std > 0:
        n_re = rng.standard_normal(out.shape) * noise_std
        n_im = rng.standard_normal(out.shape) * noise_std
        out = (out + (n_re + 1j * n_im)).astype(np.complex64)
    return out
