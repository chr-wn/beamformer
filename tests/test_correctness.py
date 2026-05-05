"""
Correctness tests. GPU kernel should match the NumPy reference within FP32 noise.
Works without data.h5 (creates synthetic IQ).
"""

from __future__ import annotations

import os
import math
import numpy as np
import pytest

from beamformer.params import AcqParams, GridParams, BeamformParams, Grid
from beamformer.reference import beamform_reference
from beamformer.mlx_kernel import beamform_mlx


def _synthetic_iq(F: int = 2, C: int = 64, S: int = 512, seed: int = 0) -> tuple[np.ndarray, AcqParams]:
    rng = np.random.default_rng(seed)
    iq = (rng.standard_normal((F, C, S)) + 1j * rng.standard_normal((F, C, S))).astype(np.complex64)
    acq = AcqParams(
        pitch_m=3.0e-4,
        sampling_rate_hz=20e6,
        tx_freq_hz=5e6,
        tx_cycles=1,
        tx_angle_deg=0.0,
        num_channels=C,
    )
    return iq, acq


def _small_grid(acq: AcqParams) -> Grid:
    gp = GridParams(nx=33, nz=65, grid_x_spacing_m=4e-4, grid_z_spacing_m=4e-4)
    return Grid.from_params(gp)


def _diff_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    return dict(
        max_abs=float(np.abs(diff).max()),
        mean_abs=float(np.abs(diff).mean()),
        rel_l2=float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-30)),
        cosine=float(np.dot(a.ravel(), b.ravel().conj()).real /
                     (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
    )


@pytest.mark.parametrize("f_number", [0.0, 1.0, 1.5, 3.0])
def test_synthetic_matches_reference(f_number: float):
    iq, acq = _synthetic_iq()
    g = _small_grid(acq)
    bf = BeamformParams(speed_of_sound=1540.0, f_number=f_number)

    ref = beamform_reference(iq, acq, g, bf)
    out = beamform_mlx(iq, acq, g, bf)

    m = _diff_metrics(out, ref)
    # FP32 vs FP64 reference. We get cosine ~ 1 - 1e-6 typically.
    assert m["cosine"] > 0.9999, m
    assert m["rel_l2"] < 5e-3, m


def test_handles_odd_frame_count():
    """The kernel batches frames in groups of FRAMES_PER_THREAD; the last
    batch can be short. Make sure tail handling is correct for F not
    divisible by FRAMES_PER_THREAD."""
    iq_full, acq = _synthetic_iq(F=7)  # 7 % 4 = 3
    g = _small_grid(acq)
    bf = BeamformParams(speed_of_sound=1540.0, f_number=1.5)
    ref = beamform_reference(iq_full, acq, g, bf)
    out = beamform_mlx(iq_full, acq, g, bf)
    m = _diff_metrics(out, ref)
    assert m["cosine"] > 0.9999, m


def test_zero_input_is_zero():
    F, C, S = 1, 32, 256
    iq = np.zeros((F, C, S), dtype=np.complex64)
    acq = AcqParams(pitch_m=3e-4, sampling_rate_hz=20e6, tx_freq_hz=5e6,
                    tx_cycles=1, num_channels=C)
    g = Grid.from_params(GridParams(nx=8, nz=16, grid_x_spacing_m=4e-4, grid_z_spacing_m=4e-4))
    out = beamform_mlx(iq, acq, g, BeamformParams())
    assert out.shape == (1, 8, 16)
    assert np.all(out == 0)


def test_fp16_path_still_close():
    iq, acq = _synthetic_iq()
    iq = (iq * 50).astype(np.complex64)  # realistic-ish range
    g = _small_grid(acq)
    bf = BeamformParams(speed_of_sound=1540.0, f_number=1.5)

    ref = beamform_reference(iq, acq, g, bf)
    out16 = beamform_mlx(iq, acq, g, bf, iq_dtype="fp16")
    m = _diff_metrics(out16, ref)
    assert m["cosine"] > 0.999, m
    assert m["rel_l2"] < 1e-2, m


def test_apodization_disabled_uses_all_channels():
    """Disabling the F-number apodization should let energy spread further:
    edge-column magnitudes should be larger with apodization off."""
    iq, acq = _synthetic_iq()
    g = _small_grid(acq)
    on  = beamform_mlx(iq, acq, g, BeamformParams(f_number=1.5))
    off = beamform_mlx(iq, acq, g, BeamformParams(f_number=0.0))
    assert np.abs(off[:, 0, :]).mean() > np.abs(on[:, 0, :]).mean()


def test_real_data_if_available():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "data.h5")
    if not os.path.exists(path):
        pytest.skip("real data.h5 not present")
    from beamformer.data import load_h5
    from beamformer.params import default_grid_for
    iq, acq = load_h5(path)
    gp = default_grid_for(acq, iq.shape[2], iq.shape[1])
    g = Grid.from_params(gp)
    bf = BeamformParams(speed_of_sound=1540.0, f_number=1.5)
    ref = beamform_reference(iq[:2], acq, g, bf)
    out = beamform_mlx(iq[:2], acq, g, bf)
    m = _diff_metrics(out, ref)
    assert m["cosine"] > 0.99999, m
    assert m["rel_l2"] < 5e-3, m
    assert np.abs(out).max() > 100.0

