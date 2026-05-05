"""Acquisition / grid / beamforming parameter dataclasses"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class AcqParams:
    """Probe + transmit parameters that come with the data."""

    pitch_m: float
    sampling_rate_hz: float
    tx_freq_hz: float
    tx_cycles: int
    tx_angle_deg: float = 0.0
    num_channels: int | None = None  # filled in from data shape

    @property
    def tx_angle_rad(self) -> float:
        return math.radians(self.tx_angle_deg)

    def element_x(self, num_channels: int) -> np.ndarray:
        """Element x-positions (m), centered at 0."""
        n = num_channels
        return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * self.pitch_m


@dataclass(frozen=True)
class GridParams:
    """Output image grid in meters."""

    nx: int
    nz: int
    grid_x_spacing_m: float
    grid_z_spacing_m: float
    z0_m: float = 0.0  # depth of first row (z=0 is probe face by default)


@dataclass(frozen=True)
class BeamformParams:
    """Reconstruction options."""

    speed_of_sound: float = 1540.0
    f_number: float = 1.5  # dynamic receive aperture; <=0 disables apodization
    t0_s: float = 0.0      # acquisition start time (sample 0 corresponds to this t)


@dataclass(frozen=True)
class Grid:
    """Materialized grid coordinates (in meters)."""

    x: np.ndarray  # (nx,) float32
    z: np.ndarray  # (nz,) float32

    @classmethod
    def from_params(cls, gp: GridParams) -> "Grid":
        x = (np.arange(gp.nx, dtype=np.float32) - (gp.nx - 1) / 2.0) * gp.grid_x_spacing_m
        z = gp.z0_m + np.arange(gp.nz, dtype=np.float32) * gp.grid_z_spacing_m
        return cls(x=x.astype(np.float32), z=z.astype(np.float32))


def default_grid_for(acq: AcqParams, num_samples: int, num_channels: int,
                     speed_of_sound: float = 1540.0,
                     pixels_per_wavelength: float = 2.0,
                     lateral_pad_pct: float = 0.0) -> GridParams:
    """A reasonable default imaging grid for ultrasound: lateral spans the
    array (optionally padded), axial covers full sample-dictated depth, and the
    grid spacing is set to (lambda / pixels_per_wavelength) in both axes.

    Useful for tests and benchmarks if no explicit grid is specified.
    """
    lam = speed_of_sound / acq.tx_freq_hz  # wavelength (m)
    dx = lam / pixels_per_wavelength
    dz = lam / pixels_per_wavelength
    aperture_m = acq.pitch_m * num_channels
    lateral_extent = aperture_m * (1.0 + 2 * lateral_pad_pct)
    nx = int(round(lateral_extent / dx)) | 1  # odd so a column lies on x=0
    max_depth = (num_samples / acq.sampling_rate_hz) * speed_of_sound / 2.0
    nz = int(round(max_depth / dz))
    return GridParams(nx=nx, nz=nz, grid_x_spacing_m=dx, grid_z_spacing_m=dz)
