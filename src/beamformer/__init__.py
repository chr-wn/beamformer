"""Fast 2d das ultrasound beamformer for Apple Silicon."""

from .params import AcqParams, GridParams, BeamformParams, Grid, default_grid_for
from .data import load_h5
from .reference import beamform_reference

__all__ = [
    "AcqParams",
    "GridParams",
    "BeamformParams",
    "Grid",
    "default_grid_for",
    "load_h5",
    "beamform_reference",
]

def __getattr__(name):
    """Lazy-load the MLX path so import works without MLX installed."""
    if name in ("beamform_mlx", "MlxBeamformer"):
        from . import mlx_kernel
        return getattr(mlx_kernel, name)
    raise AttributeError(name)
