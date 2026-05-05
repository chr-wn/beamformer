"""
DAS algorithm based on the paper (meant for validation).
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from .params import AcqParams, BeamformParams, Grid


def beamform_reference(
  iq: np.ndarray,
  acq: AcqParams,
  grid: Grid,
  bf: BeamformParams = BeamformParams(),
) -> np.ndarray:
  """Beamform a stack of IQ frames with NumPy using a sparse DAS matrix.

  Parameters
  ----------
  iq : (F, C, S) complex64
  acq : AcqParams
  grid : Grid (nx, nz)
  bf : BeamformParams

  Returns
  -------
  out : (F, nx, nz) complex64
  """
  if iq.ndim != 3:
    raise ValueError(f"iq must be (F,C,S), got {iq.shape}")
  
  F, C, S = iq.shape
  nx, nz = grid.x.shape[0], grid.z.shape[0]

  fs = np.float64(acq.sampling_rate_hz)
  f0 = np.float64(acq.tx_freq_hz)
  c = np.float64(bf.speed_of_sound)
  t0 = np.float64(bf.t0_s)
  theta = np.float64(acq.tx_angle_rad)
  elem_x = acq.element_x(C).astype(np.float64)

  x = grid.x.astype(np.float64)
  z = grid.z.astype(np.float64)

  # Generate 2D grid and flatten for 1D pixel indexing
  X, Z = np.meshgrid(x, z, indexing='ij')
  x_flat = X.ravel()
  z_flat = Z.ravel()
  P = len(x_flat)

  # TX path length (m), shape (P,)
  sin_t, cos_t = np.sin(theta), np.cos(theta)
  d_tx = x_flat * sin_t + z_flat * cos_t

  # RX distance for every (pixel, channel): (P, C)
  dx = x_flat[:, None] - elem_x[None, :]
  d_rx = np.sqrt(dx**2 + z_flat[:, None]**2)

  # Total delay and fractional sample index: (P, C)
  tau = (d_tx[:, None] + d_rx) / c - t0
  sample_idx = tau * fs

  # F-number apodization mask: |x - x_n| <= z / (2 F#)
  if bf.f_number is not None and bf.f_number > 0:
    half_aperture = z_flat[:, None] / (2.0 * bf.f_number)
    mask = np.abs(dx) <= half_aperture
  else:
    mask = np.ones((P, C), dtype=bool)

  # Valid time bounds mask
  valid_time = (sample_idx >= 0) & (sample_idx < S - 1)
  valid_mask = mask & valid_time

  # Extract valid indices to maintain sparsity
  pixel_idx, elem_idx = np.where(valid_mask)
  valid_s = sample_idx[pixel_idx, elem_idx]
  valid_tau = tau[pixel_idx, elem_idx]

  # Linear interpolation weights
  s_floor = np.floor(valid_s).astype(np.int64)
  weight_hi = valid_s - s_floor
  weight_lo = 1.0 - weight_hi

  # Phase rotator: exp(+j 2 pi f0 tau)
  phase = np.exp(1j * 2 * np.pi * f0 * valid_tau).astype(np.complex64)

  # Sparse matrix indices (row = pixel, col = channel * S + sample)
  row_indices = np.concatenate([pixel_idx, pixel_idx])
  
  col_indices = np.concatenate([
    elem_idx * S + s_floor,
    elem_idx * S + s_floor + 1
  ])

  # Sparse matrix data values
  data = np.concatenate([
    weight_lo * phase,
    weight_hi * phase
  ])

  # Construct the sparse DAS matrix M_das: shape (P, C*S)
  M_das = csr_matrix((data, (row_indices, col_indices)), shape=(P, C * S))

  # Reshape IQ data to (F, C*S) so each frame is a row
  iq_flat = iq.reshape(F, C * S)
  
  # Matrix multiplication: M_das (P, C*S) @ iq_flat.T (C*S, F) -> (P, F)
  # Transpose result to get (F, P)
  out_flat = M_das.dot(iq_flat.T).T 

  # Reshape back to the 2D spatial grid (F, nx, nz)
  out = out_flat.reshape(F, nx, nz)

  return out.astype(np.complex64)