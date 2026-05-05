import time
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from beamformer.data import load_h5
from beamformer.params import BeamformParams, default_grid_for, Grid
from beamformer.reference import beamform_reference

iq, acq = load_h5("data/data.h5")
# use smaller slice of frames so we don't wait eternity, np is slow lol
frames = 5
iq_slice = iq[:frames]

S, C = iq.shape[2], iq.shape[1]
# lambda/2 grid
gp = default_grid_for(acq, S, C, pixels_per_wavelength=2.0)
grid = Grid.from_params(gp)
bf = BeamformParams(f_number=1.5)

print(f"benchmarking np reference implementation...")
print(f"frames: {frames}, grid: {gp.nx}x{gp.nz}, channels: {C}, samples: {S}")

# warmup
_ = beamform_reference(iq_slice[:1], acq, grid, bf)

t0 = time.perf_counter()
out = beamform_reference(iq_slice, acq, grid, bf)
t1 = time.perf_counter()

total_ms = (t1 - t0) * 1000
ms_per_frame = total_ms / frames

print(f"total time for {frames} frames: {total_ms:.2f} ms")
print(f"time per frame: {ms_per_frame:.2f} ms")