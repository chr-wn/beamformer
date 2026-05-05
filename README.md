# beamformer

Fast 2D delay-and-sum (DAS) ultrasound beamformer for Apple Silicon.

Tests below are on Apple M4 Pro, 16-core GPU (should work on all M-series), on `data.h5`, best-of-30 runs after a 10-run warmup.

| config      | grid     | dt   | best  | median  | p10  | p90  | std  | fps    | μs/fr | GoP/s |
|-------------|----------|------|-------|---------|------|------|------|--------|-------|-------|
| lambda/2    | 181x346  | fp32 | 28.92m | 36.76m | 33.88m | 43.88m | 4.16m | 17287  | 57.8u | 145.07|
| lambda/2    | 181x346  | fp16 | 16.77m | 20.45m |   17.02m | 28.09m | 4.07m | 29820  | 33.5u | 250.25|
| lambda/4    | 363x691  | fp32 | 115.19m | 123.02m | 119.55m | 143.77m | 11.41m | 4341   | 230.4u | 145.90|
| lambda/4    | 363x691  | fp16 | 89.70m | 95.50m | 90.89m | 104.67m | 6.20m | 5574   | 179.4u | 187.36|
| lambda/8    | 725x1382 | fp32 | 317.53m | 367.62m | 328.14m | 445.48m | 55.89m | 1575   | 635.1u | 211.42|
| lambda/8    | 725x1382 | fp16 | 256.85m | 269.37m | 260.59m | 289.44m | 16.34m | 1947   | 513.7u | 261.36|

(for reference, baseline with numpy and A.F matrix speedup from the paper is ~131ms per frame, compared to row 1's 36.76ms median)

Metrics:
- grid: number of pixels in the grid (nx, nz)
- dt: data type (fp32 or fp16)
- best: best of 30 runs
- median: median of 30 runs
- p10: 10th percentile of 30 runs
- p90: 90th percentile of 30 runs
- std: standard deviation of 30 runs
- fps: frames per second
- μs/fr: microseconds per frame
- GoP/s: giga operations per second

## Install and run

Requires Python 3.10+, macOS 14+, Apple Silicon.

1. Install
```bash
git clone git@github.com:chr-wn/beamformer.git
cd beamformer
python -m venv .venv && source .venv/bin/activate # optional
pip install -e .
mkdir -p data/
```

2. Place `data.h5` into `data/`.

3. Benchmark

```bash
# benchmark with standardized warmup and 30 timed runs; appends results to history.jsonl
python bench/bench.py

# baseline numpy version
python bench/bench_numpy.py

# show the rolling history (one row per (run, config))
python bench/show_history.py --latest 5

# ablation study (helps find performance bottlenecks)
python bench/ablate.py

# render a single b-mode png
python scripts/make_bmode.py --frame 2 --ppwl 4 --dynamic-range 60

# render all frames to docs/frames/
python scripts/make_bmode_all.py --global-norm

# test das correctness
pytest -q
```

## Library API

```python
from beamformer import (
    load_h5, default_grid_for, BeamformParams, Grid, MlxBeamformer,
)

iq, acq = load_h5("data/data.h5")
gp  = default_grid_for(acq, iq.shape[2], iq.shape[1])
g   = Grid.from_params(gp)
bf  = BeamformParams(speed_of_sound=1540.0, f_number=1.5)

bm  = MlxBeamformer(acq=acq, grid=g, bf=bf, iq_dtype="fp16")
out = bm.run(iq) # (F, nx, nz) complex64 (mx.array)
```

## Code structure

```
src/beamformer/
    mlx_kernel.py       # GPU kernel & Python wrapper
    reference.py        # NumPy float64 ground-truth
    synthetic.py        # synthetic point-target IQ for PSF tests
    params.py           # AcqParams / GridParams / BeamformParams
    data.py             # HDF5 loader
    visualize.py        # B-mode log-compression / PNG renderer
bench/
    bench.py            # speed benchmark (writes results.json + history.jsonl)
    show_history.py     # render history.jsonl as a table
    history.jsonl       # append-only log of every benchmark run
scripts/
    make_bmode.py       # render a frame to PNG
    make_bmode_all.py   # render every frame, build a movie
    diagnose_synth.py   # PSF synthetic test
    show_raw.py         # raw IQ visualization
    sweep_c.py          # speed-of-sound sweep
tests/
    test_correctness.py # 9 tests (synthetic + real data)
docs/
    bmode.png psf.png … # rendered images
data/
    data.h5             # raw data
```

## How it works

The whole algorithm is one Metal compute kernel, implemented in [`src/beamformer/mlx_kernel.py`](src/beamformer/mlx_kernel.py). Canonical DAS, plus some optimizations:

- One thread per output pixel `(x,z)`, batched 4 frames at a time. Adjacent threads in a SIMD walk along the z-axis, so their IQ reads share the same cache line (reducing fetches).
- Threadgroup tile = (z=16, x=16, 1) = 256 threads = 8 SIMD-32 groups <- determined empirically
- For each pixel, we precompute the active receive aperture analytically from the F-number rule, then iterate only that channel range.
- For each (pixel, channel) we do the expensive operations (`sqrt`, `sincos`, `tau`) once, and then fan out to all 4 frames sharing those coefficients. (4 determined empirically. any bigger degrades due to register pressure)
- On the fp16 path, the IQ buffer is repacked at upload time to `(F_padded // 4, C, S, 8)`; one `half_v8` vector load then fetches all 4 frames' (re, im) pairs at one (c,s). This reduces LSU-instruction count per channel by ~half.
- Many other approaches ended up not helping, like lookup tables for sincos, reordering loops, batching by z, etc. (see my chain-of-thought google doc for some of those experiments).

## Credits:
- DAS algorithm following [Perrot et al. 2021](https://arxiv.org/abs/2007.11960). <- this was incredibly helpful!
- Gemini CLI (for help implementing Metal kernel quickly)

Enjoy :)