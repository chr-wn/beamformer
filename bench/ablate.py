"""
Ablation to find bottleneck in the DAS kernel.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beamformer.data import load_h5
from beamformer.params import BeamformParams, Grid, default_grid_for
from beamformer.mlx_kernel import FRAMES_PER_THREAD, FRAME_PACK_FP16, DEFAULT_THREADGROUP


_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
"""

_HEADER_FP16_V4 = _HEADER + r"""
typedef half __attribute__((ext_vector_type(8))) half_v8;
"""


def _make_body_fp32(mac_factor: int, loads: bool) -> str:
    """fp32 ablation body: v2-style float2 narrow loads."""
    if loads:
        load_block = (
            "const float2 lo = iq2[base];\n"
            "            const float2 hi = iq2[base + 1];\n"
            "            const float ii = fma(frac, hi.x - lo.x, lo.x);\n"
            "            const float qi = fma(frac, hi.y - lo.y, lo.y);"
        )
    else:
        load_block = (
            "const float ii = cs + frac;\n"
            "            const float qi = sn + frac;"
        )

    mac_one = (
        "acc_re[k] = fma(ii, cs, fma(-qi, sn, acc_re[k]));\n"
        "            acc_im[k] = fma(ii, sn, fma( qi, cs, acc_im[k]));"
    )

    if mac_factor == 0:
        mac_block = "acc_re[k] += ii * 0.0f + qi * 0.0f;" if loads else "acc_re[k] += 0.0f;"
    else:
        mac_block = "\n            ".join([mac_one] * mac_factor)

    return r"""
    uint zi = thread_position_in_grid.x;
    uint xi = thread_position_in_grid.y;
    uint fb = thread_position_in_grid.z;
    if (zi >= (uint)NZ || xi >= (uint)NX) return;

    const uint f0_idx = fb * (uint)FRAMES_PER_THREAD;
    if (f0_idx >= (uint)NF) return;

    const float fs           = params[0];
    const float inv_c        = params[1];
    const float t0_s         = params[2];
    const float sin_theta    = params[3];
    const float cos_theta    = params[4];
    const float half_inv_fn  = params[5];
    const float two_pi_f0    = params[6];

    const float xs  = xs_arr[xi];
    const float zs  = zs_arr[zi];
    const float dtx = xs * sin_theta + zs * cos_theta;

    device const float2* iq2 = reinterpret_cast<device const float2*>(iq);
    const uint frame_stride = (uint)(NC * NS);

    float acc_re[FRAMES_PER_THREAD];
    float acc_im[FRAMES_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < FRAMES_PER_THREAD; ++k) {
        acc_re[k] = 0.0f;
        acc_im[k] = 0.0f;
    }

    int c_lo, c_hi;
    if (half_inv_fn > 0.0f) {
        const float half_apert = zs * half_inv_fn;
        const float center_off = (float)NC * 0.5f - 0.5f;
        const float inv_pitch  = 1.0f / PITCH;
        const float lo_f = (xs - half_apert) * inv_pitch + center_off;
        const float hi_f = (xs + half_apert) * inv_pitch + center_off;
        c_lo = max(0, (int)ceil(lo_f));
        c_hi = min(NC - 1, (int)floor(hi_f));
    } else {
        c_lo = 0;
        c_hi = NC - 1;
    }

    const int frames_here = min((int)FRAMES_PER_THREAD, NF - (int)f0_idx);

    for (int c = c_lo; c <= c_hi; ++c) {
        const float xn  = elem_x[c];
        const float dxn = xs - xn;

        const float drx     = fast::sqrt(fma(dxn, dxn, zs * zs));
        const float tau     = (dtx + drx) * inv_c - t0_s;
        const float s_idx   = tau * fs;
        const float s_floor = floor(s_idx);
        const int   s0      = int(s_floor);
        if (s0 < 0 || s0 >= NS - 1) continue;
        const float frac    = s_idx - s_floor;

        float cs;
        const float sn = fast::sincos(two_pi_f0 * tau, cs);

        const uint base_c = (uint)c * (uint)NS + (uint)s0;
        #pragma unroll
        for (int k = 0; k < FRAMES_PER_THREAD; ++k) {
            if (k >= frames_here) break;
            const uint base = (f0_idx + (uint)k) * frame_stride + base_c;
            """ + load_block + r"""
            """ + mac_block + r"""
        }
    }

    #pragma unroll
    for (int k = 0; k < FRAMES_PER_THREAD; ++k) {
        if (k >= frames_here) break;
        const uint out_idx = ((f0_idx + (uint)k) * (uint)NX + xi)
                             * (uint)NZ + zi;
        out[2u * out_idx + 0u] = acc_re[k];
        out[2u * out_idx + 1u] = acc_im[k];
    }
"""


def _make_body_fp16_v4(mac_factor: int, loads: bool) -> str:
    """fp16 ablation body: v4 packed half_v8 loads (matches production kernel).

    The IQ buffer layout is (F_pad//4, C, S, 8) of float16.  Two half_v8
    loads outside the per-frame loop replace v2's eight narrow half2 loads
    inside the loop.
    """
    if loads:
        load_block = (
            "const uint slot_lo = base_fb + (uint)c * (uint)NS + (uint)s0;\n"
            "        const half_v8 v_lo = iq8[slot_lo];\n"
            "        const half_v8 v_hi = iq8[slot_lo + 1u];"
        )
        extract_block = (
            "const float lo_re = (float)v_lo[2 * k + 0];\n"
            "            const float lo_im = (float)v_lo[2 * k + 1];\n"
            "            const float hi_re = (float)v_hi[2 * k + 0];\n"
            "            const float hi_im = (float)v_hi[2 * k + 1];\n"
            "            const float ii = fma(frac, hi_re - lo_re, lo_re);\n"
            "            const float qi = fma(frac, hi_im - lo_im, lo_im);"
        )
    else:
        load_block = ""
        extract_block = (
            "const float ii = cs + frac;\n"
            "            const float qi = sn + frac;"
        )

    mac_one = (
        "acc_re[k] = fma(ii, cs, fma(-qi, sn, acc_re[k]));\n"
        "            acc_im[k] = fma(ii, sn, fma( qi, cs, acc_im[k]));"
    )

    if mac_factor == 0:
        mac_block = "acc_re[k] += ii * 0.0f + qi * 0.0f;" if loads else "acc_re[k] += 0.0f;"
    else:
        mac_block = "\n            ".join([mac_one] * mac_factor)

    return r"""
    uint zi = thread_position_in_grid.x;
    uint xi = thread_position_in_grid.y;
    uint fb = thread_position_in_grid.z;
    if (zi >= (uint)NZ || xi >= (uint)NX) return;

    const uint f0_idx = fb * (uint)FRAMES_PER_THREAD;
    if (f0_idx >= (uint)NF) return;

    const float fs           = params[0];
    const float inv_c        = params[1];
    const float t0_s         = params[2];
    const float sin_theta    = params[3];
    const float cos_theta    = params[4];
    const float half_inv_fn  = params[5];
    const float two_pi_f0    = params[6];

    const float xs  = xs_arr[xi];
    const float zs  = zs_arr[zi];
    const float dtx = xs * sin_theta + zs * cos_theta;

    device const half_v8* iq8 = reinterpret_cast<device const half_v8*>(iq);
    const uint slots_per_fb = (uint)NC * (uint)NS;
    const uint base_fb      = fb * slots_per_fb;

    float acc_re[FRAMES_PER_THREAD];
    float acc_im[FRAMES_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < FRAMES_PER_THREAD; ++k) {
        acc_re[k] = 0.0f;
        acc_im[k] = 0.0f;
    }

    int c_lo, c_hi;
    if (half_inv_fn > 0.0f) {
        const float half_apert = zs * half_inv_fn;
        const float center_off = (float)NC * 0.5f - 0.5f;
        const float inv_pitch  = 1.0f / PITCH;
        const float lo_f = (xs - half_apert) * inv_pitch + center_off;
        const float hi_f = (xs + half_apert) * inv_pitch + center_off;
        c_lo = max(0, (int)ceil(lo_f));
        c_hi = min(NC - 1, (int)floor(hi_f));
    } else {
        c_lo = 0;
        c_hi = NC - 1;
    }

    const int frames_here = min((int)FRAMES_PER_THREAD, NF - (int)f0_idx);

    for (int c = c_lo; c <= c_hi; ++c) {
        const float xn  = elem_x[c];
        const float dxn = xs - xn;

        const float drx     = fast::sqrt(fma(dxn, dxn, zs * zs));
        const float tau     = (dtx + drx) * inv_c - t0_s;
        const float s_idx   = tau * fs;
        const float s_floor = floor(s_idx);
        const int   s0      = int(s_floor);
        if (s0 < 0 || s0 >= NS - 1) continue;
        const float frac    = s_idx - s_floor;

        float cs;
        const float sn = fast::sincos(two_pi_f0 * tau, cs);

        """ + load_block + r"""

        #pragma unroll
        for (int k = 0; k < FRAMES_PER_THREAD; ++k) {
            if (k >= frames_here) break;
            """ + extract_block + r"""
            """ + mac_block + r"""
        }
    }

    #pragma unroll
    for (int k = 0; k < FRAMES_PER_THREAD; ++k) {
        if (k >= frames_here) break;
        const uint out_idx = ((f0_idx + (uint)k) * (uint)NX + xi)
                             * (uint)NZ + zi;
        out[2u * out_idx + 0u] = acc_re[k];
        out[2u * out_idx + 1u] = acc_im[k];
    }
"""


def _build(NF, NC, NS, NX, NZ, pitch_m, use_fp16, mac_factor, loads, tag):
    suffix = f"{'fp16v4' if use_fp16 else 'fp32'}_{tag}"
    pitch_lit = f"{pitch_m:.10g}f"
    base_header = _HEADER_FP16_V4 if use_fp16 else _HEADER
    header = (
        base_header
        + f"\n#define PITCH ({pitch_lit})\n"
        + f"#define FRAMES_PER_THREAD ({FRAMES_PER_THREAD})\n"
    )
    pitch_tag = (
        "P" + pitch_lit
        .replace('-', 'm').replace('.', 'd')
        .replace('+', 'p').replace('e', 'E')
    )
    if use_fp16:
        source = _make_body_fp16_v4(mac_factor, loads)
    else:
        source = _make_body_fp32(mac_factor, loads)
    return mx.fast.metal_kernel(
        name=f"ablate_{suffix}_F{NF}_C{NC}_S{NS}_X{NX}_Z{NZ}_{pitch_tag}",
        input_names=["iq", "xs_arr", "zs_arr", "elem_x", "params"],
        output_names=["out"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


@dataclass
class _Setup:
    F: int; C: int; S: int; NX: int; NZ: int; pitch_m: float
    iq_mx: mx.array; xs: mx.array; zs: mx.array
    elem_x: mx.array; params: mx.array; use_fp16: bool


def _pack_iq_fp16(iq_complex: np.ndarray) -> np.ndarray:
    """Pack complex64 IQ into (F_pad//4, C, S, 8) float16 — same as production."""
    F, C, S = iq_complex.shape
    P = FRAME_PACK_FP16  # = 4
    F_pad = ((F + P - 1) // P) * P
    iq_real_h = np.stack([iq_complex.real, iq_complex.imag], axis=-1).astype(np.float16)
    if F_pad != F:
        pad = np.zeros((F_pad - F, C, S, 2), dtype=np.float16)
        iq_real_h = np.concatenate([iq_real_h, pad], axis=0)
    iq_blocks = (iq_real_h
                   .reshape(F_pad // P, P, C, S, 2)
                   .transpose(0, 2, 3, 1, 4)
                   .reshape(F_pad // P, C, S, P * 2))
    return np.ascontiguousarray(iq_blocks)


def _setup(args, dtype: str) -> _Setup:
    iq, acq = load_h5(args.data)
    F, C, S = iq.shape
    use_fp16 = (dtype == "fp16")
    if use_fp16:
        iq_packed = _pack_iq_fp16(iq)
        iq_mx = mx.array(iq_packed)
    else:
        iq_real = np.ascontiguousarray(iq).view(np.float32)
        iq_mx = mx.array(iq_real)
    mx.eval(iq_mx); mx.synchronize()
    gp = default_grid_for(acq, S, C, speed_of_sound=args.c)
    g = Grid.from_params(gp)
    NX, NZ = int(g.x.size), int(g.z.size)
    xs = mx.array(g.x.astype(np.float32))
    zs = mx.array(g.z.astype(np.float32))
    elem_x = mx.array(acq.element_x(C).astype(np.float32))
    import math
    bf = BeamformParams(speed_of_sound=args.c, f_number=args.fnumber)
    half_inv_fn = (1.0 / (2.0 * bf.f_number)) if (bf.f_number and bf.f_number > 0) else 0.0
    params = mx.array(np.array([
        float(acq.sampling_rate_hz),
        float(1.0 / bf.speed_of_sound),
        float(bf.t0_s),
        float(math.sin(acq.tx_angle_rad)),
        float(math.cos(acq.tx_angle_rad)),
        float(half_inv_fn),
        float(2.0 * math.pi * acq.tx_freq_hz),
        0.0,
    ], dtype=np.float32))
    return _Setup(F, C, S, NX, NZ, acq.pitch_m, iq_mx, xs, zs, elem_x, params, use_fp16)


def _time(kernel, st: _Setup, runs: int, warmup: int) -> dict:
    tg = DEFAULT_THREADGROUP
    f_blocks = (st.F + FRAMES_PER_THREAD - 1) // FRAMES_PER_THREAD
    gx = ((st.NZ + tg[0] - 1) // tg[0]) * tg[0]
    gy = ((st.NX + tg[1] - 1) // tg[1]) * tg[1]
    gz = ((f_blocks + tg[2] - 1) // tg[2]) * tg[2]
    template = [("NF", st.F), ("NC", st.C), ("NS", st.S),
                ("NX", st.NX), ("NZ", st.NZ)]
    out_shape = [(st.F, st.NX, st.NZ, 2)]

    def _once():
        outs = kernel(
            inputs=[st.iq_mx, st.xs, st.zs, st.elem_x, st.params],
            template=template, grid=(gx, gy, gz), threadgroup=tg,
            output_shapes=out_shape, output_dtypes=[mx.float32],
        )
        mx.eval(outs[0]); mx.synchronize()

    for _ in range(warmup): _once()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _once()
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts)
    return dict(best=float(a.min()), median=float(np.median(a)))


# (tag, mac_factor, loads)
SPECS = [
    ("full",         1, True),
    ("mac_0",        0, True),
    ("mac_2x",       2, True),
    ("mac_4x",       4, True),
    ("no_load",      1, False),
    ("no_load_mac0", 0, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "data.h5"))
    ap.add_argument("--c", type=float, default=1540.0)
    ap.add_argument("--fnumber", type=float, default=1.5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--dtype", default="both", choices=["fp32", "fp16", "both"])
    args = ap.parse_args()

    dtypes = ["fp32", "fp16"] if args.dtype == "both" else [args.dtype]

    for dt in dtypes:
        st = _setup(args, dt)
        print(f"\n=== {dt}  grid {st.NX}x{st.NZ}  F={st.F} C={st.C} S={st.S} ===")
        baseline = None
        rows = []
        for tag, mac, loads in SPECS:
            k = _build(st.F, st.C, st.S, st.NX, st.NZ,
                       st.pitch_m, st.use_fp16, mac, loads, tag)
            t = _time(k, st, args.runs, args.warmup)
            if tag == "full":
                baseline = t["best"]
            d = (t["best"] - baseline) * 1e3
            pct = 100.0 * (t["best"] / baseline - 1)
            rows.append((tag, mac, loads, t["best"]*1e3, t["median"]*1e3, d, pct))

        hdr = (f"{'tag':14s}  {'mac':>3s}  {'load':>4s}  "
               f"{'best (ms)':>10s}  {'median':>9s}  {'Δ ms':>7s}  {'%':>6s}")
        print(hdr); print("-"*len(hdr))
        for tag, mac, loads, b, m, d, pct in rows:
            print(f"{tag:14s}  {mac:>3d}  {str(loads):>4s}  "
                  f"{b:9.2f}  {m:8.2f}  {d:6.2f}  {pct:5.1f}")


if __name__ == "__main__":
    main()
