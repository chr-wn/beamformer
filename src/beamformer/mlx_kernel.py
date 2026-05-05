"""
DAS algorithm on a custom MLX kernel.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import mlx.core as mx
from .params import AcqParams, BeamformParams, Grid

FRAMES_PER_THREAD = 4
FRAME_PACK_FP16 = FRAMES_PER_THREAD
DEFAULT_THREADGROUP = (16, 16, 1)


_KERNEL_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
"""


def _make_kernel_body_fp32() -> str:
    """v2-style kernel for the fp32 IQ path: float2 narrow loads, no F-pack."""
    return r"""
    // Thread coords: (z, x, frame-batch). Z is innermost so adjacent SIMD
    // lanes walk along the depth axis -> contiguous IQ-sample stride per
    // channel, which keeps the L1 cache hot.
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
    const float half_inv_fn  = params[5];   // 1 / (2 F#); 0 disables apod
    const float two_pi_f0    = params[6];

    const float xs  = xs_arr[xi];
    const float zs  = zs_arr[zi];
    const float dtx = xs * sin_theta + zs * cos_theta;     // TX path length

    device const float2* iq2 = reinterpret_cast<device const float2*>(iq);
    const uint frame_stride = (uint)(NC * NS);

    float acc_re[FRAMES_PER_THREAD];
    float acc_im[FRAMES_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < FRAMES_PER_THREAD; ++k) {
        acc_re[k] = 0.0f;
        acc_im[k] = 0.0f;
    }

    // Active receive aperture from the F-number: solve |x_n - xs| <= z/(2F#)
    // for the channel index n analytically and iterate only that range.
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

    // Channel reduction loop. For each channel we do the heavy work
    // (sqrt, tau, sincos) ONCE, then fan out to FRAMES_PER_THREAD frames.
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
            const float2 lo = iq2[base];
            const float2 hi = iq2[base + 1];
            const float ii = fma(frac, hi.x - lo.x, lo.x);
            const float qi = fma(frac, hi.y - lo.y, lo.y);
            acc_re[k] = fma(ii, cs, fma(-qi, sn, acc_re[k]));
            acc_im[k] = fma(ii, sn, fma( qi, cs, acc_im[k]));
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


def _make_kernel_body_fp16_packed() -> str:
    """fp16 IQ path: F-packed layout + half8 vector load.

    The IQ buffer's outer dim is F_padded//FRAME_PACK_FP16; the innermost
    dim holds 4 frames of (re, im) interleaved at one (c, s) point. A
    single half8 load (16 bytes) fetches all 4 frames at one sample.

    Apple's compiler accepts ``ext_vector_type(8)`` on half (a clang
    extension) and emits a single 16-byte load. With 32 lanes per SIMD
    that's a 512-byte SIMD-load span = ~4 cache lines, the same total
    cache footprint as v2's narrow loads, but with 1/4 the LSU
    instructions per channel iteration.
    """
    return r"""
    typedef half __attribute__((ext_vector_type(8))) half_v8;

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

    // F-packed buffer: each half8 element = 4 frames' (re, im) at one
    // (c, s). The buffer's outer dim is F_padded // FRAMES_PER_THREAD.
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

    // Padded frames live in the buffer as zeros so out-of-batch contributions
    // are zero; we just don't write their outputs.
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

        // Two wide loads (16 bytes each) replace v2's eight half2 loads.
        // v_lo[2k..2k+1] = frame k's (re, im) at sample s0.
        const uint slot_lo = base_fb + (uint)c * (uint)NS + (uint)s0;
        const half_v8 v_lo = iq8[slot_lo];
        const half_v8 v_hi = iq8[slot_lo + 1u];

        #pragma unroll
        for (int k = 0; k < FRAMES_PER_THREAD; ++k) {
            if (k >= frames_here) break;
            const float lo_re = (float)v_lo[2 * k + 0];
            const float lo_im = (float)v_lo[2 * k + 1];
            const float hi_re = (float)v_hi[2 * k + 0];
            const float hi_im = (float)v_hi[2 * k + 1];
            const float ii = fma(frac, hi_re - lo_re, lo_re);
            const float qi = fma(frac, hi_im - lo_im, lo_im);
            acc_re[k] = fma(ii, cs, fma(-qi, sn, acc_re[k]));
            acc_im[k] = fma(ii, sn, fma( qi, cs, acc_im[k]));
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


def _build_kernel(NF: int, NC: int, NS: int, NX: int, NZ: int,
                  pitch_m: float, use_fp16: bool):
    """Build (or fetch from MLX's JIT cache) a kernel specialized to dims.

    fp16 uses the F-packed layout + half8 load; fp32 uses v2-style narrow
    loads. The pitch and dim constants are baked in so the compiler can
    fold aperture-range arithmetic and unroll the channel-reduction loop.
    """
    suffix = "fp16p" if use_fp16 else "fp32"
    pitch_lit = f"{pitch_m:.10g}f"
    header = (
        _KERNEL_HEADER
        + f"\n#define PITCH ({pitch_lit})\n"
        + f"#define FRAMES_PER_THREAD ({FRAMES_PER_THREAD})\n"
    )
    pitch_tag = (
        "P" + pitch_lit
        .replace('-', 'm').replace('.', 'd')
        .replace('+', 'p').replace('e', 'E')
    )
    body = _make_kernel_body_fp16_packed() if use_fp16 else _make_kernel_body_fp32()
    return mx.fast.metal_kernel(
        name=f"das_v4_{suffix}_F{NF}_C{NC}_S{NS}_X{NX}_Z{NZ}_{pitch_tag}",
        input_names=["iq", "xs_arr", "zs_arr", "elem_x", "params"],
        output_names=["out"],
        header=header,
        source=body,
        ensure_row_contiguous=True,
    )


def pack_iq_fp16(iq_np: np.ndarray) -> np.ndarray:
    """Repack (F, C, S) complex64 IQ into the fp16 path's expected layout:
    (F_padded // FRAME_PACK_FP16, C, S, FRAME_PACK_FP16 * 2) of float16.
    The innermost dim holds 4 frames' interleaved (re, im) pairs at one
    (channel, sample) point. F is padded to a multiple of FRAME_PACK_FP16
    with zeros (the kernel's frames_here check prevents out-of-batch
    output writes).
    """
    if iq_np.dtype != np.complex64:
        iq_np = iq_np.astype(np.complex64)
    F, C, S = iq_np.shape
    P = FRAME_PACK_FP16
    F_padded = ((F + P - 1) // P) * P
    if F_padded != F:
        pad = np.zeros((F_padded - F, C, S), dtype=np.complex64)
        iq_np = np.concatenate([iq_np, pad], axis=0)
    iq_real = iq_np.view(np.float32).reshape(F_padded, C, S, 2)
    iq_blocks = (iq_real
                 .reshape(F_padded // P, P, C, S, 2)
                 .transpose(0, 2, 3, 1, 4)
                 .reshape(F_padded // P, C, S, P * 2))
    return np.ascontiguousarray(iq_blocks).astype(np.float16)


@dataclass
class MlxBeamformer:
    """Reusable, JIT-cached beamformer.

    Build once for a given (acq, grid, bf, iq_dtype) configuration; call
    ``run`` repeatedly. The Metal kernel is compiled the first time and
    cached by MLX. For the fp16 path, the F-packed IQ buffer is also
    cached by input identity so back-to-back ``run(same_iq)`` calls don't
    repack.
    """

    acq: AcqParams
    grid: Grid
    bf: BeamformParams = BeamformParams()
    iq_dtype: str = "fp32"  # "fp32" or "fp16"
    threadgroup: tuple[int, int, int] = DEFAULT_THREADGROUP

    _xs: mx.array = None
    _zs: mx.array = None
    _elem_x: mx.array = None
    _params: mx.array = None
    _kernel_cache: dict = field(default_factory=dict)
    _packed_cache: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.iq_dtype not in ("fp32", "fp16"):
            raise ValueError(f"iq_dtype must be 'fp32' or 'fp16', got {self.iq_dtype!r}")
        nx, nz = int(self.grid.x.size), int(self.grid.z.size)
        if nx <= 0 or nz <= 0:
            raise ValueError("grid must have nx > 0, nz > 0")
        self._nx, self._nz = nx, nz

        self._xs = mx.array(self.grid.x.astype(np.float32))
        self._zs = mx.array(self.grid.z.astype(np.float32))

        nc = self.acq.num_channels
        if nc is None:
            raise ValueError("AcqParams.num_channels must be set")
        self._nc = int(nc)
        self._elem_x = mx.array(self.acq.element_x(nc).astype(np.float32))

        bf = self.bf
        half_inv_fn = (1.0 / (2.0 * bf.f_number)) if (bf.f_number and bf.f_number > 0) else 0.0
        params = np.array([
            float(self.acq.sampling_rate_hz),
            float(1.0 / bf.speed_of_sound),
            float(bf.t0_s),
            float(math.sin(self.acq.tx_angle_rad)),
            float(math.cos(self.acq.tx_angle_rad)),
            float(half_inv_fn),
            float(2.0 * math.pi * self.acq.tx_freq_hz),
            0.0,
        ], dtype=np.float32)
        self._params = mx.array(params)

    def _prep_iq_fp32(self, iq):
        """Return mx.array view of (F, C, S*2) float32. Zero-copy when possible."""
        if isinstance(iq, np.ndarray):
            if iq.dtype != np.complex64:
                iq = iq.astype(np.complex64)
            iq = np.ascontiguousarray(iq)
            F, C, S = iq.shape
            return mx.array(iq.view(np.float32)), F, C, S
        if iq.dtype == mx.complex64:
            F, C, S = iq.shape
            return mx.view(iq, dtype=mx.float32), F, C, S
        if iq.dtype == mx.float32 and iq.ndim == 3 and iq.shape[2] % 2 == 0:
            F, C, S2 = iq.shape
            return iq, F, C, S2 // 2
        raise TypeError(f"unsupported iq dtype {iq.dtype} for fp32 path")

    def _prep_iq_fp16(self, iq):
        """Return mx.array of shape (F_pad//4, C, S, 8) fp16, plus original (F, C, S).

        Accepts either a complex64 buffer (numpy or mx.array) OR a
        pre-converted fp16 real-valued buffer of shape (F, C, S*2) --
        the latter is what ``bench/bench.py`` passes after its
        ``upload_iq`` call. Cached by id(iq) so back-to-back
        ``run(same_iq)`` doesn't repack.
        """
        cache_key = id(iq)
        cached = self._packed_cache.get(cache_key)
        if cached is not None:
            return cached

        # Convert inputs to a numpy fp16 (F, C, S, 2) view that we can
        # transpose-repack without re-splitting complex pairs.
        if isinstance(iq, mx.array):
            if iq.dtype == mx.complex64:
                arr = np.asarray(iq)
                F, C, S = arr.shape
                iq_real_h = (np.stack([arr.real, arr.imag], axis=-1)
                               .astype(np.float16))                 # (F, C, S, 2)
            elif iq.dtype == mx.float16 and iq.ndim == 3 and iq.shape[2] % 2 == 0:
                iq_real_h = np.asarray(iq).reshape(iq.shape[0], iq.shape[1],
                                                   iq.shape[2] // 2, 2)
                F, C, S = iq_real_h.shape[:3]
            else:
                raise TypeError(
                    f"fp16 path needs complex64 mx.array, fp16 (F,C,S*2) mx.array, "
                    f"or complex64 numpy input; got dtype {iq.dtype} shape {iq.shape}"
                )
        else:
            arr = iq
            if arr.dtype != np.complex64:
                arr = arr.astype(np.complex64)
            F, C, S = arr.shape
            iq_real_h = (np.stack([arr.real, arr.imag], axis=-1)
                           .astype(np.float16))                     # (F, C, S, 2)

        # Pad F up to a multiple of FRAME_PACK_FP16 with zeros; padded
        # frames contribute zero in the channel sum (the kernel's
        # frames_here check still gates output writes).
        P = FRAME_PACK_FP16
        F_pad = ((F + P - 1) // P) * P
        if F_pad != F:
            pad = np.zeros((F_pad - F, C, S, 2), dtype=np.float16)
            iq_real_h = np.concatenate([iq_real_h, pad], axis=0)
        iq_blocks = (iq_real_h
                       .reshape(F_pad // P, P, C, S, 2)
                       .transpose(0, 2, 3, 1, 4)
                       .reshape(F_pad // P, C, S, P * 2))
        iq_packed_np = np.ascontiguousarray(iq_blocks)
        iq_packed_mx = mx.array(iq_packed_np)
        mx.eval(iq_packed_mx); mx.synchronize()
        result = (iq_packed_mx, F, C, S)
        self._packed_cache[cache_key] = result
        return result

    def run(self, iq: np.ndarray | mx.array) -> mx.array:
        """Beamform a stack of IQ frames; returns complex64 mx.array (F, NX, NZ)."""
        use_fp16 = (self.iq_dtype == "fp16")
        if use_fp16:
            iq_mx, F, C, S = self._prep_iq_fp16(iq)
        else:
            iq_mx, F, C, S = self._prep_iq_fp32(iq)
        if C != self._nc:
            raise ValueError(f"channel count mismatch {C} != {self._nc}")

        nx, nz = self._nx, self._nz
        kkey = (F, C, S, nx, nz, use_fp16)
        kernel = self._kernel_cache.get(kkey)
        if kernel is None:
            kernel = _build_kernel(F, C, S, nx, nz,
                                   pitch_m=self.acq.pitch_m,
                                   use_fp16=use_fp16)
            self._kernel_cache[kkey] = kernel

        tg = self.threadgroup
        f_blocks = (F + FRAMES_PER_THREAD - 1) // FRAMES_PER_THREAD
        gx = ((nz + tg[0] - 1) // tg[0]) * tg[0]
        gy = ((nx + tg[1] - 1) // tg[1]) * tg[1]
        gz = ((f_blocks + tg[2] - 1) // tg[2]) * tg[2]

        outs = kernel(
            inputs=[iq_mx, self._xs, self._zs, self._elem_x, self._params],
            template=[("NF", F), ("NC", C), ("NS", S), ("NX", nx), ("NZ", nz)],
            grid=(gx, gy, gz),
            threadgroup=tg,
            output_shapes=[(F, nx, nz, 2)],
            output_dtypes=[mx.float32],
        )
        out_real = outs[0]
        return mx.view(out_real, dtype=mx.complex64).reshape(F, nx, nz)


def beamform_mlx(
    iq: np.ndarray,
    acq: AcqParams,
    grid: Grid,
    bf: BeamformParams = BeamformParams(),
    iq_dtype: str = "fp32",
) -> np.ndarray:
    bm = MlxBeamformer(acq=acq, grid=grid, bf=bf, iq_dtype=iq_dtype)
    out_mx = bm.run(iq)
    mx.eval(out_mx)
    return np.asarray(out_mx)
