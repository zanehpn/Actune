"""Optimized packed-INT4 / dynamic-INT8 Triton Linear.

Drop-in replacement for ``w4a8_triton.W4A8TritonLinear`` that is **bit-exact**
with it: verified equal on 3072 configurations spanning six layer shapes, four
seeds, sixteen row counts from 1 to 1177, with and without bias, and over
normal / 1e-4 / 1e3 / zero-row activation regimes.

Bit-exactness is what makes the swap free: the dot product accumulates in INT32,
so retiling and relayout cannot change its value, and the quantization formulas
and epilogue operation order are reproduced exactly.  Traces already collected
with the original kernel stay valid; only latency changes.

On SM120, measured BF16 M=1 Llama shapes additionally use exact INT32 split-K
with a final integer reduction. Eager calls replay a module-local CUDA Graph;
external graph capture uses direct launches. Weight packing is unchanged.

Four changes against the original kernel:

1. ``packed_weight`` is stored K-major as ``[in_features // 2, out_features]``
   with the nibble pair ``(k, k + in_features // 2)`` sharing a byte.  The
   original layout ``[out_features, in_features // 2]`` made the weight tile
   load stride by ``in_features // 2`` bytes along ``n`` -- one useful byte per
   32-byte sector -- and made adjacent ``k`` re-read the same byte.  The new
   layout is contiguous along ``n`` and reads each byte once.  This is the
   dominant win.
2. ``rows`` is a runtime argument rather than ``tl.constexpr``, so a varying
   token count no longer forces a fresh Triton specialization per shape.
3. Wider tiles (BLOCK_N=128, BLOCK_J=64) with ``num_stages=3`` software
   pipelining, replacing BLOCK_N=64 / BLOCK_K=32 with no pipelining.
4. Per-row dynamic INT8 activation quantization is fused into one kernel
   instead of six eager elementwise/reduction launches.

Reproducing torch bit-for-bit in step 4 needs two non-obvious details:
``torch.round`` is round-half-to-even (``libdevice.rint``, not
``libdevice.round``), and torch lowers CUDA tensor/scalar division to a
multiply by the fp32 reciprocal, which sits one ULP from a correctly rounded
divide on roughly 10% of rows.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch
import triton
import triton.language as tl

from .resident_a8_triton import resident_a8_eligible, launch_resident_a8

from .decode_w4a8_triton import (
    decode_eligible, decode_linear, graph_decode_linear, _quantize_decode_kernel, _DECODE_STATE,
)


@triton.jit
def _w4a8_gemm_kernel(
    activation_ptr,
    packed_weight_ptr,
    activation_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    output_ptr,
    rows,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offsets_m < rows
    mask_n = offsets_n < out_features
    half = in_features // 2
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for start_j in range(0, half, BLOCK_J):
        offsets_j = start_j + tl.arange(0, BLOCK_J)
        packed = tl.load(
            packed_weight_ptr + offsets_j[:, None] * out_features + offsets_n[None, :],
            mask=mask_n[None, :],
            other=0,
        )
        # (v ^ 8) - 8 sign-extends a 4-bit two's-complement nibble.
        weight_low = (((packed & 15).to(tl.int8)) ^ 8) - 8
        weight_high = ((((packed >> 4) & 15).to(tl.int8)) ^ 8) - 8
        activation_low = tl.load(
            activation_ptr + offsets_m[:, None] * in_features + offsets_j[None, :],
            mask=mask_m[:, None],
            other=0,
        )
        activation_high = tl.load(
            activation_ptr + offsets_m[:, None] * in_features + (offsets_j[None, :] + half),
            mask=mask_m[:, None],
            other=0,
        )
        accumulator += tl.dot(activation_low, weight_low, out_dtype=tl.int32)
        accumulator += tl.dot(activation_high, weight_high, out_dtype=tl.int32)

    output = accumulator.to(tl.float32)
    output *= tl.load(activation_scale_ptr + offsets_m, mask=mask_m, other=0.0)[:, None]
    output *= tl.load(weight_scale_ptr + offsets_n, mask=mask_n, other=0.0)[None, :]
    if has_bias:
        output += tl.load(bias_ptr + offsets_n, mask=mask_n, other=0.0)[None, :]
    tl.store(
        output_ptr + offsets_m[:, None] * out_features + offsets_n[None, :],
        output,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _w4a8_gemv_kernel(
    activation_ptr,
    packed_weight_ptr,
    activation_scale_ptr,
    weight_scale_ptr,
    bias_ptr,
    output_ptr,
    rows,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < out_features
    half = in_features // 2
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for start_j in range(0, half, BLOCK_J):
        offsets_j = start_j + tl.arange(0, BLOCK_J)
        packed = tl.load(
            packed_weight_ptr + offsets_j[:, None] * out_features + offsets_n[None, :],
            mask=mask_n[None, :],
            other=0,
        )
        weight_low = ((((packed & 15).to(tl.int8)) ^ 8) - 8).to(tl.int32)
        weight_high = (((((packed >> 4) & 15).to(tl.int8)) ^ 8) - 8).to(tl.int32)
        activation_low = tl.load(activation_ptr + row * in_features + offsets_j).to(tl.int32)
        activation_high = tl.load(activation_ptr + row * in_features + offsets_j + half).to(tl.int32)
        accumulator += tl.sum(activation_low[:, None] * weight_low, axis=0)
        accumulator += tl.sum(activation_high[:, None] * weight_high, axis=0)

    output = accumulator.to(tl.float32)
    output *= tl.load(activation_scale_ptr + row)
    output *= tl.load(weight_scale_ptr + offsets_n, mask=mask_n, other=0.0)
    if has_bias:
        output += tl.load(bias_ptr + offsets_n, mask=mask_n, other=0.0)
    tl.store(output_ptr + row * out_features + offsets_n, output, mask=mask_n)


@triton.jit
def _quantize_rows_kernel(
    source_ptr,
    activation_ptr,
    scale_ptr,
    in_features: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused per-row dynamic INT8 quantization.

    Reproduces ``amax`` (order-independent, hence exact) and the fp32
    ``divide / round-half-to-even / clamp`` chain of the eager implementation.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    base = source_ptr + row * in_features

    amax = tl.zeros((), dtype=tl.float32)
    for start in range(0, in_features, BLOCK_K):
        mask = start + offsets < in_features
        values = tl.load(base + start + offsets, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(values)))
    # torch lowers CUDA tensor/scalar division to a multiply by the fp32
    # reciprocal, which is one ULP away from a correctly rounded divide on
    # roughly 10% of rows.  Reproduce that, not IEEE division.
    floor = tl.full((), 1e-8, tl.float32)
    reciprocal = tl.full((), 1.0 / 127.0, tl.float32)
    scale = tl.maximum(amax, floor) * reciprocal
    tl.store(scale_ptr + row, scale)

    for start in range(0, in_features, BLOCK_K):
        mask = start + offsets < in_features
        values = tl.load(base + start + offsets, mask=mask, other=0.0).to(tl.float32)
        # div_rn is the correctly-rounded division torch performs; tl.fdiv's
        # ieee_rounding flag is not honoured on this backend and loses ties
        # such as 63.49999875 -> 63.5.  rint matches torch.round's
        # round-half-to-even.  Both are required for bit-exact parity.
        scaled = tl.extra.libdevice.rint(tl.extra.libdevice.div_rn(values, scale))
        scaled = tl.minimum(tl.maximum(scaled, -128.0), 127.0)
        tl.store(activation_ptr + row * in_features + start + offsets,
                 scaled.to(tl.int8), mask=mask)


class _SharedA8Entry:
    """One task-local dynamic-A8 result shared by sibling projections."""

    __slots__ = (
        "source",
        "in_features",
        "activation",
        "scale",
    )

    def __init__(
        self,
        source: torch.Tensor,
        in_features: int,
        activation: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self.source = source
        self.in_features = in_features
        self.activation = activation
        self.scale = scale


class SharedA8CacheStats:
    """Mutable statistics and one-entry workspace for a policy forward."""

    __slots__ = ("enabled", "entry", "a4_entry", "hits", "misses")

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.entry: _SharedA8Entry | None = None
        self.a4_entry = None
        self.hits = 0
        self.misses = 0


_SHARED_A8_STATE: ContextVar[SharedA8CacheStats | None] = ContextVar(
    "vla_shared_dynamic_a8_state", default=None
)


def clear_shared_a8_cache() -> None:
    """Drop the task-local sibling-projection activation cache."""

    state = _SHARED_A8_STATE.get()
    if state is None:
        _SHARED_A8_STATE.set(SharedA8CacheStats())
    else:
        state.entry = None
        state.a4_entry = None


@contextmanager
def shared_a8_cache_scope(
    *, enabled: bool = True
) -> Iterator[SharedA8CacheStats]:
    """Bound cache lifetime to one policy forward and release its workspace."""

    state = SharedA8CacheStats(enabled=enabled)
    token = _SHARED_A8_STATE.set(state)
    try:
        yield state
    finally:
        _SHARED_A8_STATE.reset(token)


def _quantize_rows_shared(
    source: torch.Tensor,
    flat: torch.Tensor,
    *,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize once and reuse for adjacent Linear calls on the same tensor.

    Llama Q/K/V projections receive the identical ``hidden_states`` object,
    as do Gate/Up.  Dynamic per-row A8 depends only on that input and K, not on
    the output width or the weight backend, so the packed activation and scale
    are bit-for-bit reusable across W4, W8, and nested signed-R4 GEMMs.
    """

    if not source.is_cuda or flat.shape[-1] != in_features:
        raise ValueError("shared A8 quantization requires a CUDA input with matching K")
    state = _SHARED_A8_STATE.get()
    if state is None:
        state = SharedA8CacheStats()
        _SHARED_A8_STATE.set(state)
    entry = state.entry
    if (
        state.enabled
        and entry is not None
        and entry.source is source
        and entry.in_features == in_features
        and entry.activation.shape == flat.shape
    ):
        state.hits += 1
        return entry.activation, entry.scale

    activation = torch.empty(flat.shape, device=flat.device, dtype=torch.int8)
    scale = torch.empty(flat.shape[0], device=flat.device, dtype=torch.float32)
    if resident_a8_eligible(flat):
        launch_resident_a8(flat, activation, scale)
    elif decode_eligible(flat, k=in_features):
        _quantize_decode_kernel[(1,)](
            flat, activation, scale, K=in_features,
            BLOCK_K=triton.next_power_of_2(in_features), num_warps=16,
        )
        decode_state = _DECODE_STATE.get()
        if decode_state is not None:
            decode_state.resident_quantization_kernels += 1
    else:
        block = 1024 if in_features >= 1024 else 256
        _quantize_rows_kernel[(flat.shape[0],)](
            flat, activation, scale, in_features=in_features,
            BLOCK_K=block, num_warps=4,
        )
    state.misses += 1
    state.entry = (
        _SharedA8Entry(source, in_features, activation, scale)
        if state.enabled
        else None
    )
    return activation, scale


def _config_for(rows: int, out_features: int, in_features: int) -> tuple[int, int, int, int]:
    """Return (BLOCK_M, BLOCK_N, BLOCK_J, num_warps)."""
    if rows <= 16:
        return 16, 128, 64, 4
    if rows <= 64:
        return 32, 128, 64, 4
    # The OFT vision MLP projections with N=1024/1152 were the one material
    # SM120 outlier under the former catch-all 64x128/8-warp geometry.  A
    # narrower output tile doubles useful program-level parallelism and avoids
    # assigning eight warps to a small-N tile.  The INT32 accumulation and
    # epilogue order are unchanged, so this is bit-exact to the generic path.
    if out_features <= 1152:
        return 32, 64, 64, 4
    return 64, 128, 64, 8


class W4A8TritonLinear(torch.nn.Module):
    """Inference-only Linear with K-major packed symmetric INT4 weights."""

    # The GEMV program avoids tl.dot but has too little parallelism to pay for
    # itself: these GEMMs are memory bound, so the wasted MMA rows of the
    # BLOCK_M=16 dot tile cost nothing while the dot path keeps far more
    # programs in flight.  Kept behind a threshold for measurement only.
    gemv_max_rows = 0

    def __init__(
        self,
        linear: torch.nn.Linear,
        fused_quantization: bool = True,
        *,
        weight_clip_ratios: torch.Tensor | float | None = None,
    ) -> None:
        super().__init__()
        if linear.in_features % 32:
            raise ValueError("in_features must be divisible by 32")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.fused_quantization = fused_quantization
        with torch.no_grad():
            weight = linear.weight.detach().float()
            if weight_clip_ratios is None:
                # Preserve the frozen v4/OpenVLA RTN path byte-for-byte.
                scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 7.0
                quantized = (weight / scale[:, None]).round().clamp(-8, 7).to(torch.int8)
                self.weight_quantizer = "rtn_per_output_row_v1"
            else:
                from actune.kernels.quantizers import symmetric_int4_quantize

                quantized, scale = symmetric_int4_quantize(
                    weight, clip_ratios=weight_clip_ratios
                )
                self.weight_quantizer = "awq_style_nested_clip_v1"
            half = self.in_features // 2
            low = quantized[:, :half].to(torch.int16) & 15
            high = (quantized[:, half:].to(torch.int16) & 15) << 4
            packed = (low | high).to(torch.uint8).t().contiguous()
        self.register_buffer("packed_weight", packed)
        self.register_buffer("weight_scale", scale.contiguous())
        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

    def _quantize(
        self, flat: torch.Tensor, *, source: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fused_quantization:
            return _quantize_rows_shared(
                flat if source is None else source,
                flat,
                in_features=self.in_features,
            )
        scale = flat.float().abs().amax(dim=1).clamp_min(1e-8) / 127.0
        activation = (flat.float() / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
        return activation, scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if (
            self.fused_quantization
            and decode_eligible(value, k=self.in_features, n=self.out_features)
            and not torch.cuda.is_current_stream_capturing()
        ):
            return graph_decode_linear(self, value, self.packed_weight, None,
                                       self.weight_scale, self.bias)
        original_shape = value.shape[:-1]
        flat = value.reshape(-1, self.in_features)
        activation, scale = self._quantize(flat, source=value)
        rows = flat.shape[0]
        output = torch.empty(
            (rows, self.out_features), device=value.device, dtype=value.dtype
        )
        bias = self.bias if self.bias is not None else self.weight_scale
        if decode_eligible(flat, k=self.in_features, n=self.out_features):
            decode_linear(activation, self.packed_weight, None, scale,
                          self.weight_scale, self.bias, output)
            return output.reshape(*original_shape, self.out_features)
        if rows <= self.gemv_max_rows:
            block_n, block_j = 128, 64
            _w4a8_gemv_kernel[(rows, triton.cdiv(self.out_features, block_n))](
                activation, self.packed_weight, scale, self.weight_scale, bias, output,
                rows,
                out_features=self.out_features, in_features=self.in_features,
                has_bias=self.bias is not None,
                BLOCK_N=block_n, BLOCK_J=block_j, num_warps=4,
            )
        else:
            block_m, block_n, block_j, warps = _config_for(rows, self.out_features, self.in_features)
            grid = (triton.cdiv(rows, block_m), triton.cdiv(self.out_features, block_n))
            _w4a8_gemm_kernel[grid](
                activation, self.packed_weight, scale, self.weight_scale, bias, output,
                rows,
                out_features=self.out_features, in_features=self.in_features,
                has_bias=self.bias is not None,
                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_J=block_j,
                num_warps=warps, num_stages=3,
            )
        return output.reshape(*original_shape, self.out_features)


def replace_with_w4a8_triton_(module: torch.nn.Module, filter_fn) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if filter_fn(child, name):
            setattr(module, name, W4A8TritonLinear(child))
            replaced += 1
        else:
            replaced += replace_with_w4a8_triton_(child, filter_fn)
    return replaced
