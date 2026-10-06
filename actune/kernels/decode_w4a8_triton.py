"""Exact INT32 split-K kernels for single-token packed W4 / signed-R4 decode.

Weights retain the production K-major nibble layout. Partial sums are reduced
as integers; scaling and bias happen once, after the complete K reduction.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_decode_kernel(X, A, S, K: tl.constexpr, BLOCK_K: tl.constexpr):
    k = tl.arange(0, BLOCK_K)
    x = tl.load(X + k, k < K, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0), 1e-8) * (1.0 / 127.0)
    q = tl.extra.libdevice.rint(tl.extra.libdevice.div_rn(x, scale))
    q = tl.minimum(tl.maximum(q, -128.0), 127.0).to(tl.int8)
    tl.store(A + k, q, k < K)
    tl.store(S, scale)


@triton.jit
def _decode_partial_kernel(
    A,
    W,
    R,
    P,
    N: tl.constexpr,
    K: tl.constexpr,
    RESIDUAL: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
    USE_MMA: tl.constexpr,
):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    half: tl.constexpr = K // 2
    tiles_per_split: tl.constexpr = (K // 2 + BLOCK_J * SPLIT_K - 1) // (BLOCK_J * SPLIT_K)
    if USE_MMA:
        m = tl.arange(0, 16)
        acc = tl.full((16, BLOCK_N), 0, tl.int32)
    else:
        acc = tl.full((BLOCK_N,), 0, tl.int32)
    for tile in range(tiles_per_split):
        j = (ps * tiles_per_split + tile) * BLOCK_J + tl.arange(0, BLOCK_J)
        mask = (j[:, None] < half) & (n[None, :] < N)
        packed = tl.load(W + j[:, None] * N + n[None, :], mask, 0)
        lo = (((packed & 15).to(tl.int8)) ^ 8) - 8
        hi = ((((packed >> 4) & 15).to(tl.int8)) ^ 8) - 8
        if RESIDUAL:
            residual = tl.load(R + j[:, None] * N + n[None, :], mask, 0)
            rlo = (((residual & 15).to(tl.int8)) ^ 8) - 8
            rhi = ((((residual >> 4) & 15).to(tl.int8)) ^ 8) - 8
            lo = (lo.to(tl.int16) * 16 + rlo.to(tl.int16)).to(tl.int8)
            hi = (hi.to(tl.int16) * 16 + rhi.to(tl.int16)).to(tl.int8)
        if USE_MMA:
            al = tl.load(A + j[None, :] + tl.zeros((16, 1), tl.int32), (m[:, None] == 0) & (j[None, :] < half), 0)
            ah = tl.load(
                A + j[None, :] + half + tl.zeros((16, 1), tl.int32), (m[:, None] == 0) & (j[None, :] < half), 0
            )
            acc += tl.dot(al, lo, out_dtype=tl.int32)
            acc += tl.dot(ah, hi, out_dtype=tl.int32)
        else:
            al = tl.load(A + j, j < half, 0).to(tl.int32)
            ah = tl.load(A + j + half, j < half, 0).to(tl.int32)
            acc += tl.sum(al[:, None] * lo.to(tl.int32), axis=0)
            acc += tl.sum(ah[:, None] * hi.to(tl.int32), axis=0)
    if USE_MMA:
        tl.store(P + ps * N + n[None, :] + tl.zeros((16, 1), tl.int32), acc, (m[:, None] == 0) & (n[None, :] < N))
    else:
        tl.store(P + ps * N + n, acc, n < N)


@triton.jit
def _decode_reduce_kernel(
    P,
    AS,
    WS,
    B,
    Y,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    RESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    s = tl.arange(0, BLOCK_S)
    partial = tl.load(P + s[:, None] * N + n[None, :], (s[:, None] < SPLIT_K) & (n[None, :] < N), 0)
    acc = tl.sum(partial, axis=0).to(tl.float32)
    acc *= tl.load(AS)
    ws = tl.load(WS + n, n < N, 0)
    if RESIDUAL:
        ws = ws / 16.0
    acc *= ws
    if HAS_BIAS:
        acc += tl.load(B + n, n < N, 0)
    tl.store(Y + n, acc, n < N)


def launch_decode(
    activation, packed, residual, activation_scale, weight_scale, bias, output, partial, *, config=(64, 64, 16, True)
):
    """Launch into caller-owned buffers; useful for graph-safe benchmarking."""
    bn, bj, splits, mma = config
    n = packed.shape[1]
    k = packed.shape[0] * 2
    high = residual is not None
    _decode_partial_kernel[(triton.cdiv(n, bn), splits)](
        activation,
        packed,
        packed if residual is None else residual,
        partial,
        N=n,
        K=k,
        RESIDUAL=high,
        SPLIT_K=splits,
        BLOCK_N=bn,
        BLOCK_J=bj,
        USE_MMA=mma,
        num_warps=4,
        num_stages=2,
    )
    _decode_reduce_kernel[(triton.cdiv(n, 128),)](
        partial,
        activation_scale,
        weight_scale,
        weight_scale if bias is None else bias,
        output,
        N=n,
        SPLIT_K=splits,
        RESIDUAL=high,
        HAS_BIAS=bias is not None,
        BLOCK_N=128,
        BLOCK_S=triton.next_power_of_2(splits),
        num_warps=4,
    )


@triton.jit
def _decode_group_partial_kernel(
    A,
    WEIGHTS,
    RESIDUALS,
    P,
    N: tl.constexpr,
    K: tl.constexpr,
    MODES: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    group = tl.program_id(2)
    for i in tl.static_range(len(MODES)):
        if group == i:
            _decode_partial_kernel(
                A, WEIGHTS[i], RESIDUALS[i], P + i * SPLIT_K * N, N, K, MODES[i], SPLIT_K, BLOCK_N, BLOCK_J, True
            )


@triton.jit
def _decode_group_reduce_kernel(
    P,
    AS,
    SCALES,
    BIASES,
    OUTPUTS,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MODES: tl.constexpr,
    HAS_BIASES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    group = tl.program_id(1)
    for i in tl.static_range(len(MODES)):
        if group == i:
            _decode_reduce_kernel(
                P + i * SPLIT_K * N,
                AS,
                SCALES[i],
                BIASES[i],
                OUTPUTS[i],
                N,
                SPLIT_K,
                MODES[i],
                HAS_BIASES[i],
                BLOCK_N,
                BLOCK_S,
            )


def launch_decode_group(
    activation, weights, residuals, activation_scale, scales, biases, outputs, partial, *, splits=16, bn=128
):
    """Two launches for two or three projections, including mixed W4/R4."""
    n = weights[0].shape[1]
    k = weights[0].shape[0] * 2
    modes = tuple(r is not None for r in residuals)
    _decode_group_partial_kernel[(triton.cdiv(n, bn), splits, len(weights))](
        activation,
        tuple(weights),
        tuple(w if r is None else r for w, r in zip(weights, residuals)),
        partial,
        N=n,
        K=k,
        MODES=modes,
        SPLIT_K=splits,
        BLOCK_N=bn,
        BLOCK_J=64,
        num_warps=4,
        num_stages=2,
    )
    _decode_group_reduce_kernel[(triton.cdiv(n, 128), len(weights))](
        partial,
        activation_scale,
        tuple(scales),
        tuple(s if b is None else b for s, b in zip(scales, biases)),
        tuple(outputs),
        N=n,
        SPLIT_K=splits,
        MODES=modes,
        HAS_BIASES=tuple(b is not None for b in biases),
        BLOCK_N=128,
        BLOCK_S=triton.next_power_of_2(splits),
        num_warps=4,
    )


# A request-local switch also provides an exact legacy control for validation.

_DECODE_ENABLED: ContextVar[bool] = ContextVar("vla_decode_optimization_enabled", default=True)


@dataclass
class DecodeStats:
    split_k_calls: int = 0
    projection_groups: int = 0
    projection_calls_fused: int = 0
    partial_kernels: int = 0
    reduction_kernels: int = 0
    resident_quantization_kernels: int = 0
    graph_replays: int = 0
    graph_quantization_kernels: int = 0
    pending: dict[Any, Any] = field(default_factory=dict, repr=False)

    def metrics(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in (
                "split_k_calls",
                "projection_groups",
                "projection_calls_fused",
                "partial_kernels",
                "reduction_kernels",
                "resident_quantization_kernels",
                "graph_replays",
                "graph_quantization_kernels",
            )
        }


_DECODE_STATE: ContextVar[DecodeStats | None] = ContextVar("vla_decode_state", default=None)


@contextmanager
def decode_optimization_scope(*, enabled: bool = True):
    """Bound fusion results and execution counters to a single model forward."""
    state = DecodeStats()
    switch = _DECODE_ENABLED.set(enabled)
    token = _DECODE_STATE.set(state)
    try:
        yield state
    finally:
        state.pending.clear()
        _DECODE_STATE.reset(token)
        _DECODE_ENABLED.reset(switch)


@lru_cache(maxsize=None)
def _sm120(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (12, 0)


def decode_eligible(value: torch.Tensor, *, k: int, n: int | None = None) -> bool:
    """Restrict promotion to the measured Llama decode shapes and architecture."""
    return (
        _DECODE_ENABLED.get()
        and value.is_cuda
        and value.dtype == torch.bfloat16
        and value.numel() == k
        and value.shape[-1] == k
        and value.is_contiguous()
        and k in (4096, 11008)
        and (n is None or (k, n) in ((4096, 4096), (4096, 11008), (11008, 4096)))
        and _sm120(value.device)
    )


def decode_linear(activation, packed, residual, activation_scale, weight_scale, bias, output):
    n = packed.shape[1]
    bn = 128 if residual is not None or n > 4096 else 64
    partial = torch.empty((32, n), device=output.device, dtype=torch.int32)
    launch_decode(
        activation, packed, residual, activation_scale, weight_scale, bias, output, partial, config=(bn, 64, 32, True)
    )
    state = _DECODE_STATE.get()
    if state is not None:
        state.split_k_calls += 1
        state.partial_kernels += 1
        state.reduction_kernels += 1


class DecodeGraphPlan:
    """Replay the fixed decode launch sequence with private staging buffers.

    Inputs are copied and outputs cloned on every call, so callers never see
    mutable graph-owned output storage. Plans are keyed by CUDA stream and
    immutable packed tensor identities by the owning inference module.
    """

    def __init__(self, value, weights, residuals, scales, biases):
        import threading

        self.lock = threading.Lock()
        self.weights = tuple(weights)
        self.residuals = tuple(residuals)
        self.scales = tuple(scales)
        self.biases = tuple(biases)
        self.k = weights[0].shape[0] * 2
        self.n = weights[0].shape[1]
        self.groups = len(weights)
        self.input = torch.empty((1, self.k), device=value.device, dtype=value.dtype)
        self.activation = torch.empty_like(self.input, dtype=torch.int8)
        self.scale = torch.empty(1, device=value.device, dtype=torch.float32)
        self.output = torch.empty((self.groups, self.n), device=value.device, dtype=value.dtype)
        self.outputs = tuple(self.output[i : i + 1] for i in range(self.groups))
        self.partial = torch.empty((self.groups, 32, self.n), device=value.device, dtype=torch.int32)
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=value.device)
        current = torch.cuda.current_stream(value.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            self.input.copy_(value.reshape(1, self.k))
            for _ in range(3):
                self._launch()
        current.wait_stream(stream)
        with torch.cuda.graph(self.graph, stream=stream):
            self._launch()
        current.wait_stream(stream)

    def _launch(self):
        _quantize_decode_kernel[(1,)](
            self.input, self.activation, self.scale, K=self.k, BLOCK_K=triton.next_power_of_2(self.k), num_warps=16
        )
        if self.groups == 1:
            bn = 128 if self.residuals[0] is not None or self.n > 4096 else 64
            launch_decode(
                self.activation,
                self.weights[0],
                self.residuals[0],
                self.scale,
                self.scales[0],
                self.biases[0],
                self.output,
                self.partial,
                config=(bn, 64, 32, True),
            )
        else:
            bn = (
                64
                if all(r is not None for r in self.residuals)
                or (self.n == 4096 and any(r is not None for r in self.residuals))
                else 128
            )
            launch_decode_group(
                self.activation,
                self.weights,
                self.residuals,
                self.scale,
                self.scales,
                self.biases,
                self.outputs,
                self.partial,
                splits=32,
                bn=bn,
            )

    @property
    def workspace_bytes(self):
        return sum(
            t.numel() * t.element_size() for t in (self.input, self.activation, self.scale, self.output, self.partial)
        )

    def run(self, value):
        with self.lock:
            self.input.copy_(value.reshape(1, self.k))
            self.graph.replay()
            # A single contiguous clone gives every projection fresh storage.
            output = self.output.clone()
        state = _DECODE_STATE.get()
        if state is not None:
            state.graph_replays += 1
            state.graph_quantization_kernels += 1
            state.resident_quantization_kernels += 1
            state.partial_kernels += 1
            state.reduction_kernels += 1
            if self.groups == 1:
                state.split_k_calls += 1
            else:
                state.projection_groups += 1
                state.projection_calls_fused += self.groups
        return output


def _tensor_version(tensor):
    if tensor is None or tensor.is_inference():
        return None
    return tensor._version


def graph_plan_key(value, weights, residuals, scales, biases):
    return (
        torch.cuda.current_stream(value.device).cuda_stream,
        tuple((id(t), _tensor_version(t)) for t in (*weights, *residuals, *scales, *biases)),
    )


def graph_decode_linear(owner, value, packed, residual, scale, bias):
    cache = getattr(owner, "_decode_graph_plans", None)
    if cache is None:
        cache = {}
        owner._decode_graph_plans = cache
    key = graph_plan_key(value, (packed,), (residual,), (scale,), (bias,))
    plan = cache.get(key)
    if plan is None:
        # Keep only current buffers for each precision/stream; weights are frozen
        # during normal inference. Clearing avoids stale storage after model moves.
        if len(cache) >= 8:
            cache.clear()
        plan = DecodeGraphPlan(value, (packed,), (residual,), (scale,), (bias,))
        cache[key] = plan
    return plan.run(value).reshape(*value.shape[:-1], packed.shape[1])


def decode_workspace_summary(module):
    """Count explicit persistent buffers after warmup (excludes CUDA metadata)."""
    plans = {}
    mlp_plans = {}
    for child in module.modules():
        for plan in getattr(child, "_decode_graph_plans", {}).values():
            plans[id(plan)] = plan
        controller = getattr(child, "_decode_mlp_controller", None)
        if controller is not None:
            for plan in controller.plans.values():
                mlp_plans[id(plan)] = plan
        group = getattr(child, "_decode_projection_group", None)
        if group is not None:
            for plan in group.plans.values():
                plans[id(plan)] = plan
    return {
        "decode_graph_plan_count": len(plans),
        "mlp_graph_plan_count": len(mlp_plans),
        "mlp_graph_input_output_bytes": sum(
            t.numel() * t.element_size() for plan in mlp_plans.values() for t in (plan.input, plan.output)
        ),
        "decode_graph_explicit_workspace_bytes": sum(plan.workspace_bytes for plan in plans.values()),
    }
