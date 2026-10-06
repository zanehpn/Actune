"""Exact resident-row A8 for measured SM120 prefill and OFT shapes."""

import torch
import triton
import triton.language as tl
from .runtime_options import _RUNTIME_ENABLED, _RUNTIME_STATE, _DENSE_CANDIDATES
from .decode_w4a8_triton import _DECODE_ENABLED, _sm120

_SHAPES = frozenset(((256, 1152), (256, 4304), (261, 4096), (284, 4096), (284, 11008), (598, 4096)))


@triton.jit
def _resident_a8_kernel(X, A, S, K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    value = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 0), 1e-8) * (1.0 / 127.0)
    q = tl.extra.libdevice.rint(tl.extra.libdevice.div_rn(value, scale))
    q = tl.minimum(tl.maximum(q, -128.0), 127.0).to(tl.int8)
    tl.store(A + row * K + k, q, k < K)
    tl.store(S + row, scale)


def resident_a8_eligible(value):
    return (
        _DENSE_CANDIDATES.get()
        and _RUNTIME_ENABLED.get()
        and _DECODE_ENABLED.get()
        and value.is_cuda
        and value.dtype == torch.bfloat16
        and tuple(value.shape) in _SHAPES
        and value.is_contiguous()
        and _sm120(value.device)
    )


def launch_resident_a8(value, activation, scale):
    m, k = value.shape
    _resident_a8_kernel[(m,)](value, activation, scale, K=k, BLOCK_K=triton.next_power_of_2(k), num_warps=16)
    state = _RUNTIME_STATE.get()
    if state is not None:
        state.resident_a8_kernels += 1
