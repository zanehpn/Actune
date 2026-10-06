"""Request-local controls and counters for the second runtime optimization pass."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, asdict

_RUNTIME_ENABLED = ContextVar("vla_runtime_optimization_enabled", default=True)
# Dense candidates remain opt-in: full-policy measurements did not improve.
_DENSE_CANDIDATES = ContextVar("vla_dense_runtime_candidates", default=False)
_RUNTIME_STATE = ContextVar("vla_runtime_optimization_stats", default=None)


@dataclass
class RuntimeStats:
    resident_a8_kernels: int = 0
    mlp_graph_replays: int = 0
    predecoded_r4_gemms: int = 0


@contextmanager
def runtime_optimization_scope(*, enabled=True, dense_candidates=None):
    dense_token = None if dense_candidates is None else _DENSE_CANDIDATES.set(dense_candidates)
    state = RuntimeStats()
    switch = _RUNTIME_ENABLED.set(enabled)
    token = _RUNTIME_STATE.set(state)
    try:
        yield state
    finally:
        _RUNTIME_STATE.reset(token)
        _RUNTIME_ENABLED.reset(switch)
        if dense_token is not None:
            _DENSE_CANDIDATES.reset(dense_token)


def runtime_stats_dict(state):
    return asdict(state)
