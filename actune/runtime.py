"""One selected policy call per request with inter-call hardware preparation."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import numpy as np

from .features import action_forecast_features
from .precision import precision_profile
from .tree import Decision


@dataclass
class Prediction:
    actions: np.ndarray
    gripper_margin: float


class Controller:
    """Own one episode/one GPU. The supplied policy must finish GPU work on return.

    policy(observation, configuration) -> Prediction; all resident-layer
    dispatch takes place in precision_profile(configuration). Pass a CUDA
    synchronize callback when the policy enqueues work asynchronously.
    device.set(frequency_mhz, power_w) and device.snapshot() control hardware.
    """
    def __init__(self, tree, policy, *, device=None, hardware_policy=None,
                 state_bound=None, synchronize=lambda: None, audit=None):
        self.tree, self.policy, self.device = tree, policy, device
        self.hardware_policy, self.synchronize, self.audit = hardware_policy, synchronize, audit
        if (device is None) != (hardware_policy is None):
            raise ValueError("Hardware device and calibrated policy must be supplied together")
        self.bound = np.asarray(state_bound if state_bound is not None else np.zeros(tree.state_dim), float)
        if self.bound.shape != (tree.state_dim,) or not np.isfinite(self.bound).all() or (self.bound < 0).any():
            raise ValueError("Invalid training-calibrated forecast bound")
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="actune-dvfs") if device else None
        self.pending = None
        self.current = None
        self.started = False
        self.closed = False
        self.trace = None

    def _apply(self, point):
        self.device.set(*point)
        state = self.device.snapshot()
        if (state["application_sm_mhz"] != point[0] or state["power_limit_mw"] != point[1] * 1000):
            raise RuntimeError("GPU operating point readback mismatch")
        return tuple(point)

    def drain(self):
        begin = time.perf_counter()
        if self.pending is not None:
            pending, self.pending = self.pending, None
            self.current = pending.result()
        return (time.perf_counter() - begin) * 1000

    def start_episode(self):
        if self.closed:
            raise RuntimeError("Controller is closed")
        self.drain()
        self.previous_features = self.previous_state = None
        self.dwell = 0
        if self.device:
            self.current = self._apply(self.hardware_policy.fallback)
        self.started = True

    def predict(self, observation):
        if not self.started or self.closed:
            raise RuntimeError("Call start_episode() before inference")
        begin = time.perf_counter()
        wait = self.drain()
        try:
            state = np.asarray(observation.get("states", []), float).reshape(-1)
        except (TypeError, ValueError):
            state = np.full(self.tree.state_dim, np.nan)
        decision = self.tree.route(self.previous_features, state)
        active_point = self.current
        with precision_profile(decision.configuration):
            result = self.policy(observation, decision.configuration)
            self.synchronize()
        if self.audit is not None:
            self.audit(decision.configuration)
        self.dwell += 1
        # Invalid history routes the following request to the fallback.
        try:
            self.previous_features = action_forecast_features(result.actions, gripper_margin=result.gripper_margin)
        except (ValueError, TypeError):
            self.previous_features = None
        forecast = Decision(None, self.tree.fallback)
        if state.shape == self.bound.shape and np.isfinite(state).all():
            delta = np.zeros_like(state) if self.previous_state is None else state - self.previous_state
            extrapolated = state + np.clip(delta, -self.bound, self.bound)
            forecast = self.tree.route(self.previous_features, extrapolated)
            self.previous_state = state.copy()
        else:
            self.previous_state = None
        scheduled = False
        if self.device:
            point = self.hardware_policy.lookup(forecast)
            if point != self.current and self.dwell >= 2:
                self.pending = self.pool.submit(self._apply, point)
                self.dwell = 0
                scheduled = True
        self.trace = dict(region=decision.region, configuration=decision.configuration,
            operating_point=active_point, forecast_region=forecast.region,
            forecast_configuration=forecast.configuration, update_scheduled=scheduled,
            unhidden_wait_ms=wait, prediction_ms=(time.perf_counter() - begin) * 1000)
        return result

    def end_episode(self):
        self.drain()
        self.started = False

    def close(self):
        try:
            self.end_episode()
        finally:
            if self.pool:
                self.pool.shutdown(wait=True)
            self.closed = True
            # Device ownership stays with its context manager, which restores
            # original settings after all controller workers have drained.

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
