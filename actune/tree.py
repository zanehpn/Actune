"""Suite/trajectory-balanced loss trees; validation never refits a split."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from ._tree_growth import grow, candidates, predict
from .features import ACTION_FORECAST_FEATURE_NAMES, action_forecast_features

CONFIGURATIONS = ("q00", "q01", "q11", "q10")


def action_loss(actions, reference, gripper_weight=0.1):
    """Postprocessed actions [..., H, 7]; gripper values must be discrete."""
    a, b = np.asarray(actions, float), np.asarray(reference, float)
    if a.shape != b.shape or a.ndim < 2 or a.shape[-1] != 7 or a.shape[-2] == 0:
        raise ValueError("Action chunks must have matching [..., H, 7] shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(gripper_weight) or gripper_weight < 0:
        raise ValueError("Actions and nonnegative loss weight must be finite")
    return np.abs(a[..., :6] - b[..., :6]).mean(axis=(-2, -1)) + gripper_weight * (
        a[..., 6] != b[..., 6]).mean(axis=-1)


def balanced_weights(suites, trajectories, valid):
    suites, trajectories, valid = np.asarray(suites), np.asarray(trajectories), np.asarray(valid, bool)
    if not (suites.shape == trajectories.shape == valid.shape) or suites.ndim != 1:
        raise ValueError("Expected one suite, trajectory ID and history flag per context")
    groups = np.unique(suites)
    weights = np.zeros(len(valid))
    for suite in groups:
        ids = np.unique(trajectories[(suites == suite) & valid])
        if not len(ids):
            raise ValueError(f"No valid fitting histories in suite {suite}")
        for identity in ids:
            mask = (suites == suite) & (trajectories == identity) & valid
            weights[mask] = 1.0 / (len(groups) * len(ids) * mask.sum())
    return weights


def trajectory_mean(values, suites, trajectories):
    values, suites, trajectories = np.asarray(values), np.asarray(suites), np.asarray(trajectories)
    return float(np.mean([
        np.mean([values[(suites == s) & (trajectories == t)].mean()
                 for t in np.unique(trajectories[suites == s])])
        for s in np.unique(suites)]))


def _split(data):
    x, losses = np.asarray(data["x"], float), np.asarray(data["losses"], float)
    history = np.asarray(data["has_history"], bool)
    suites, trajectories = np.asarray(data["suite"]), np.asarray(data["trajectory"])
    n = len(x)
    if (x.ndim != 2 or x.shape[1] < 14 or losses.shape != (n, 4) or n == 0
            or any(a.shape != (n,) for a in (history, suites, trajectories))
            or not np.isfinite(losses).all() or (losses < 0).any()):
        raise ValueError("Invalid calibration arrays (13 action features followed by proprioception)")
    valid = history & np.isfinite(x).all(axis=1)
    return x, losses, valid, suites, trajectories


def fit_candidates(fitting, validation, *, max_depth=5, min_leaf=20, max_leaves=None):
    """Return every feasible subtree and held-out loss; do not auto-select K=3.

    Each input is a mapping with x, losses (four columns in CONFIGURATIONS
    order), has_history, suite, trajectory, and optionally observation_id.
    Trajectory IDs must identify the original demonstration, not row numbers.
    """
    if not 0 <= max_depth <= 5 or min_leaf < 20:
        raise ValueError("Paper capacity is depth <= 5 and >= 20 contexts per child")
    tx, tl, valid, ts, tids = _split(fitting)
    vx, vl, has, vs, vids = _split(validation)
    if tx.shape[1] != vx.shape[1] or set(ts) != set(vs):
        raise ValueError("Fitting/validation feature dimensions and suites must match")
    if set(zip(ts.tolist(), tids.tolist())) & set(zip(vs.tolist(), vids.tolist())):
        raise ValueError("Fitting and validation trajectories overlap")
    if "observation_id" in fitting and "observation_id" in validation:
        if set(fitting["observation_id"]) & set(validation["observation_id"]):
            raise ValueError("Fitting and validation observations overlap")
    weights = balanced_weights(ts, tids, valid)
    n = int(valid.sum())
    grown = grow(tx[valid], tl[valid] * (weights[valid] * n)[:, None], max_depth, min_leaf)
    curve = candidates(grown, n, all_sizes=True)
    if max_leaves is not None:
        curve = [c for c in curve if c["leaves"] <= max_leaves]
    for c in curve:
        q = np.full(len(vx), CONFIGURATIONS.index("q11"), int)
        q[has] = predict(c["tree"], vx[has])
        c["validation_loss"] = trajectory_mean(vl[np.arange(len(vl)), q], vs, vids)
        c["state_dim"] = tx.shape[1] - 13
    return curve


@dataclass(frozen=True)
class Decision:
    region: str | None
    configuration: str


class PrecisionTree:
    def __init__(self, tree, state_dim, *, fallback="q11"):
        if not isinstance(state_dim, int) or state_dim < 1 or fallback not in CONFIGURATIONS:
            raise ValueError("Invalid tree dimensions or fallback")
        self.tree, self.state_dim, self.fallback = deepcopy(tree), state_dim, fallback
        def check(t, depth=0):
            if depth > 5 or t.get("profile_index") not in range(4):
                raise ValueError("Invalid tree depth or configuration index")
            if "left" in t:
                if (not isinstance(t.get("feature"), int) or not 0 <= t["feature"] < 13 + state_dim
                        or not np.isfinite(t.get("threshold", np.nan)) or "right" not in t):
                    raise ValueError("Invalid tree split")
                check(t["left"], depth + 1)
                check(t["right"], depth + 1)
            elif "right" in t:
                raise ValueError("Incomplete tree split")
        check(self.tree)

    def route_vector(self, x):
        x = np.asarray(x, float)
        if x.shape != (13 + self.state_dim,) or not np.isfinite(x).all():
            return Decision(None, self.fallback)
        node, region = self.tree, "root"
        while "left" in node:
            side = "left" if x[node["feature"]] <= node["threshold"] else "right"
            region += "/" + side
            node = node[side]
        return Decision(region, CONFIGURATIONS[node["profile_index"]])

    def route(self, previous_features, state):
        if previous_features is None:
            return Decision(None, self.fallback)
        try:
            state = np.asarray(state, float).reshape(-1)
            if len(state) != self.state_dim:
                return Decision(None, self.fallback)
            x = np.r_[[previous_features[k] for k in ACTION_FORECAST_FEATURE_NAMES], state]
            return self.route_vector(x)
        except (KeyError, TypeError, ValueError):
            return Decision(None, self.fallback)

    def save(self, path):
        Path(path).write_text(json.dumps(dict(schema="actune_precision_tree_v1", tree=self.tree,
            state_dim=self.state_dim, fallback=self.fallback,
            configurations=CONFIGURATIONS), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("schema") != "actune_precision_tree_v1" or tuple(data["configurations"]) != CONFIGURATIONS:
            raise ValueError("Unsupported tree artifact")
        return cls(data["tree"], data["state_dim"], fallback=data["fallback"])


def routing_cost(tree, x, *, rounds=21, repeats=3):
    """CPU traversal only, in microseconds/decision; excludes feature extraction."""
    rows = np.asarray(x, float)
    if rows.ndim != 2 or len(rows) == 0 or not np.isfinite(rows).all():
        raise ValueError("Routing benchmark needs nonempty finite rows")
    def scalar(row):
        node = tree
        while "left" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return node["profile_index"]
    for row in rows:
        scalar(row)
    times = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            for row in rows:
                scalar(row)
        times.append((time.perf_counter_ns() - start) / (len(rows) * repeats * 1000))
    return float(np.median(times))
