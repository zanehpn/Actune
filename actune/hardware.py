"""Joint frequency/power search from paired training replay measurements."""
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

REFERENCE = (1800, 300)


def region_key(decision):
    return f"{decision.region or 'fallback'}:{decision.configuration}"


@dataclass
class HardwarePolicy:
    table: dict
    fallback: tuple = REFERENCE
    reference: tuple = REFERENCE

    def __post_init__(self):
        self.table = {k: tuple(v) for k, v in self.table.items()}
        self.reference, self.fallback = tuple(self.reference), tuple(self.fallback)
        for point in [self.reference, self.fallback, *self.table.values()]:
            if len(point) != 2 or any(type(x) is not int or x <= 0 for x in point):
                raise ValueError("Operating point must be positive integer MHz and watts")

    def lookup(self, decision):
        return self.table.get(region_key(decision), self.fallback)

    def save(self, path):
        Path(path).write_text(json.dumps(dict(schema="actune_hardware_policy_v1",
            table=self.table, fallback=self.fallback, reference=self.reference), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.pop("schema", None) != "actune_hardware_policy_v1":
            raise ValueError("Unsupported hardware policy")
        return cls(**data)


def select_hardware_policy(records, *, reference=REFERENCE, latency_budget=1.10, min_samples=6):
    """Each row pairs a candidate call with the same reference observation/RNG.

    Required fields: key, sample_id, point, reference_ms, reference_j,
    candidate_ms, candidate_j, exact_actions, reference_resident_bytes,
    resident_bytes, reference_peak_bytes, peak_bytes. sample_id includes replay
    repeat. Latencies are complete prediction-call timings, not GEMM timings.
    No benchmark outcomes enter this search. Diagnostic verification follows.
    """
    if not 1 <= latency_budget <= 1.10 or min_samples < 1:
        raise ValueError("Mean inference slowdown budget must be between zero and 10%")
    by_key = defaultdict(lambda: defaultdict(list))
    for row in records:
        point = tuple(row["point"])
        HardwarePolicy({}, reference=point)
        numeric = [row[k] for k in ("reference_ms", "reference_j", "candidate_ms", "candidate_j")]
        if not np.isfinite(numeric).all() or min(numeric) <= 0:
            raise ValueError("Replay energy and latency must be finite and positive")
        by_key[row["key"]][point].append(row)
    if not by_key:
        raise ValueError("No paired training replay records")
    choices, reference_ms, reference_j = {}, {}, {}
    for key, points in sorted(by_key.items()):
        identities = {point: [str(r["sample_id"]) for r in rows] for point, rows in points.items()}
        if any(len(ids) != len(set(ids)) for ids in identities.values()):
            raise ValueError("Duplicate sample/repeat identity within an operating point")
        # Incomplete sweeps are not allowed to change the workload distribution.
        first = set(next(iter(identities.values())))
        if any(set(ids) != first for ids in identities.values()):
            raise ValueError("Operating points do not cover identical replay samples")
        rt = float(np.mean([sum(r["reference_ms"] for r in rows) for rows in points.values()]))
        re = float(np.mean([sum(r["reference_j"] for r in rows) for rows in points.values()]))
        reference_ms[key], reference_j[key] = rt, re
        options = [(tuple(reference), rt, re)]
        if len(first) >= min_samples:
            for point, rows in points.items():
                ok = all(r["exact_actions"] and r["resident_bytes"] <= r["reference_resident_bytes"]
                         and r["peak_bytes"] <= r["reference_peak_bytes"] for r in rows)
                if ok and point != tuple(reference):
                    time_ratio = sum(r["candidate_ms"] for r in rows) / sum(r["reference_ms"] for r in rows)
                    energy_ratio = sum(r["candidate_j"] for r in rows) / sum(r["reference_j"] for r in rows)
                    options.append((point, rt * time_ratio, re * energy_ratio))
        choices[key] = options
    keys = sorted(choices)
    total_t, total_e = sum(reference_ms.values()), sum(reference_j.values())
    proposals = []
    def offer(mapping):
        t = sum(x[1] for x in mapping.values()) / total_t
        e = sum(x[2] for x in mapping.values()) / total_e
        if t <= latency_budget + 1e-12:
            proposals.append((e, t, mapping))
    # Uniform settings and latency-penalty sweeps are both candidate tables.
    common = set.intersection(*[{r[0] for r in v} for v in choices.values()])
    for point in sorted(common):
        offer({k: next(r for r in choices[k] if r[0] == point) for k in keys})
    for penalty in [0., *np.logspace(-4, 4, 161)]:
        offer({k: min(choices[k], key=lambda r: (r[2] / total_e + penalty * r[1] / total_t, r[1], r[0]))
               for k in keys})
    e, t, mapping = min(proposals, key=lambda p: (p[0], p[1]))
    table = {k: r[0] for k, r in mapping.items()}
    fallback = table.get("fallback:q11", tuple(reference))
    return HardwarePolicy(table, fallback, tuple(reference)), dict(estimated_energy_ratio=e,
        estimated_inference_ratio=t, latency_budget=latency_budget)


def verify_diagnostic_policy(policy, records):
    """On a failed disjoint diagnostic, revert the entire table to reference."""
    points = {policy.fallback, *policy.table.values()}
    covered = {tuple(r["point"]) for r in records}
    valid = points <= covered and all(r["exact_actions"]
        and r["resident_bytes"] <= r["reference_resident_bytes"]
        and r["peak_bytes"] <= r["reference_peak_bytes"] for r in records)
    return policy if valid else HardwarePolicy({}, policy.reference, policy.reference)


def forecast_bound(states, trajectories, frames, *, chunk_size, quantile=0.95):
    """Fit a per-coordinate extrapolation cap from training trajectories only."""
    states, trajectories, frames = np.asarray(states, float), np.asarray(trajectories), np.asarray(frames)
    if states.ndim != 2 or len(states) != len(trajectories) or len(states) != len(frames) or not np.isfinite(states).all():
        raise ValueError("Invalid forecast calibration inputs")
    deltas = []
    for identity in np.unique(trajectories):
        ids = np.flatnonzero(trajectories == identity)
        ids = ids[np.argsort(frames[ids], kind="stable")]
        gaps = np.diff(frames[ids])
        if (gaps <= 0).any():
            raise ValueError("Frames must be unique within a training trajectory")
        if len(gaps):
            deltas.extend(np.abs(np.diff(states[ids], axis=0)) * chunk_size / gaps[:, None])
    if not deltas or chunk_size <= 0 or not 0 < quantile <= 1:
        raise ValueError("Need consecutive training contexts and a valid chunk size")
    return np.quantile(deltas, quantile, axis=0)
