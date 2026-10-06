"""Shared resident bank accounting and request-scoped precision dispatch."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math

from ._allocation.pi05_budget import pair
from .tree import CONFIGURATIONS

_PROFILE = ContextVar("actune_precision_profile", default=None)


def active_precision_profile():
    return _PROFILE.get()


@contextmanager
def precision_profile(configuration):
    if configuration not in CONFIGURATIONS:
        raise ValueError("Unknown ActTune configuration")
    token = _PROFILE.set(configuration)
    try:
        yield
    finally:
        _PROFILE.reset(token)


@dataclass
class SharedBank:
    configurations: dict
    shapes: dict
    activation_elements: dict
    shape_groups: list
    version_bytes: dict
    budget: str
    storage_limit: int

    def resident_bytes(self):
        return sum(self.version_bytes[name][mode] for name in self.shapes
                   for mode in {c[name] for c in self.configurations.values()})

    def metrics(self, configuration, activation_elements=None):
        assignment = self.configurations[configuration]
        counts = self.activation_elements if activation_elements is None else activation_elements
        if set(counts) != set(self.shapes) or any(type(v) is not int or v <= 0 for v in counts.values()):
            raise ValueError("Every target must execute with a positive activation-element count")
        total = sum(math.prod(s) for s in self.shapes.values())
        weights = sum(math.prod(self.shapes[n]) * pair(m)[0] for n, m in assignment.items()) / total
        activations = sum(counts[n] * pair(m)[1] for n, m in assignment.items()) / sum(counts.values())
        groups = [sum(counts[n] * pair(assignment[n])[1] for n in g) / sum(counts[n] for n in g)
                  for g in self.shape_groups]
        return dict(weight_bits=weights, activation_bits=activations, group_activation_bits=groups)

    def check_request(self, configuration, counts=None):
        m = self.metrics(configuration, counts)
        if m["weight_bits"] > 4 + 1e-12:
            raise ValueError("Active weight budget exceeds four bits")
        if self.budget == "w4a4" and max(m["activation_bits"], *m["group_activation_bits"]) > 4 + 1e-12:
            raise ValueError("Active activation budget exceeds four bits")
        if self.budget == "w4a8" and any(pair(v)[1] != 8 for v in self.configurations[configuration].values()):
            raise ValueError("W4A8 requires A8 on every target")
        return m

    def validate(self):
        if self.budget not in ("w4a4", "w4a8") or set(self.configurations) != set(CONFIGURATIONS):
            raise ValueError("Expected a W4A4/W4A8 bank with four configurations")
        if not self.shapes or any(len(s) != 2 or any(type(v) is not int or v <= 0 for v in s) for s in self.shapes.values()):
            raise ValueError("Invalid matrix shapes")
        if sorted(n for g in self.shape_groups for n in g) != sorted(self.shapes) or any(not g for g in self.shape_groups):
            raise ValueError("Invocation-shape groups must partition the targeted layers")
        if any(set(c) != set(self.shapes) for c in self.configurations.values()):
            raise ValueError("Configuration coverage differs from targeted layers")
        versions = {n: {c[n] for c in self.configurations.values()} for n in self.shapes}
        if sorted(len(v) for v in versions.values() if len(v) > 1) != [2, 2]:
            raise ValueError("Paper bank requires exactly two binary switchable matrices")
        if len({tuple(c[n] for n in sorted(self.shapes)) for c in self.configurations.values()}) != 4:
            raise ValueError("Configurations must realize all four alternatives")
        for n, modes in versions.items():
            for mode in modes:
                pair(mode)
                size = self.version_bytes[n][mode]
                if type(size) is not int or size <= 0:
                    raise ValueError("Version storage must include packed weights, scales and bias")
        for q in CONFIGURATIONS:
            self.check_request(q)
        if self.storage_limit <= 0 or self.resident_bytes() > self.storage_limit:
            raise ValueError("All resident versions together exceed storage budget")
        return self
