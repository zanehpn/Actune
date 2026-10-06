import copy
import tempfile
from pathlib import Path
import threading
import unittest

import numpy as np

from actune._tree_growth import candidates, grow
from actune.allocation import build_banks
from actune.features import ACTION_FORECAST_FEATURE_NAMES, action_forecast_features
from actune.hardware import HardwarePolicy, select_hardware_policy, verify_diagnostic_policy, forecast_bound
from actune.precision import active_precision_profile, precision_profile
from actune.runtime import Controller, Prediction
from actune.tree import PrecisionTree, action_loss, balanced_weights, fit_candidates


def branch(feature=0):
    return dict(profile_index=2, feature=feature, threshold=0.5,
        left=dict(profile_index=0), right=dict(profile_index=2))


def split(prefix):
    x = np.zeros((120, 14))
    x[:, 0] = np.repeat([0., 1., 2.], 40)
    losses = np.full((120, 4), 2.)
    for i in range(3):
        losses[i*40:(i+1)*40, i] = 0.
    return dict(x=x, losses=losses, has_history=np.ones(120, bool),
        suite=np.array(["spatial"] * 120), trajectory=np.array([prefix + str(i // 20) for i in range(120)]))


class TreeTests(unittest.TestCase):
    def test_balancing_is_suite_then_trajectory(self):
        w = balanced_weights(["a", "a", "a", "b"], ["x", "x", "y", "z"], [1, 1, 1, 1])
        np.testing.assert_allclose(w, [.125, .125, .25, .5])

    def test_loss_has_gripper_penalty(self):
        a = np.zeros((2, 7)); b = a.copy(); b[:, :6] = 2; b[0, 6] = 1
        self.assertAlmostEqual(action_loss(a, b), 2.05)

    def test_features_and_fallback(self):
        f = action_forecast_features(np.zeros((8, 7)), gripper_margin=0.3)
        self.assertEqual(len(f), 13)
        self.assertEqual(f["cancellation"], 0.)
        tree = PrecisionTree(branch(), 1)
        self.assertEqual(tree.route(None, [0]).configuration, "q11")
        self.assertEqual(tree.route(f, [np.nan]).configuration, "q11")
        self.assertEqual(tree.route(f, [0]).configuration, "q00")

    def test_exact_pruning_includes_every_size(self):
        train = split("train")
        grown = grow(train["x"], train["losses"])
        def exhaustive(t):
            possibilities = [(1, t["risk"])]
            if "left" in t:
                possibilities += [(a+b, ra+rb) for a, ra in exhaustive(t["left"])
                                  for b, rb in exhaustive(t["right"])]
            return possibilities
        expected = {}
        for k, risk in exhaustive(grown):
            expected[k] = min(expected.get(k, float("inf")), risk)
        cs = candidates(grown, 120, all_sizes=True)
        self.assertEqual({c["leaves"] for c in cs}, set(expected))
        for c in cs:
            self.assertAlmostEqual(c["fitting_loss"], expected[c["leaves"]] / 120)

    def test_validation_never_refits_and_no_fixed_class_count(self):
        fit, val = split("fit"), split("val")
        cs = fit_candidates(fit, val)
        self.assertEqual([c["leaves"] for c in cs], [1, 2, 3])
        self.assertEqual(cs[-1]["validation_loss"], 0.)
        val["losses"] = val["losses"][:, ::-1]
        other = fit_candidates(fit, val)
        self.assertEqual([c["tree"] for c in cs], [c["tree"] for c in other])

    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_candidates(split("same"), split("same"))

    def test_history_free_validation_uses_fallback(self):
        val = split("val"); val["has_history"][:] = False
        cs = fit_candidates(split("fit"), val)
        self.assertEqual(len({c["validation_loss"] for c in cs}), 1)

    def test_roundtrip(self):
        tree = PrecisionTree(branch(), 1)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "tree.json"; tree.save(p)
            self.assertEqual(PrecisionTree.load(p).tree, tree.tree)


class BankTests(unittest.TestCase):
    def fixture(self, budget):
        shapes = {f"layer{i}": [8, 128] for i in range(8)}
        counts = {n: 128 for n in shapes}
        modes = [f"w{w}a{a}" for w in (2, 4, 8) for a in (2, 4, 8)]
        errors = {n: {m: 1 / int(m[1]) + 1 / int(m[3]) for m in modes} for n in shapes}
        sizes = {n: {m: 8 * 128 * int(m[1]) // 8 + 32 for m in modes} for n in shapes}
        return build_banks(shapes, counts, [list(shapes)], errors, sizes, budget=budget, storage_limit=10000)

    def test_four_configurations_share_exactly_two_alternatives(self):
        for budget in ("w4a8", "w4a4"):
            bank = next(iter(self.fixture(budget).values()))
            bank.validate()
            full_copies = sum(sum(bank.version_bytes[n][m] for n, m in c.items()) for c in bank.configurations.values())
            self.assertLess(bank.resident_bytes(), full_copies)
            self.assertEqual(len({tuple(c.values()) for c in bank.configurations.values()}), 4)

    def test_inactive_versions_count_and_coverage(self):
        bank = next(iter(self.fixture("w4a8").values()))
        bank.storage_limit = bank.resident_bytes() - 1
        with self.assertRaisesRegex(ValueError, "resident"):
            bank.validate()
        with self.assertRaisesRegex(ValueError, "target"):
            bank.check_request("q11", {})

    def test_request_profile_does_not_leak(self):
        self.assertIsNone(active_precision_profile())
        with precision_profile("q00"):
            with precision_profile("q11"):
                self.assertEqual(active_precision_profile(), "q11")
            self.assertEqual(active_precision_profile(), "q00")
        self.assertIsNone(active_precision_profile())


class HardwareTests(unittest.TestCase):
    def rows(self, latency=1.05, exact=True):
        return [dict(key="root:q00", sample_id=str(i), point=[1500, 225],
            reference_ms=10., reference_j=2., candidate_ms=10*latency, candidate_j=1.,
            exact_actions=exact, reference_resident_bytes=100, resident_bytes=100,
            reference_peak_bytes=200, peak_bytes=200) for i in range(6)]

    def test_joint_point_and_latency_constraint(self):
        policy, report = select_hardware_policy(self.rows())
        self.assertEqual(policy.table["root:q00"], (1500, 225))
        self.assertLessEqual(report["estimated_inference_ratio"], 1.10)
        slow, _ = select_hardware_policy(self.rows(1.11))
        self.assertEqual(slow.table["root:q00"], (1800, 300))

    def test_parity_memory_and_minimum_support(self):
        for rows in [self.rows(exact=False), self.rows()[:5]]:
            policy, _ = select_hardware_policy(rows)
            self.assertEqual(policy.table["root:q00"], (1800, 300))
        rows = self.rows(); rows[0]["peak_bytes"] += 1
        policy, _ = select_hardware_policy(rows)
        self.assertEqual(policy.table["root:q00"], (1800, 300))

    def test_failed_diagnostic_reverts_whole_table(self):
        policy, _ = select_hardware_policy(self.rows())
        self.assertEqual(verify_diagnostic_policy(policy, []).table, {})

    def test_forecast_bound_uses_within_trajectory_deltas(self):
        bound = forecast_bound([[0.], [1.], [100.], [102.]], ["a", "a", "b", "b"], [0, 8, 0, 8], chunk_size=8, quantile=1)
        np.testing.assert_array_equal(bound, [2.])

    def test_single_call_dwell_and_no_update_during_inference(self):
        class Device:
            def __init__(self): self.points = []; self.point = None; self.in_call = False
            def set(self, f, p):
                if self.in_call: raise AssertionError("updated during inference")
                self.point = f, p; self.points.append(self.point)
            def snapshot(self):
                return dict(application_sm_mhz=self.point[0], power_limit_mw=self.point[1]*1000)
        device = Device(); calls = []
        tree = PrecisionTree(branch(), 1)
        hw = HardwarePolicy({"root/left:q00": (1500, 225)})
        def policy(obs, config):
            device.in_call = True
            try:
                calls.append((config, device.point))
                self.assertEqual(active_precision_profile(), config)
                return Prediction(np.zeros((8, 7)), 1.)
            finally: device.in_call = False
        with Controller(tree, policy, device=device, hardware_policy=hw, state_bound=[1.]) as controller:
            controller.start_episode()
            controller.predict(dict(states=[0.]))
            self.assertFalse(controller.trace["update_scheduled"])
            controller.predict(dict(states=[0.]))
            self.assertTrue(controller.trace["update_scheduled"])
            controller.predict(dict(states=[0.]))
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0], "q11")
        self.assertEqual(calls[-1][1], (1500, 225))
        self.assertEqual(device.points, [(1800, 300), (1500, 225)])

    def test_forecast_miss_does_not_override_current_precision(self):
        class Device:
            def set(self, f, p): self.point = f, p
            def snapshot(self):
                return dict(application_sm_mhz=self.point[0], power_limit_mw=self.point[1]*1000)
        device = Device()
        # This split uses CURRENT proprioception, so the third observation
        # deliberately disagrees with the forecast from the second call.
        tree = PrecisionTree(branch(feature=13), 1)
        hw = HardwarePolicy({"root/left:q00": (1500, 225)})
        def policy(obs, config): return Prediction(np.zeros((8, 7)), 1.)
        with Controller(tree, policy, device=device, hardware_policy=hw) as c:
            c.start_episode()
            c.predict(dict(states=[0.]))
            c.predict(dict(states=[0.]))
            c.predict(dict(states=[1.]))
            self.assertEqual(c.trace["configuration"], "q11")
            self.assertEqual(c.trace["operating_point"], (1500, 225))
            c.predict(dict(states=["invalid"]))
            self.assertEqual(c.trace["configuration"], "q11")


if __name__ == "__main__":
    unittest.main()
