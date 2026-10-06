"""Real integer kernels against explicit calibrated-code multiplication."""
import importlib
import unittest

import torch

from actune.calibration import activation_reference, calibrate_linear, jacobian_row_weights
from actune.kernels.quantizers import weight_codes


class CPUCalibrationTests(unittest.TestCase):
    def test_all_imports_are_self_contained(self):
        for name in ("w4a8_triton", "w2a8_triton", "w8a8_triton", "wxa4_native", "grouped", "tuning"):
            importlib.import_module("actune.kernels." + name)
        importlib.import_module("actune.device.dvfs_dcgm_persistent")

    def test_nine_calibrated_pairs_and_jacobian(self):
        torch.manual_seed(7)
        layer = torch.nn.Linear(64, 8, bias=False).to(torch.bfloat16)
        x = torch.randn(8, 64, dtype=torch.bfloat16, requires_grad=True)
        hidden = layer(x)
        gains = jacobian_row_weights(hidden, {"linear": hidden})["linear"]
        fitted = calibrate_linear(layer, x, gains, clip_grid=(1., .9))
        self.assertEqual(len(fitted["errors"]), 9)
        self.assertTrue(all(v >= 0 for v in fitted["errors"].values()))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class GPUKernelTests(unittest.TestCase):
    def test_all_nine_native_pairs(self):
        from actune.kernels.grouped import GroupedLinear
        torch.manual_seed(7)
        for shape in [(1, 64, 32), (17, 128, 96)]:
            m, k, n = shape
            linear = torch.nn.Linear(k, n, bias=True, device="cuda", dtype=torch.bfloat16)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            for w in (2, 4, 8):
                for a in (2, 4, 8):
                    with self.subTest(shape=shape, w=w, a=a), torch.inference_mode():
                        backend = GroupedLinear(linear, w, a, None)
                        actual = backend(x)
                        q, scale = weight_codes(linear.weight, w, a)
                        expected = torch.nn.functional.linear(activation_reference(x, a),
                            q.float() * scale[:, None], linear.bias.float()).to(x.dtype)
                        # Groupwise FP32 rescaling may round differently from a
                        # float reference GEMM. Codebook/math agreement is tested
                        # to BF16 accuracy; bit-exact action parity is separate.
                        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                        self.assertEqual(actual.shape, (m, n))


if __name__ == "__main__":
    unittest.main()
