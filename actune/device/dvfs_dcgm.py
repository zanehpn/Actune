"""DCGM hostengine control of one GPU's application clocks and power cap.

The hostengine performs writes; NVML is used only for local readback. Memory
application clocks stay at their initial value. Actual SM clocks may be lower
than application clocks, particularly when a power cap is binding.
"""
from __future__ import annotations

import ctypes as C
import json
import subprocess

from .dvfs_nvml import NVMLClocks


class DCGMClocks(NVMLClocks):
    def __init__(self, gpu_index, expected_uuid, group_id):
        if type(group_id) is not int or group_id < 0:
            raise ValueError("explicit DCGM group ID required")
        super().__init__(gpu_index, expected_uuid)
        self.group_id = group_id
        self._modified = False
        try:
            result = self._command("group", "-g", str(group_id), "-i", "-j")
            data = json.loads(result)
            # dcgmi 3.x represents all scalar output as {'value': ...}.
            def values(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key == "Entities" and isinstance(value, dict):
                            yield value.get("value")
                        yield from values(value)
                elif isinstance(node, list):
                    for value in node:
                        yield from values(value)
            if list(values(data)) != [f"GPU {gpu_index}"]:
                raise ValueError("DCGM group must contain exactly the calibrated GPU")
            initial = self.snapshot()
            self.initial = {"sm_mhz": initial["application_sm_mhz"],
                            "memory_mhz": initial["application_memory_mhz"],
                            "power_limit_w": initial["power_limit_mw"] / 1000}
            if any(v is None or v <= 0 for v in self.initial.values()):
                raise ValueError("cannot read initial DCGM device configuration")
            self.power_limit_w = self.initial["power_limit_w"]
            self.power_min_w = self._power_bounds()[0] / 1000
            self.power_max_w = self._power_bounds()[1] / 1000
        except BaseException:
            super().close()
            raise

    @staticmethod
    def _command(*args):
        result = subprocess.run(["dcgmi", *args], text=True, capture_output=True, timeout=10)
        if result.returncode:
            raise RuntimeError(f"DCGM command failed ({result.returncode}): {result.stdout} {result.stderr}")
        return result.stdout

    def _power_bounds(self):
        low, high = C.c_uint(), C.c_uint()
        self._check(self.lib.nvmlDeviceGetPowerManagementLimitConstraints(self.handle, C.byref(low), C.byref(high)))
        return low.value, high.value

    def snapshot(self):
        return {**super().snapshot(), "control_backend": "dcgm_application_clocks",
                "application_sm_mhz": self._read("nvmlDeviceGetApplicationsClock", C.c_uint(1)),
                "application_memory_mhz": self._read("nvmlDeviceGetApplicationsClock", C.c_uint(2)),
                "power_limit_mw": self._read("nvmlDeviceGetPowerManagementLimit")}

    def set(self, mhz, power_limit_w=None):
        if type(mhz) is not int or mhz <= 0:
            raise ValueError("positive integer SM application frequency required")
        cap = self.power_limit_w if power_limit_w is None else power_limit_w
        if not self.power_min_w <= cap <= self.power_max_w:
            raise ValueError("power cap outside device limits")
        self._modified = True  # partial DCGM failures must still restore.
        self._set_config(mhz, cap)
        actual = self.snapshot()
        if (actual["application_sm_mhz"] != mhz
                or actual["application_memory_mhz"] != self.initial["memory_mhz"]
                or abs(actual["power_limit_mw"] / 1000 - cap) > .5):
            raise RuntimeError(f"DCGM readback mismatch: {actual}")
        self.power_limit_w = cap

    def _set_config(self, mhz, cap):
        self._command("config", "-g", str(self.group_id), "--set",
                      "-a", f"{self.initial['memory_mhz']},{mhz}", "-P", f"{cap:g}")

    def reset(self):
        if self._modified:
            self.set(self.initial["sm_mhz"], self.initial["power_limit_w"])
            self._modified = False

    def close(self):
        try:
            if hasattr(self, "initial"):
                self.reset()
        finally:
            super().close()
