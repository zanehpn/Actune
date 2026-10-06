"""Opt-in NVML clock backend. Core clocks only; no power/memory-clock edits."""

from __future__ import annotations

import ctypes as C
import fcntl
import os
from pathlib import Path
import tempfile


def _lease_path(gpu_uuid):
    """Use a relative path to the shared system-temp lease across workspaces."""
    location = Path(tempfile.gettempdir()) / f"bits-dvfs-{gpu_uuid}.lock"
    return Path(os.path.relpath(location))


class NVMLClocks:
    dry_run = False

    def __init__(self, gpu_index: int, expected_uuid: str):
        if type(gpu_index) is not int or gpu_index < 0 or not expected_uuid.startswith("GPU-"):
            raise ValueError("physical GPU index and expected UUID are required")
        self.lib = C.CDLL("libnvidia-ml.so.1")
        self._check(self.lib.nvmlInit_v2())
        self.handle = C.c_void_p()
        self._lease = None
        try:
            self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(C.c_uint(gpu_index), C.byref(self.handle)))
            uuid = C.create_string_buffer(128)
            self._check(self.lib.nvmlDeviceGetUUID(self.handle, uuid, C.c_uint(len(uuid))))
            self.uuid = uuid.value.decode()
            if self.uuid != expected_uuid:
                raise ValueError(f"GPU identity mismatch: got {self.uuid}")
            self._lease = _lease_path(self.uuid).open("a")
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status):
        if status != 0:
            raise RuntimeError(f"NVML returned {status}; check device support and clock-control permission")

    def _read(self, name, *args, wide=False):
        value = C.c_ulonglong() if wide else C.c_uint()
        api = getattr(self.lib, name, None)
        if api is None:
            return None
        status = api(self.handle, *args, C.byref(value))
        return value.value if status == 0 else None

    def snapshot(self):
        class Utilization(C.Structure):
            _fields_ = [("gpu", C.c_uint), ("memory", C.c_uint)]
        utilization = Utilization()
        status = self.lib.nvmlDeviceGetUtilizationRates(self.handle, C.byref(utilization))
        return {
            "dry_run": False, "gpu_uuid": self.uuid,
            "sm_mhz": self._read("nvmlDeviceGetClockInfo", C.c_uint(1)),
            "memory_mhz": self._read("nvmlDeviceGetClockInfo", C.c_uint(2)),
            "pstate": self._read("nvmlDeviceGetPerformanceState"),
            "energy_mj": self._read("nvmlDeviceGetTotalEnergyConsumption", wide=True),
            "power_mw": self._read("nvmlDeviceGetPowerUsage"),
            "temperature_c": self._read("nvmlDeviceGetTemperature", C.c_uint(0)),
            "clock_event_reasons": self._read("nvmlDeviceGetCurrentClocksThrottleReasons", wide=True),
            "gpu_utilization_pct": utilization.gpu if status == 0 else None,
        }

    def set(self, mhz: int):
        self._check(self.lib.nvmlDeviceSetGpuLockedClocks(self.handle, C.c_uint(mhz), C.c_uint(mhz)))

    def reset(self):
        self._check(self.lib.nvmlDeviceResetGpuLockedClocks(self.handle))

    def close(self):
        if self._lease is not None:
            self._lease.close()
            self._lease = None
        self.lib.nvmlShutdown()
