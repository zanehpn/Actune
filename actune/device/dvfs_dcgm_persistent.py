"""Persistent external DCGM connection; identical clock/readback contract."""
import ctypes as C

from .dvfs_dcgm import DCGMClocks


class PersistentDCGMClocks(DCGMClocks):
    def __init__(self, gpu_index, expected_uuid, group_id):
        self._connection = self._status = None
        self._native_enabled = False
        self.gpu_index = gpu_index
        super().__init__(gpu_index, expected_uuid, group_id)
        try:
            # Resolve installed bindings through the standard Python import path.
            import dcgm_agent
            import dcgm_structs
            import dcgmvalue
            import pydcgm
            self._agent, self._structs = dcgm_agent, dcgm_structs
            self._blank = dcgmvalue.DCGM_INT32_BLANK
            # Connect to the privileged existing hostengine, never start one.
            self._connection = pydcgm.DcgmHandle(ipAddress='127.0.0.1', timeoutMs=3000)
            self._status = self._agent.dcgmStatusCreate()
            self._native_enabled = True
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        return {**super().snapshot(), 'control_backend': 'dcgm_persistent_application_clocks'}

    def _set_config(self, mhz, cap):
        if not self._native_enabled:
            return super()._set_config(mhz, cap)
        if float(cap) != int(cap):
            raise ValueError('DCGM power cap requires whole watts')
        config = self._structs.c_dcgmDeviceConfig_v1()
        config.gpuId = self.gpu_index
        # Leave unrelated settings alone; zero means a real setting in DCGM.
        config.mEccMode = config.mComputeMode = self._blank
        config.mPerfState.syncBoost = self._blank
        config.mPerfState.targetClocks.memClock = self.initial['memory_mhz']
        config.mPerfState.targetClocks.smClock = mhz
        config.mPowerLimit.type = self._structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL
        config.mPowerLimit.val = int(cap)
        self._agent.dcgmStatusClear(self._status)
        self._agent.dcgmConfigSet(self._connection.handle, C.c_void_p(self.group_id), config, self._status)
        if self._agent.dcgmStatusGetCount(self._status):
            raise RuntimeError('DCGM configuration error: ' + str(self._agent.dcgmStatusPopError(self._status)))

    def reset(self):
        try:
            super().reset()
        except Exception:
            # Restore via the independently tested CLI if the connection fails.
            self._native_enabled = False
            super().reset()

    def close(self):
        try:
            super().close()
        finally:
            try:
                if self._status is not None:
                    self._agent.dcgmStatusDestroy(self._status)
                    self._status = None
            finally:
                if self._connection is not None:
                    self._connection.Shutdown()
                    self._connection = None
