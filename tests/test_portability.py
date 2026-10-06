import fcntl
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from actune.device.dvfs_dcgm import DCGMClocks
from actune.device.dvfs_nvml import _lease_path


class DevicePortabilityTests(unittest.TestCase):
    def test_relative_lease_excludes_another_workspace(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for name in ("temporary", "workspace_one", "workspace_two"):
                (base / name).mkdir()
            try:
                with patch("actune.device.dvfs_nvml.tempfile.gettempdir", return_value=str(base / "temporary")):
                    os.chdir(base / "workspace_one")
                    first = _lease_path("GPU-test")
                    self.assertFalse(first.is_absolute())
                    identity = first.resolve()
                    with first.open("a") as owner:
                        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        os.chdir(base / "workspace_two")
                        second = _lease_path("GPU-test")
                        self.assertFalse(second.is_absolute())
                        self.assertEqual(second.resolve(), identity)
                        with second.open("a") as contender:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.chdir(previous)

    def test_dcgm_executable_is_resolved_by_environment(self):
        with patch("actune.device.dvfs_dcgm.subprocess.run",
                   return_value=SimpleNamespace(returncode=0, stdout="ok", stderr="")) as run:
            self.assertEqual(DCGMClocks._command("group", "-l"), "ok")
        command = run.call_args.args[0]
        self.assertEqual(command, ["dcgmi", "group", "-l"])
        self.assertFalse(Path(command[0]).is_absolute())


if __name__ == "__main__":
    unittest.main()
