"""Protect existing data when Docker or model configuration changes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("dev_runtime", Path(__file__).parents[1] / "dev_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class ExistingDataGuards(unittest.TestCase):
    def setUp(self):
        self.expected = {"docker_context": "desktop-linux", "volumes": {"original-volume": "old-date"},
                         "configuration": {"model": "original"}, "database_cluster_id": "original-cluster"}

    def test_different_docker_context_cannot_start(self):
        with patch.object(runtime, "load_identity", return_value=self.expected), patch.object(runtime, "docker", return_value="another-engine"):
            with self.assertRaisesRegex(RuntimeError, "context changed"):
                runtime.verify_storage()

    def test_recreated_volume_with_same_name_cannot_start(self):
        with patch.object(runtime, "load_identity", return_value=self.expected), patch.object(runtime, "docker", return_value="desktop-linux"), \
             patch.object(runtime, "volumes", return_value={"original-volume": "new-date"}):
            with self.assertRaisesRegex(RuntimeError, "missing or were replaced"):
                runtime.verify_storage()

    def test_changed_embedding_contract_cannot_start(self):
        with patch.object(runtime, "load_identity", return_value=self.expected), patch.object(runtime, "docker", return_value="desktop-linux"), \
             patch.object(runtime, "volumes", return_value=self.expected["volumes"]), patch.object(runtime, "configuration", return_value={"model": "other"}):
            with self.assertRaisesRegex(RuntimeError, "contract changed"):
                runtime.verify_storage()

    def test_replaced_database_in_existing_volume_cannot_start(self):
        with patch.object(runtime, "verify_storage", return_value=self.expected), \
             patch.object(runtime, "database_state", return_value={"cluster_id": "empty-new-cluster"}):
            with self.assertRaisesRegex(RuntimeError, "cluster identity changed"):
                runtime.verify_data()

    def test_cuda_failure_does_not_fall_back_to_cpu(self):
        cfg = SimpleNamespace(local_embedding_device="cuda", local_embedding_preload=True)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(runtime, "settings", return_value=cfg), patch.dict("sys.modules", {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                runtime.check_gpu()


if __name__ == "__main__":
    unittest.main()
