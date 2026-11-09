"""CPU-only loader checks without importing a ComfyUI or PyTorch installation."""
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from pf_quant_compress.archive import compress_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, array):
        self.array = array

    def numpy(self):
        return self.array

    def reshape(self, *shape):
        return Tensor(self.array.reshape(*shape))

    def view(self, dtype):
        return Tensor(self.array.view(dtype))


class LoaderTests(unittest.TestCase):
    def test_decodes_archive_with_four_worker_queue(self):
        torch = SimpleNamespace(int8=np.int8, uint8=np.uint8,
                                empty=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=dtype)))
        spec = spec_from_file_location("pfqc_light", ROOT / "loader.py", submodule_search_locations=[str(ROOT)])
        loader = module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": torch, spec.name: loader}):
            spec.loader.exec_module(loader)
            with TemporaryDirectory(dir=ROOT) as directory:
                source = Path(directory) / "source.safetensors"
                packed = Path(directory) / "packed.safetensors"
                tensors = {}
                for i in range(6):
                    name = f"layer{i}"
                    tensors[name + ".weight"] = np.tile(np.arange(-16, 16, dtype=np.int8), (16, 32))
                    tensors[name + ".weight_scale"] = np.ones((16, 1), dtype=np.float32)
                    tensors[name + ".comfy_quant"] = np.frombuffer(
                        b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":4}', dtype=np.uint8)
                save_file(tensors, source)
                report = compress_checkpoint(source, packed, target_bpw=8, max_error=0)
                self.assertEqual(report["encoded_weight_count"], 6)

                @contextmanager
                def stored_reader(path, archive, read_only):
                    with safe_open(path, framework="np") as reader:
                        yield lambda name: Tensor(reader.get_tensor(name))

                with patch.object(loader, "_stored_reader", stored_reader), patch.object(loader, "wait", wraps=loader.wait) as wait:
                    result, _ = loader.decode_checkpoint(packed)
                self.assertEqual(loader.DECODE_WORKERS, 4)
                self.assertTrue(wait.called)
                self.assertTrue(all(call.kwargs["return_when"] == loader.FIRST_COMPLETED for call in wait.call_args_list))
                for name, expected in tensors.items():
                    np.testing.assert_array_equal(result[name].array, expected)


if __name__ == "__main__":
    unittest.main()
