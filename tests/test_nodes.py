from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
NODE_ID = "LoadDiffusionModelPFQC"
REGISTERED = "moss/example.pfqc.safetensors"


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.folders = SimpleNamespace(
            get_filename_list=Mock(return_value=[REGISTERED, "moss.safetensors", "other.ckpt"]),
            get_full_path_or_raise=Mock(return_value="/models/" + REGISTERED),
        )
        spec = spec_from_file_location("pfqc_test_node", ROOT / "__init__.py")
        package = module_from_spec(spec)
        modules = patch.dict(sys.modules, {"folder_paths": self.folders, spec.name: package})
        modules.start()
        self.addCleanup(modules.stop)
        spec.loader.exec_module(package)
        self.node = package.NODE_CLASS_MAPPINGS[NODE_ID]()
        self.loader = sys.modules[spec.name + ".loader"]
        self.assertEqual(package.NODE_DISPLAY_NAME_MAPPINGS[NODE_ID], "Load Diffusion Model (PFQC)")

    def test_discovery_and_resolution_use_comfy_folders(self):
        self.assertEqual(self.node.INPUT_TYPES()["required"]["model_name"], ([REGISTERED],))
        with patch("pfqc_test_node.nodes.load_diffusion_model") as load:
            self.assertEqual(self.node.load(REGISTERED), (load.return_value,))
            load.assert_called_once_with("/models/" + REGISTERED)
        self.folders.get_full_path_or_raise.assert_called_once_with("diffusion_models", REGISTERED)

    def test_rejects_non_archive_and_unregistered_paths(self):
        for name in ("moss.safetensors", "../moss.pfqc.safetensors", "/tmp/moss.pfqc.safetensors"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.node.load(name)
        self.folders.get_full_path_or_raise.assert_not_called()

    def test_passes_state_dict_and_metadata_to_native_loader_and_can_reload(self):
        path = "/models/" + REGISTERED
        state_dict = {"layer.weight": object()}
        metadata = {"model": "MOSS"}
        native = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace())
        comfy = ModuleType("comfy")
        comfy.sd = ModuleType("comfy.sd")
        comfy.sd.load_diffusion_model_state_dict = native
        with (
            patch.dict(sys.modules, {"comfy": comfy, "comfy.sd": comfy.sd}),
            patch.object(self.loader, "decode_checkpoint", return_value=(state_dict, metadata)) as decode,
        ):
            model = self.loader.load_diffusion_model(path)
            native.assert_called_once_with(state_dict, metadata=metadata, disable_dynamic=False)
            decode.assert_called_once_with(path)
            factory, args = model.cached_patcher_init
            self.assertIsNot(factory(*args, disable_dynamic=True), model)
            native.assert_called_with(state_dict, metadata=metadata, disable_dynamic=True)
            self.assertEqual(decode.call_count, 2)
