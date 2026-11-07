from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from pf_quant_compress import decoder as canonical_decoder
from pf_quant_compress.archive import METADATA_KEY, open_archive
from pf_quant_compress.cli import main as pf_compress
from safetensors import safe_open
from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("pfqc_decoder_test", ROOT / "loader.py", submodule_search_locations=[str(ROOT)])
loader = module_from_spec(spec)
sys.modules[spec.name] = loader
spec.loader.exec_module(loader)
decode_checkpoint = loader.decode_checkpoint


WEIGHT = "linear.weight"


class BundleTests(unittest.TestCase):
    def test_bundle_matches_canonical_sources(self):
        upstream = Path(canonical_decoder.__file__).parent
        for filename in ("decoder.py", "formats.py", "safetensors_io.py"):
            with self.subTest(filename=filename):
                bundled = ROOT / "_pfqc_decoder" / filename
                self.assertEqual(bundled.read_text(encoding="utf-8"), (upstream / filename).read_text(encoding="utf-8"))


def raw_tensor(tensor):
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def marker(**settings):
    return torch.tensor(list(json.dumps(settings).encode("utf-8")), dtype=torch.uint8)


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory(dir=ROOT)
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.source = cls.directory / "fixture.safetensors"
        cls.archive = cls.directory / "fixture.pfqc.safetensors"
        cls.restored = cls.directory / "restored.safetensors"
        tensors = {
            WEIGHT: torch.from_numpy(np.random.default_rng(817).integers(-16, 16, (24, 1024), dtype=np.int8)),
            "linear.weight_scale": torch.full((24, 1), 0.125, dtype=torch.float32),
            "linear.comfy_quant": marker(format="int8_tensorwise", convrot=True, convrot_groupsize=4),
            "other.weight": torch.tensor([[128, 7, 9]], dtype=torch.uint8),
            "other.weight_scale": torch.ones((1, 1), dtype=torch.float32),
            "other.comfy_quant": marker(format="int6_rowwise"),
            "native.bf16": torch.tensor([0x7FC1, -32768], dtype=torch.int16).view(torch.bfloat16),
            "native.fp32": torch.tensor([0x7FC00123, -2147483648], dtype=torch.int32).view(torch.float32),
            "empty": torch.empty((2, 0, 3), dtype=torch.float16),
            "scalar": torch.tensor(42, dtype=torch.int64),
        }
        save_file(tensors, cls.source, metadata={"origin": "fixture", "config": '{"test":true}'})
        if pf_compress([
            "compress", str(cls.source), "--target-bpw", "5.2", "--max-error", "0.003",
            "--output", str(cls.archive),
        ]):
            raise RuntimeError("Failed to generate the PFQC test fixture")

    def test_matches_standalone_decompress_bit_for_bit(self):
        self.assertEqual(pf_compress(["decompress", str(self.archive), "--output", str(self.restored)]), 0)
        with open_archive(self.archive) as archive:
            self.assertEqual(archive.records[WEIGHT]["codec"], "deflate")
            self.assertEqual(archive.records[WEIGHT]["storage"], "optimized_int8")

        for verify_checksums in (False, True):
            actual, metadata = decode_checkpoint(self.archive, verify_checksums=verify_checksums)
            with safe_open(self.restored, framework="pt", device="cpu") as expected:
                self.assertEqual(metadata, expected.metadata())
                self.assertEqual(set(actual), set(expected.keys()))
                for name, tensor in actual.items():
                    with self.subTest(name=name, verify_checksums=verify_checksums):
                        reference = expected.get_tensor(name)
                        self.assertEqual(tensor.device.type, "cpu")
                        self.assertEqual(tensor.shape, reference.shape)
                        self.assertEqual(tensor.dtype, reference.dtype)
                        self.assertEqual(raw_tensor(tensor), raw_tensor(reference))

    def test_default_decode_never_calls_sha256(self):
        bundled = sys.modules[spec.name + "._pfqc_decoder.decoder"]
        with (
            patch.object(loader, "sha256", side_effect=AssertionError("hashed a stored tensor")),
            patch.object(bundled, "sha256", side_effect=AssertionError("hashed a decoded stream")),
        ):
            tensors, _ = decode_checkpoint(self.archive)
        self.assertIn(WEIGHT, tensors)

    def test_missing_pfqc_manifest(self):
        with self.assertRaisesRegex(ValueError, "Missing PFQC compression metadata"):
            decode_checkpoint(self.source)

    def test_rejects_invalid_archives(self):
        original = self.archive.read_bytes()
        end = 8 + int.from_bytes(original[:8], "little")
        header = json.loads(original[8:end])
        payload = original[end:]
        for issue, message in (
            ("version", "Unsupported PFQC schema version"),
            ("codec", "Unsupported tensor codec"),
            ("tensor", "Missing storage tensor"),
            ("family", "Invalid family linkage"),
            ("stream", "DEFLATE"),
            ("stored", "Checksum mismatch"),
        ):
            with self.subTest(issue=issue):
                manifest = json.loads(header["__metadata__"][METADATA_KEY])
                record = next(item for item in manifest["tensors"] if item["name"] == WEIGHT)
                edited = bytearray(payload)
                if issue == "version":
                    manifest["schema_version"] = 2
                elif issue == "codec":
                    record["codec"] = "unknown"
                elif issue == "tensor":
                    record["name"] = "missing.weight"
                elif issue == "family":
                    record["family"] = "missing.weight"
                elif issue == "stream":
                    start, _ = header[WEIGHT]["data_offsets"]
                    edited[start] = 0b111
                else:
                    start, _ = header["native.bf16"]["data_offsets"]
                    edited[start] ^= 1
                modified = {**header, "__metadata__": {METADATA_KEY: json.dumps(manifest)}}
                raw_header = json.dumps(modified).encode()
                raw_header += b" " * (-len(raw_header) % 8)
                broken = self.directory / f"{issue}.pfqc.safetensors"
                broken.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header + edited)
                if issue == "stored":
                    decode_checkpoint(broken)
                    with self.assertRaisesRegex(ValueError, message):
                        decode_checkpoint(broken, verify_checksums=True)
                else:
                    for verify_checksums in (False, True):
                        with self.subTest(verify_checksums=verify_checksums), self.assertRaisesRegex(ValueError, message):
                            decode_checkpoint(broken, verify_checksums=verify_checksums)


if __name__ == "__main__":
    unittest.main()
