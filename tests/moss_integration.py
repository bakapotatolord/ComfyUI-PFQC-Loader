"""Opt-in MOSS smoke test and benchmark. Run with ComfyUI's Python."""

import argparse
import asyncio
import gc
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NODE_ID = "LoadDiffusionModelPFQC"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy", type=Path, help="Installed ComfyUI directory")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--verify-reference", type=Path, help="Only compare decoded tensors to this checkpoint")
    parser.add_argument("--benchmark-checksums", action="store_true", help="Time CPU reconstruction with and without SHA-256")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--image", type=Path, help="Save the final generated image as PNG")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--legacy-offload", action="store_true")
    parser.add_argument("--disable-fast-disk", action="store_true", help="Compare ComfyUI's RAM offload policy")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must include one warmup and at least one warmed run")
    args.comfy = args.comfy.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.benchmark_checksums:
        benchmark_checksums(args.checkpoint, args.runs)
        return
    if args.verify_reference:
        verify_reference(args.checkpoint, args.verify_reference)
        return

    sys.path.insert(0, str(args.comfy))
    sys.argv = ["comfy", "--disable-all-custom-nodes", "--offline", "--deterministic"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.legacy_offload:
        sys.argv.append("--disable-dynamic-vram")
    if args.disable_fast_disk:
        sys.argv.append("--disable-fast-disk")

    import main as comfy_main
    import comfy.model_management as mm
    from comfy.utils import set_progress_bar_global_hook
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage
    from comfy_kitchen.registry import registry
    import folder_paths
    import nodes
    import psutil
    import torch

    assert comfy_main.nodes is nodes
    assert torch.cuda.is_available(), "This benchmark requires CUDA"
    assert asyncio.run(nodes.load_custom_node(str(ROOT))), "PFQC node registration failed"
    node = nodes.NODE_CLASS_MAPPINGS[NODE_ID]()
    loader = importlib.import_module(type(node).__module__.rsplit(".", 1)[0] + ".loader")
    name = args.checkpoint.name
    folder_paths.add_model_folder_path("diffusion_models", str(args.checkpoint.parent), is_default=True)
    is_pfqc = name.endswith(".pfqc.safetensors")
    assert (name in node.INPUT_TYPES()["required"]["model_name"][0]) == is_pfqc
    process = psutil.Process()

    def load_model():
        return node.load(name)[0] if is_pfqc else nodes.UNETLoader().load_unet(name, "default")[0]

    def memory():
        info = process.memory_info()
        return {
            "rss_mib": info.rss / 2**20,
            "peak_rss_mib": getattr(info, "peak_wset", info.rss) / 2**20,
            "peak_torch_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        }

    with patch.object(loader, "decode_checkpoint", wraps=loader.decode_checkpoint) as decode:
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        model = load_model()
        load_time = perf_counter() - started
        assert decode.call_count == int(is_pfqc)
        diffusion = model.model.diffusion_model
        weight = diffusion.layers[0].proj.qkv.weight
        assert type(diffusion).__name__ == "MOSSImage"
        assert weight._qdata.dtype == torch.int8 and weight._params.convrot
        assert weight._params.scale.dtype == torch.float32
        report = {
            "checkpoint": name,
            "bytes": args.checkpoint.stat().st_size,
            "load_s": load_time,
            "load_memory": memory(),
            "patcher": type(model).__name__,
            "offload_device": str(model.offload_device),
            "fast_disk": model.fast_disk,
            "layout": weight.layout_cls.__name__,
            "convrot_group_size": weight._params.convrot_groupsize,
        }
        print(json.dumps(report), flush=True)
        if args.load_only:
            return
        del diffusion, weight

        assert asyncio.run(nodes.load_custom_node(str(args.comfy / "custom_nodes/moss-image-smaller-encoder")))
        adapter = nodes.NODE_CLASS_MAPPINGS["MOSSSmallerEncoderAdapter"]()
        clip, = nodes.CLIPLoader().load_clip("Qwen3.5-4B_int8_convrot.safetensors", "moss_image")
        adapter_name = str(Path("4B") / "moss_qwen35_4b_to_9b_1600_train_mlp2048.safetensors")
        clip, = adapter.apply(clip, adapter_name)
        positive, = nodes.CLIPTextEncode().encode(clip, "a latina boxer woman has large eyes and wide smile, jumping")
        negative, = nodes.ConditioningZeroOut().zero_out(positive)
        vae, = nodes.VAELoader().load_vae("flux2-vae_small.safetensors")
        del clip
        latent = EmptyFlux2LatentImage.execute(512, 768)[0]
        lora_name = str(Path("moss") / "moss_image_sft_4step_v2.0.safetensors")
        patched, = nodes.LoraLoaderModelOnly().load_lora_model_only(model, lora_name, 1.0)
        assert patched is not model and patched.clone_base_uuid == model.clone_base_uuid and patched.patches

        dispatch = set()
        get_implementation = registry.get_implementation

        def record_dispatch(op, *op_args, **kwargs):
            impl = get_implementation(op, *op_args, **kwargs)
            if "int8" in op:
                dispatch.add((op, impl.__module__))
            return impl

        report["runs"] = []
        for run in range(args.runs):
            steps = []
            torch.cuda.reset_peak_memory_stats()
            set_progress_bar_global_hook(lambda *args, **kwargs: steps.append(perf_counter()))
            started = perf_counter()
            if run == 0:
                with patch.object(registry, "get_implementation", record_dispatch):
                    sampled = nodes.common_ksampler(patched, 433126007329043, 4, 1, "euler", "simple",
                                                    positive, negative, latent)[0]
            else:
                sampled = nodes.common_ksampler(patched, 433126007329043, 4, 1, "euler", "simple",
                                                positive, negative, latent)[0]
            torch.cuda.synchronize()
            sample_end = perf_counter()
            set_progress_bar_global_hook(None)
            image, = nodes.VAEDecode().decode(vae, sampled)
            pixels = image.cpu().contiguous().numpy()
            assert torch.isfinite(image).all()
            assert len(steps) == 4
            assert decode.call_count == int(is_pfqc), "Sampling unexpectedly decoded the checkpoint"
            run_report = {
                "run": run + 1,
                "sample_s": sample_end - started,
                "post_first_step_s": (steps[-1] - steps[0]) / (len(steps) - 1),
                "generation_s": perf_counter() - started,
                "image_sha256": sha256(pixels).hexdigest(),
                "latent_sha256": sha256(sampled["samples"].cpu().contiguous().numpy()).hexdigest(),
                "model_loaded_mib": patched.loaded_size() / 2**20,
                **memory(),
            }
            report["runs"].append(run_report)
            print(json.dumps(run_report), flush=True)

        if args.image:
            from PIL import Image

            Image.fromarray((pixels[0] * 255).clip(0, 255).astype("uint8")).save(args.image)

        report["int8_dispatch"] = sorted(dispatch)
        assert ("int8_linear", "comfy_kitchen.backends.cuda") in dispatch
        mm.unload_model_and_clones(patched)
        assert all(loaded.clone_base_uuid != model.clone_base_uuid for loaded in mm.loaded_models())
        if is_pfqc:
            factory, factory_args = model.cached_patcher_init
            del patched, model
            gc.collect()
            started = perf_counter()
            reloaded = factory(*factory_args)
            report["reload_s"] = perf_counter() - started
            assert decode.call_count == 2 and reloaded.model.diffusion_model.layers[0].proj.qkv.weight._params.convrot
        print(json.dumps(report, indent=2), flush=True)


def import_decoder():
    spec = importlib.util.spec_from_file_location(
        "pfqc_decode_check", ROOT / "loader.py", submodule_search_locations=[str(ROOT)],
    )
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    return loader


def benchmark_checksums(path, runs):
    loader = import_decoder()
    timings = {True: [], False: []}
    for run in range(runs):
        order = (True, False) if run % 2 == 0 else (False, True)
        for verify_checksums in order:
            started = perf_counter()
            tensors, _ = loader.decode_checkpoint(path, verify_checksums=verify_checksums)
            elapsed = perf_counter() - started
            timings[verify_checksums].append(elapsed)
            print(json.dumps({
                "pair": run + 1,
                "verify_checksums": verify_checksums,
                "decode_s": elapsed,
                "tensor_count": len(tensors),
            }), flush=True)
            del tensors
            gc.collect()
    print(json.dumps({
        "verified_median_s": median(timings[True]),
        "unchecked_median_s": median(timings[False]),
    }), flush=True)


def verify_reference(path, reference):
    from safetensors import safe_open
    import torch

    loader = import_decoder()
    actual, metadata = loader.decode_checkpoint(path)
    with safe_open(reference, framework="pt", device="cpu") as expected:
        assert metadata == expected.metadata()
        assert set(actual) == set(expected.keys())
        for name, tensor in actual.items():
            other = expected.get_tensor(name)
            assert (tensor.shape, tensor.dtype) == (other.shape, other.dtype), name
            assert torch.equal(tensor.reshape(-1).view(torch.uint8), other.reshape(-1).view(torch.uint8)), name
    print(f"{len(actual)} tensors and checkpoint metadata match the standalone reference bit-for-bit.")


if __name__ == "__main__":
    main()
