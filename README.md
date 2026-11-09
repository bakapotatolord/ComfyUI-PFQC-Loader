# ComfyUI PFQC Loader

**Load Diffusion Model (PFQC)** reconstructs a diffusion checkpoint from a `.pfqc.safetensors` archive and hands it to ComfyUI as a standard `MODEL`. PFQC is PotatoForge's storage format for compressed quantized checkpoints. PFQC files are valid safetensors containers, but their encoded weights require reconstruction before an ordinary ComfyUI model loader can use them.

## Compatibility

The loader supports PFQC format **v1** with Zstandard INT8 ConvRot weights. Stored tensors retain their original bytes, including FP32 scales, `.comfy_quant` markers, and higher-precision tensors. ComfyUI determines whether it can execute the resulting model and quantization formats. Experimental DEFLATE and research archives are not supported.

The loader uses the native state-dictionary loading and patcher APIs in ComfyUI 0.43.0. Use a ComfyUI build with native support for your checkpoint's quantization and model architecture.

## Install

1. Put this repository in `ComfyUI/custom_nodes/ComfyUI-PFQC-Loader`.
2. Restart ComfyUI. The decoder ships with the node and runs offline. It uses ComfyUI's Python 3.11 or later, NumPy 1.26 or later, safetensors 0.8.0 or later, and PyTorch.

## Use with MOSS Image SFT

1. Place `moss_image_sft_int8_convrot.pfqc.safetensors` in `ComfyUI/models/diffusion_models` or a `diffusion_models` directory registered in `extra_model_paths.yaml`.
2. In an existing MOSS workflow, replace **Load Diffusion Model** with **Load Diffusion Model (PFQC)** under **loaders/PotatoForge**.
3. Select the archive and reconnect the `MODEL` output. Keep your MOSS text encoder, adapter, VAE, and sampler connections.

PFQC can change quantized weight codes and scales during compression. For a loader comparison, use the ordinary checkpoint produced by `pf-compress decompress` as the reference.

## Loading and memory

The loader validates the manifest and decodes encoded weights into their final CPU tensor buffers. ComfyUI receives a complete state dictionary, so leave enough system RAM for the reconstructed weights and the rest of the workflow. The archive is never expanded to a temporary checkpoint during loading.

On Windows with supported ComfyUI DynamicVRAM, unchanged tensors use ComfyUI's read-only file mapping. Tensor storage keeps the mapping alive. This avoids a copy-on-write commit charge for the archive. MOSS's peak physical working set was essentially unchanged because the reconstructed INT8 weights still require private CPU backing. The loader uses the safetensors path when read-only mapping is unavailable or ComfyUI requests a legacy patcher.

The read-only patcher supports normal ComfyUI LoRA loading and blocks explicit in-place patch requests on mapped parameters. Third-party nodes that modify tensor storage directly bypass this guard and must use a writable loader path. A legacy ComfyUI session with `--disable-dynamic-vram` uses the safetensors fallback.

Routine loading verifies every tensor's SHA-256 and checks archive structure, tensor sizes, and complete Zstandard frames. Four workers decode independent weights directly into their final CPU buffers with bounded compressed input chunks. The decoder limits frame windows to 128 MiB.

The bundle includes the official [Zstandard 1.5.7 Windows x64 library](https://github.com/facebook/zstd/releases/tag/v1.5.7) and license. The DLL is required and its checksum is verified before loading. Windows CNG accelerates SHA-256 where available; `hashlib` is the fallback.

Decoding happens when the model object is created. Denoising, LoRA patching, and offloading use ComfyUI's native model and patcher. If ComfyUI needs a fresh patcher, the reload recipe opens and decodes the archive again.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Empty model list | Confirm the exact `.pfqc.safetensors` suffix and that the file is in a registered `diffusion_models` directory. |
| Missing PFQC metadata | Select a PFQC archive. Renaming an ordinary safetensors file does not make it a PFQC archive. |
| Unsupported schema or codec | Check the archive's writer version. This loader supports format v1 with `stored` and `zstd` records. |
| Checksum or stream error | Run `pf-compress verify` on the archive to check its integrity. |
| ComfyUI cannot detect the model | Confirm the archive was made from a diffusion checkpoint supported by your ComfyUI build. |

## Tests

Run `python -m unittest discover -s tests` with PyTorch and the pinned compressor source available on `PYTHONPATH`. The tests use the compressor CLI to generate a small archive and compare the bundled decoder's CPU tensors to the standalone decompressed file.

`tests/moss_integration.py` exercises the installed ComfyUI APIs with a complete MOSS checkpoint. Use ComfyUI's Python and see `--help` for the archive/reference comparison, model-loading, and generation options. The generation benchmark uses the local MOSS 4B text encoder and adapter, Flux 2 VAE, and MOSS four-step speed LoRA. [MOSS validation and benchmark](docs/moss-validation.md)

## Updating the bundled decoder

The compressor repository is the canonical source. Copy `decoder.py`, `formats.py`, `safetensors_io.py`, `native.py`, `zstd.py`, and `_vendor/zstd-win-x64` from its `src/pf_quant_compress` directory into `_pfqc_decoder`. The unit tests check that the copies match.
