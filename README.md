# ComfyUI PFQC Loader

**Load Diffusion Model (PFQC)** reconstructs a diffusion checkpoint from a `.pfqc.safetensors` archive and hands it to ComfyUI as a standard `MODEL`. PFQC is PotatoForge's storage format for compressed quantized checkpoints. PFQC files are valid safetensors containers, but their encoded weights require reconstruction before an ordinary ComfyUI model loader can use them.

## Compatibility

The loader supports PFQC schemas **1 and 2**. It bundles the compressor's decoder for raw DEFLATE INT8 ConvRot weights and passes stored tensors through with their original bytes, including FP32 row scales, `.comfy_quant` markers, and higher-precision tensors. ComfyUI determines whether it can execute the resulting model and quantization formats. The primary model is MOSS Image SFT INT8 ConvRot.

The compressor writes schema v2 when an experimental fit selects a multiplier above 2×. Update the installed loader and restart ComfyUI before loading v2 archives. This decoder also opens experimental v1 archives created with earlier wide-grid builds.

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

Routine loading verifies every tensor's SHA-256 and checks archive structure, tensor sizes, and complete DEFLATE streams. Four workers decode independent weights into their final CPU buffers. The native path limits each worker to one compressed input below 64 MiB, for up to 256 MiB of compressed buffers in flight. Larger weights stream through zlib serially.

The bundle includes the official libdeflate **1.26 Windows x86-64** build and license. Windows x64 Python uses the bundled DLL when its digest matches. A missing or unsupported DLL selects zlib. Windows 10 or later can use CNG for SHA-256, with `hashlib` available on all supported platforms. The node loads its native libraries locally. [Upstream release and asset digest](https://github.com/ebiggers/libdeflate/releases/expanded_assets/v1.26)

Decoding happens when the model object is created. Denoising, LoRA patching, and offloading use ComfyUI's native model and patcher. If ComfyUI needs a fresh patcher, the reload recipe opens and decodes the archive again.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Empty model list | Confirm the exact `.pfqc.safetensors` suffix and that the file is in a registered `diffusion_models` directory. |
| Missing PFQC metadata | Select a PFQC archive. Renaming an ordinary safetensors file does not make it a PFQC archive. |
| Unsupported schema or codec | Check the archive's writer version. This loader supports schemas 1 and 2 with `stored` and `deflate` codecs. |
| Checksum or stream error | Run `pf-compress verify` on the archive to check its integrity. |
| ComfyUI cannot detect the model | Confirm the archive was made from a diffusion checkpoint supported by your ComfyUI build. |

## Tests

Run `python -m unittest discover -s tests` with PyTorch and the pinned compressor source available on `PYTHONPATH`. The tests use the compressor CLI to generate a small archive and compare the bundled decoder's CPU tensors to the standalone decompressed file.

`tests/moss_integration.py` exercises the installed ComfyUI APIs with a complete MOSS checkpoint. Use ComfyUI's Python and see `--help` for the archive/reference comparison, model-loading, and generation options. The generation benchmark uses the local MOSS 4B text encoder and adapter, Flux 2 VAE, and MOSS four-step speed LoRA. [MOSS validation and benchmark](docs/moss-validation.md)

## Updating the bundled decoder

The bundle comes from `potatoforge-quant-compress` commit `985af01454e3a8029ba852cc4fb20c985691482d`. Make decoder changes there, then copy `decoder.py`, `formats.py`, `safetensors_io.py`, `native.py`, and `_vendor/libdeflate-win-x64` from its `src/pf_quant_compress` directory into `_pfqc_decoder` and update the pin. The unit tests check that the copies match the canonical source.
