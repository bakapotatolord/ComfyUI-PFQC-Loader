# ComfyUI PFQC Loader

**Load Diffusion Model (PFQC)** reconstructs a diffusion checkpoint from a `.pfqc.safetensors` archive and hands it to ComfyUI as a standard `MODEL`. PFQC is PotatoForge's storage format for compressed quantized checkpoints. PFQC files are valid safetensors containers, but their encoded weights require reconstruction before an ordinary ComfyUI model loader can use them.

## Compatibility

The loader supports PFQC schema **1**, written by `potatoforge-quant-compress` **0.1.0**. It uses the compressor's decoder for raw DEFLATE INT8 ConvRot weights and passes stored tensors through with their original bytes, including FP32 row scales, `.comfy_quant` markers, and higher-precision tensors. ComfyUI determines whether it can execute the resulting model and quantization formats. The primary model is MOSS Image SFT INT8 ConvRot.

The loader uses the native state-dictionary loading and patcher APIs in ComfyUI 0.43.0. Use a ComfyUI build with native support for your checkpoint's quantization and model architecture.

## Install

1. Put this repository in `ComfyUI/custom_nodes/ComfyUI-PFQC-Loader`.
2. Install `potatoforge-quant-compress` from a source checkout of **v0.1.0** (commit `a560599abcfd7a54e83aaa33fa9c2c3a383cd810`) into the Python environment that runs ComfyUI. The decoder requires Python 3.11 or later, NumPy 1.26 or later, and safetensors 0.8.0 or later. PyTorch comes from ComfyUI.

   ```sh
   python -m pip install "/path/to/potatoforge-quant-compress"
   ```

   For Windows portable, use `python_embeded\python.exe` from the portable root in place of `python`. If the prerequisites and setuptools are already installed, use `--no-deps --no-build-isolation` to install from the local checkout offline.

3. Restart ComfyUI. Decoding itself runs offline and needs only the archive.

## Use with MOSS Image SFT

1. Place `moss_image_sft_int8_convrot.pfqc.safetensors` in `ComfyUI/models/diffusion_models` or a `diffusion_models` directory registered in `extra_model_paths.yaml`.
2. In an existing MOSS workflow, replace **Load Diffusion Model** with **Load Diffusion Model (PFQC)** under **loaders/PotatoForge**.
3. Select the archive and reconnect the `MODEL` output. Keep your MOSS text encoder, adapter, VAE, and sampler connections.

PFQC can change quantized weight codes and scales during compression. For a loader comparison, use the ordinary checkpoint produced by `pf-compress decompress` as the reference.

## Loading and memory

The loader validates the manifest and streams encoded weights into their final CPU tensor buffers, one tensor at a time. Stored tensors are read through safetensors. ComfyUI receives a complete state dictionary, so leave enough system RAM for the reconstructed weights and the rest of the workflow. The archive is never expanded to a temporary checkpoint during loading.

Decoding happens when the model object is created. Denoising, LoRA patching, and offloading use ComfyUI's native model and patcher. If ComfyUI needs a fresh patcher, the reload recipe opens and decodes the archive again.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Empty model list | Confirm the exact `.pfqc.safetensors` suffix and that the file is in a registered `diffusion_models` directory. |
| Compressor dependency missing | Install the 0.1.0 compressor in the interpreter ComfyUI uses. |
| Missing PFQC metadata | Select a PFQC archive. Renaming an ordinary safetensors file does not make it a PFQC archive. |
| Unsupported schema or codec | Check the archive's writer version. This loader supports schema 1 with `stored` and `deflate` codecs. |
| Checksum or stream error | Run `pf-compress verify` on the archive to check its integrity. |
| ComfyUI cannot detect the model | Confirm the archive was made from a diffusion checkpoint supported by your ComfyUI build. |

## Tests

Run `python -m unittest discover -s tests` with the decoder and PyTorch installed. The tests build a small archive with the actual compressor CLI and compare the loaded CPU tensors to its standalone decompressed file.

`tests/moss_integration.py` exercises the installed ComfyUI APIs with a complete MOSS checkpoint. Use ComfyUI's Python and see `--help` for the archive/reference comparison, model-loading, and generation options. The generation benchmark uses the local MOSS 4B text encoder and adapter, Flux 2 VAE, and MOSS four-step speed LoRA. [MOSS validation and benchmark](docs/moss-validation.md)
