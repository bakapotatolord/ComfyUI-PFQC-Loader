# MOSS INT8 ConvRot validation

## Archive and native loading

The compressor writes schema 1 to the safetensors metadata key `potatoforge_quant_compress.manifest`. The JSON manifest identifies the format as `potatoforge_quant_compress` and records the compressor version, original safetensors metadata, logical tensor names, shapes, dtypes, decoded sizes, SHA-256 hashes, and quantization-family links.

The physical tensor keys match the original names. Encoded weights store a raw DEFLATE stream as a U8 vector. The decoder restores that stream as a row-major signed INT8 matrix. Stored tensors keep their safetensors dtype and shape. These include FP32 row scales, U8 `.comfy_quant` markers with ConvRot settings, higher-precision tensors, and unsupported families carried through by the compressor. Where compression fits a weight and scale pair, the archive stores the final codes and final FP32 scales.

The loader bundles the compressor's archive reader. The shared decoder validates the manifest, physical tensor references, quantization families, bounded streams, and decoded sizes. Stored tensors are mapped by safetensors. ComfyUI verifies SHA-256 by default. The original BF16 model and external compression settings are unnecessary.

ComfyUI 0.43.0 accepts the completed dictionary through `comfy.sd.load_diffusion_model_state_dict(sd, model_options={}, metadata=None, disable_dynamic=False)`. The loader passes the original metadata and registers the same style of `cached_patcher_init` factory as the built-in loader. Encoded tensors occupy RAM after reconstruction. Ordinary checkpoints can keep their INT8 weights file-backed and use ComfyUI's fast-disk path. The PFQC model still uses ComfyUI's native CPU/VRAM placement and dynamic offloading.

## Reproduce the fixture and runs

The fixture was created with compressor v0.1.0, commit `a560599abcfd7a54e83aaa33fa9c2c3a383cd810`, NumPy 2.4.6, and safetensors 0.8.0. The MOSS file has 168 eligible INT8 weights. The settings below exercise real DEFLATE reconstruction with no weight distortion. The synthetic fixture exercises optimized codes and changed scales separately.

Run from the loader repository with `$original` set to the path to `moss_image_sft_int8_convrot.safetensors` and `$compressorSrc` set to the compressor checkout's `src` path. The archive and reference belong in the ignored `.fixtures` directory.

```powershell
New-Item -ItemType Directory -Path .fixtures -Force | Out-Null
$env:PYTHONPATH = $compressorSrc
python -B -m pf_quant_compress.cli compress $original --target-bpw 8 --max-error 0 --output .fixtures/moss_image_sft_int8_convrot.pfqc.safetensors --report .fixtures/compress.json --verbose
python -B -m pf_quant_compress.cli decompress .fixtures/moss_image_sft_int8_convrot.pfqc.safetensors --output .fixtures/moss_image_sft_int8_convrot_restored.safetensors --report .fixtures/decompress.json
```

Run the harness once per checkpoint with ComfyUI's Python, passing the same installed ComfyUI root to each run. To compare all state-dict bytes and metadata, run the archive again with `--verify-reference .fixtures/moss_image_sft_int8_convrot_restored.safetensors`.

```powershell
& $comfyPython -B tests/moss_integration.py $comfyRoot $original
& $comfyPython -B tests/moss_integration.py $comfyRoot .fixtures/moss_image_sft_int8_convrot_restored.safetensors --image .fixtures/reference.png
& $comfyPython -B tests/moss_integration.py $comfyRoot .fixtures/moss_image_sft_int8_convrot.pfqc.safetensors --image .fixtures/pfqc.png
```

The harness uses the local MOSS workflow's Qwen 3.5 4B INT8 ConvRot encoder, 4B-to-9B adapter, Flux 2 small VAE, and MOSS four-step speed LoRA at strength 1. It disables the workflow's stylistic LoRAs and fixes resolution to 512 × 768, seed to 433126007329043, and sampling to four Euler steps with the simple scheduler, CFG 1, and denoise 1 for all paths. Each run uses the same prompt and seeded noise. Generation timing starts with prepared conditioning and includes VAE decode. The harness records two warmed runs after the first generation and emits latent and image hashes for comparison.

## Original results with SHA-256 enabled

Measured on Windows 11 with an i5-12400, 32 GB RAM, and a GTX 1660 Super with 6 GiB VRAM. ComfyUI 0.43.0 (commit `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24`) used PyTorch 2.12.0+cu130, comfy-kitchen 0.2.40, and comfy-aimdo 0.6.0.

| Measurement | Original, standard loader | Standalone reconstruction, standard loader | PFQC, direct loader |
| --- | ---: | ---: | ---: |
| File bytes | 5,413,698,958 | 5,413,698,958 | 4,954,135,614 |
| Model load, fresh process | 0.76 s | 0.23 s | 37.05 s |
| Decode and verify inside load | — | — | 36.73 s |
| Peak process RAM at load | 2.72 GiB | 2.72 GiB | 7.16 GiB |
| First generation | 33.05 s | 32.85 s | 32.88 s |
| Warmed sampler totals | 14.70 / 15.33 s | 13.61 / 13.21 s | 14.46 / 14.96 s |
| Warmed generation totals | 15.98 / 16.62 s | 14.90 / 14.44 s | 15.73 / 16.28 s |
| Warmed sampler progress intervals | 3.64 / 3.75 s | 3.22 / 3.22 s | 3.44 / 3.52 s |
| Peak process RAM through generation | 8.54 GiB | 8.56 GiB | 13.72 GiB |
| Peak device VRAM, sampled | 5,054 MiB | 5,342 MiB | 5,413 MiB |

The CLI encoded 168 weights and verified 757 tensors. The archive is 4.61 GiB, an 8.49% reduction from the original 5.04 GiB. Standalone CLI decompression took 35.65 s including output writes.

Model-load timings start in a fresh ComfyUI process. Windows filesystem caches were left in their normal state, and the reference and archive had recently been written. The standard loader maps the file and can defer I/O until first sampling, while the PFQC timer includes a full checked read. The reload recipe took 37.14 s in the same process, including 36.96 s of decoding.

RAM figures use peak Windows process working set. VRAM figures use 250 ms `nvidia-smi` samples for the entire GPU, including the desktop. PyTorch reported the same 768 MiB peak allocator usage during generation in all three runs. ComfyUI's dynamic allocations also contribute to the device-level samples. Sampler progress intervals are host-side measurements between progress callbacks after the first step. Sampler totals synchronize CUDA before recording the end time.

All 757 reconstructed tensors matched the standalone reference in name, shape, dtype, and raw bytes, and their checkpoint metadata matched. The two paths produced identical latent and image hashes in all three runs. Their saved PNG files have SHA-256 `883aa3fc81f818cfc7a9843061e6d2c1e85a8492ca3c060e4c43c4f4c114b39f`.

The bundled decoder was also tested from the installed custom-node folder. It loaded the full MOSS archive as native INT8 ConvRot in portable ComfyUI with the compressor package absent.

Both paths loaded native `TensorWiseINT8Layout` with ConvRot group size 256 and dispatched `int8_linear` through `comfy_kitchen.backends.cuda`. The model-only speed LoRA cloned and patched the native model. ComfyUI kept part of the model in VRAM and offloaded the rest to CPU. The direct path decoded once across all three generations and once more when the reload factory created a fresh model after unload.

Direct PFQC averaged 16.00 s warm, compared with 14.67 s for the standalone reconstruction and 16.30 s for the original. The two standard loads used ComfyUI's fast-disk policy, while direct PFQC used its RAM offload policy. The fast-disk reference also kept about 350 MiB more model weights resident in VRAM during its warm runs than direct PFQC. With `--disable-fast-disk`, the standalone reference took 17.47 and 15.15 s warm and kept the same image hashes. Direct PFQC fell within the range of the standard-loader runs. The comparison shows sensitivity to ComfyUI's offload policy and GPU residency, with identical CUDA INT8 dispatch and zero decoder calls during sampling.

## Historical checksum comparison with the serial decoder

Both modes reconstructed all 757 tensors from the same 4.61 GiB MOSS archive. The benchmark used portable ComfyUI's Python, the serial decoder, and normal Windows file caching. It alternated the mode order across two pairs in one process. The timer covers CPU state-dict reconstruction. Stored tensors stay memory-mapped, so some page reads can move to later use when SHA-256 is skipped.

| Mode | Pair 1 | Pair 2 | Median |
| --- | ---: | ---: | ---: |
| SHA-256 enabled | 34.23 s | 33.34 s | 33.78 s |
| SHA-256 skipped | 23.66 s | 23.04 s | 23.35 s |

Skipping SHA-256 reduced reconstruction time by **10.43 s (30.9%)** in this comparison. Both modes use the same manifest, tensor-size, and DEFLATE validation. A separate smoke test of the installed node loaded native MOSS INT8 ConvRot in 25.07 s, including 24.84 s of reconstruction.

```powershell
& $comfyPython -B tests/moss_integration.py $comfyRoot $pfqcArchive --benchmark-checksums --runs 2
```

## Verified native decoder benchmark

The three paths use the same installed ComfyUI and MOSS workflow. The serial baseline uses the verified loader from commit `fe7c304`. The optimized loader uses four workers, the bundled libdeflate 1.26 Windows x64 DLL, and Windows CNG SHA-256. The standard loader uses the original INT8 ConvRot checkpoint, whose tensor bytes match this lossless PFQC fixture.

Each cold run used a fresh copy of the model made with `robocopy /J` on the same drive. The copy avoided buffered I/O and the loader received the new path before any pre-read. A separate fresh process pre-read that copy with `--warm-cache` before timing its warm load and three generations. The table reports host-wide disk read deltas as a check on cache residency. [Microsoft Robocopy documentation](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/robocopy)

Load timing runs from the loader call to the return of a usable `MODEL`. Peak RAM is the Windows process working set. The standard loader can map weights and read some pages later, while both PFQC paths reconstruct and verify every tensor during the timed call.

| Measurement | Verified serial PFQC | Verified optimized PFQC | Standard INT8 ConvRot |
| --- | ---: | ---: | ---: |
| Cold load to `MODEL` | 36.25 s | 10.29 s | 0.77 s |
| Warm load to `MODEL` | 36.37 s | 7.88 s | 0.24 s |
| Cold / warm host disk reads during load | 4,740 / 97 MiB | 4,725 / 0 MiB | 629 / 0 MiB |
| Cold / warm peak process RAM at load | 7.16 / 7.16 GiB | 7.17 / 7.17 GiB | 2.72 / 2.72 GiB |
| First generation after warm load | 33.26 s | 31.69 s | 20.05 s |
| Warmed generations | 14.61 / 14.34 s | 14.17 / 14.56 s | 14.59 / 14.57 s |
| Peak process RAM through generation | 16.07 GiB | 16.11 GiB | 8.34 GiB |

All nine generations matched in latent and image SHA-256. Both PFQC loaders decoded once across the three generations and decoded again on an explicit reload. Every path used the native CUDA INT8 ConvRot operation.

To reproduce, extract the serial node files from `fe7c304` into a separate directory and pass that directory as `--node-dir` for the serial case. Pass the current checkout for optimized PFQC. For each cold run, point the harness at a new unbuffered copy and pass `--load-only`. For the warm runs, reuse the copy with `--warm-cache --runs 3`. For the standard case, pass the ordinary `.safetensors` file.
