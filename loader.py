from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
from time import perf_counter

from safetensors import SafetensorError, safe_open
import torch

from ._pfqc_decoder.decoder import open_archive
from ._pfqc_decoder.native import NATIVE_TENSOR_LIMIT


DECODE_WORKERS = 4


def decode_checkpoint(path, *, verify_checksums=True):
    """Reconstruct a PFQC v1 archive as CPU tensors and original metadata."""
    with (
        open_archive(path, verify_checksums=verify_checksums) as archive,
        safe_open(path, framework="pt", device="cpu") as stored,
    ):
        archive.validate_families()
        state_dict = dict.fromkeys(archive.tensors)

        def decode_weight(name, info):
            tensor = torch.empty(info.shape, dtype=torch.int8, device="cpu")
            archive.decode_into(name, tensor.numpy())
            return tensor

        pending = deque()
        pool = ThreadPoolExecutor(max_workers=DECODE_WORKERS)
        try:
            for name, info in archive.tensors.items():
                if archive.records[name]["codec"] == "stored":
                    physical = stored.get_slice(name)
                    if physical.get_dtype() != info.dtype or tuple(physical.get_shape()) != info.shape:
                        raise ValueError(f"Stored tensor metadata does not match for {name!r}")
                    tensor = stored.get_tensor(name)
                    raw = tensor.reshape(-1).view(torch.uint8).numpy()
                    archive.check_tensor(name, raw)
                    state_dict[name] = tensor
                elif info.nbytes <= NATIVE_TENSOR_LIMIT:
                    if len(pending) == DECODE_WORKERS:
                        key, result = pending.popleft()
                        state_dict[key] = result.result()
                    pending.append((name, pool.submit(decode_weight, name, info)))
                else:
                    # Oversized weights stream through zlib after all smaller jobs finish.
                    while pending:
                        key, result = pending.popleft()
                        state_dict[key] = result.result()
                    state_dict[name] = decode_weight(name, info)
            for name, result in pending:
                state_dict[name] = result.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return state_dict, archive.metadata


def load_diffusion_model(path, *, disable_dynamic=False):
    import comfy.sd

    started = perf_counter()
    try:
        state_dict, metadata = decode_checkpoint(path)
    except (OSError, SafetensorError, ValueError, RecursionError, OverflowError) as error:
        raise RuntimeError(f"Cannot decode PFQC archive {path}: {error}") from error
    logging.info("Decoded PFQC archive %s in %.2f s", path, perf_counter() - started)

    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict, metadata=metadata, disable_dynamic=disable_dynamic,
    )
    if model is None:
        hint = comfy.sd.model_detection_error_hint(path, state_dict)
        raise RuntimeError(f"Could not detect the diffusion model in {path}\n{hint}")
    model.cached_patcher_init = (load_diffusion_model, (path,))
    return model
