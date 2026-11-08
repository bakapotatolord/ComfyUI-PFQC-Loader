from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import logging
import sys
from time import perf_counter

from safetensors import SafetensorError, safe_open
import torch

from ._pfqc_decoder.decoder import open_archive
from ._pfqc_decoder.native import NATIVE_TENSOR_LIMIT


DECODE_WORKERS = 4


@contextmanager
def _stored_reader(path, archive, read_only):
    if read_only:
        from ._readonly import map_stored_tensors

        mapped = map_stored_tensors(path, archive)
        if mapped is not None:
            yield mapped.__getitem__
            return

    with safe_open(path, framework="pt", device="cpu") as stored:
        def get_tensor(name):
            info = archive.tensors[name]
            physical = stored.get_slice(name)
            if physical.get_dtype() != info.dtype or tuple(physical.get_shape()) != info.shape:
                raise ValueError(f"Stored tensor metadata does not match for {name!r}")
            return stored.get_tensor(name)

        yield get_tensor


def decode_checkpoint(path, *, verify_checksums=True, read_only=False):
    """Reconstruct PFQC CPU tensors and their original metadata."""
    with (
        open_archive(path, verify_checksums=verify_checksums) as archive,
        _stored_reader(path, archive, read_only) as get_stored,
    ):
        archive.validate_families()
        state_dict = dict.fromkeys(archive.tensors)

        def decode_weight(name, info):
            tensor = torch.empty(info.shape, dtype=torch.int8, device="cpu")
            archive.decode_into(name, tensor.numpy())
            return tensor

        pending = {}
        pool = ThreadPoolExecutor(max_workers=DECODE_WORKERS)
        try:
            for name, info in archive.tensors.items():
                if archive.records[name]["codec"] == "stored":
                    tensor = get_stored(name)
                    raw = tensor.reshape(-1).view(torch.uint8).numpy()
                    archive.check_tensor(name, raw)
                    state_dict[name] = tensor
                elif info.nbytes <= NATIVE_TENSOR_LIMIT:
                    if len(pending) == DECODE_WORKERS:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            state_dict[pending.pop(future)] = future.result()
                    pending[pool.submit(decode_weight, name, info)] = name
                else:
                    # Oversized weights stream through zlib after all smaller jobs finish.
                    for future, key in pending.items():
                        state_dict[key] = future.result()
                    pending.clear()
                    state_dict[name] = decode_weight(name, info)
            for future, name in pending.items():
                state_dict[name] = future.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return state_dict, archive.metadata


def _read_only_patcher(disable_dynamic):
    if sys.platform != "win32" or disable_dynamic:
        return None
    try:
        from ._readonly import ReadOnlyPatcher, supported
    except ImportError:
        return None
    return ReadOnlyPatcher if supported() else None


def load_diffusion_model(path, *, disable_dynamic=False):
    import comfy.sd

    started = perf_counter()
    read_only_patcher = _read_only_patcher(disable_dynamic)
    try:
        if read_only_patcher is None:
            state_dict, metadata = decode_checkpoint(path)
        else:
            state_dict, metadata = decode_checkpoint(path, read_only=True)
    except (OSError, SafetensorError, ValueError, RecursionError, OverflowError) as error:
        raise RuntimeError(f"Cannot decode PFQC archive {path}: {error}") from error
    logging.info("Decoded PFQC archive %s in %.2f s", path, perf_counter() - started)
    has_mapped_tensors = read_only_patcher is not None and any(
        hasattr(tensor.untyped_storage(), "_comfy_tensor_mmap_refs") for tensor in state_dict.values()
    )

    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict, metadata=metadata, disable_dynamic=disable_dynamic,
    )
    if model is None:
        hint = comfy.sd.model_detection_error_hint(path, state_dict)
        raise RuntimeError(f"Could not detect the diffusion model in {path}\n{hint}")
    if has_mapped_tensors:
        # Keep the native patcher's state and carry the write guard through its clones.
        model.__class__ = read_only_patcher
    model.cached_patcher_init = (load_diffusion_model, (path,))
    return model
