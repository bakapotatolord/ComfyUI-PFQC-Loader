from hashlib import sha256
import logging
from time import perf_counter

from safetensors import SafetensorError, safe_open
import torch

from ._pfqc_decoder.decoder import open_archive


def decode_checkpoint(path, *, verify_checksums=False):
    """Reconstruct a PFQC v1 archive as CPU tensors and original metadata."""
    with (
        open_archive(path, verify_checksums=verify_checksums) as archive,
        safe_open(path, framework="pt", device="cpu") as stored,
    ):
        archive.validate_families()
        state_dict = {}
        for name, info in archive.tensors.items():
            record = archive.records[name]
            if record["codec"] == "stored":
                physical = stored.get_slice(name)
                if physical.get_dtype() != info.dtype or tuple(physical.get_shape()) != info.shape:
                    raise ValueError(f"Stored tensor metadata does not match for {name!r}")
                tensor = stored.get_tensor(name)
                if verify_checksums:
                    raw = tensor.reshape(-1).view(torch.uint8).numpy()
                    if sha256(raw).hexdigest() != record["sha256"]:
                        raise ValueError(f"Checksum mismatch for {name!r}")
            else:
                tensor = torch.empty(info.shape, dtype=torch.int8, device="cpu")
                buffer = memoryview(tensor.numpy()).cast("B")
                offset = 0
                for chunk in archive.chunks(name):
                    end = offset + len(chunk)
                    buffer[offset:end] = chunk
                    offset = end
            state_dict[name] = tensor
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
