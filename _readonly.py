"""Use ComfyUI's read-only mapper on its supported dynamic loading path."""

import logging

import torch

from comfy import memory_management, model_management, model_patcher
from comfy.model_patcher import ModelPatcherDynamic, get_key_weight
from comfy.utils import _TYPES, load_safetensors


def supported():
    return (
        getattr(memory_management, "aimdo_enabled", False)
        and getattr(model_patcher, "CoreModelPatcher", None) is ModelPatcherDynamic
        and model_management.is_device_cpu(model_management.unet_offload_device())
        and model_management.get_torch_device().type == "cuda"
        and not torch.__future__.get_swap_module_params_on_conversion()
    )


def map_stored_tensors(path, archive):
    physical = archive._container.tensors
    if any(info.dtype not in _TYPES for info in physical.values()):
        return None
    try:
        tensors, _ = load_safetensors(path)
    except (ImportError, OSError, RuntimeError) as error:
        logging.info("PFQC read-only mapping unavailable, using safetensors: %s", error)
        return None

    stored = {}
    for name, info in archive.tensors.items():
        if archive.records[name]["codec"] != "stored":
            continue
        tensor = tensors.get(name)
        if tensor is None or tensor.dtype != _TYPES[info.dtype] or tuple(tensor.shape) != info.shape:
            raise ValueError(f"Stored mapping metadata does not match for {name!r}")
        if info.nbytes:
            storage = tensor.untyped_storage()
            file_slice = getattr(storage, "_comfy_tensor_file_slice", None)
            if file_slice is None or not getattr(storage, "_comfy_tensor_mmap_refs", None):
                return None
            if (file_slice.offset, file_slice.size) != (physical[name].offset, physical[name].nbytes):
                raise ValueError(f"Stored mapping does not match validated data for {name!r}")
            # Keep the mmap alive while leaving PFQC's CPU-to-device staging path intact.
            del storage._comfy_tensor_file_slice
        stored[name] = tensor

    logging.info("PFQC using ComfyUI read-only mapping for stored tensors")
    return stored


class ReadOnlyPatcher(ModelPatcherDynamic):
    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if not return_weight and (key in self.patches or force_cast) and (inplace_update or self.weight_inplace_update):
            weight, _, _ = get_key_weight(self.model, key)
            storage = getattr(weight, "_qdata", weight).untyped_storage()
            if hasattr(storage, "_comfy_tensor_mmap_refs"):
                raise RuntimeError("In-place patch requested on read-only PFQC storage. Use the legacy delegate.")
        return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)
