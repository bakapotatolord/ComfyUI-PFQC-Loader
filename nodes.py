import folder_paths

from .loader import load_diffusion_model


MODEL_FOLDER = "diffusion_models"
ARCHIVE_SUFFIX = ".pfqc.safetensors"


class LoadDiffusionModelPFQC:
    @classmethod
    def INPUT_TYPES(cls):
        models = [
            name for name in folder_paths.get_filename_list(MODEL_FOLDER)
            if name.endswith(ARCHIVE_SUFFIX)
        ]
        return {"required": {"model_name": (models,)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders/PotatoForge"

    def load(self, model_name):
        if (
            not model_name.endswith(ARCHIVE_SUFFIX)
            or model_name not in folder_paths.get_filename_list(MODEL_FOLDER)
        ):
            raise ValueError(f"Choose a {ARCHIVE_SUFFIX} archive from ComfyUI's diffusion_models folders.")
        path = folder_paths.get_full_path_or_raise(MODEL_FOLDER, model_name)
        return (load_diffusion_model(path),)


NODE_CLASS_MAPPINGS = {"LoadDiffusionModelPFQC": LoadDiffusionModelPFQC}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadDiffusionModelPFQC": "Load Diffusion Model (PFQC)"}
