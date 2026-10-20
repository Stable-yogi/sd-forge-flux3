# FLUX 3 Image engine for Forge Neo (follows backend/diffusion_engine/flux2.py).

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from .text_engine import Flux3TextProcessingEngine


class Flux3Image(ForgeDiffusionEngine):
    matched_guesses = []  # filled by scripts/flux3_register.py

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_flux2=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_flux3 = Flux3TextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen3vl_4b,
            tokenizer=clip.tokenizer.qwen3vl_4b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_flux3 = True
        dynamic_args.flux3 = True  # read by the can_concat patch (cond / uncond lengths differ)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_flux3(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_flux3.tokenize([prompt])[0])
        return token_count, max(999, token_count)
