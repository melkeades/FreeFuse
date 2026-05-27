# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FreeFuse Flux2 Klein Pipeline for multi-concept image generation.

This pipeline extends Flux2KleinPipeline with FreeFuse capabilities:
- Two-phase generation: sim-map extraction -> mask application
- Per-LoRA spatial masks from concept similarity maps
- Additive attention bias for cross-LoRA suppression
- Background detection and exclusion
"""

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diffusers.loaders import Flux2LoraLoaderMixin
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

from src.models.freefuse_transformer_flux2 import Flux2Transformer2DModel as FreeFuseFlux2Transformer2DModel
from src.attn_processor.freefuse_flux2_attn_processor import (
    FreeFuseFlux2AttnProcessor,
    FreeFuseFlux2SingleAttnProcessor,
)
from src.tuner.freefuse_lora_layer import FreeFuseLinear, convert_peft_lora_to_freefuse_lora


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from src.pipeline.freefuse_flux2_klein_pipeline import FreeFuseFlux2KleinPipeline

        >>> pipe = FreeFuseFlux2KleinPipeline.from_pretrained(
        ...     "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")
        >>> pipe.load_lora_weights("path/to/lora1", adapter_name="concept1")
        >>> pipe.load_lora_weights("path/to/lora2", adapter_name="concept2")
        >>> prompt = "A photo of <sks1> and <tok2> together"
        >>> image = pipe(
        ...     prompt=prompt, 
        ...     freefuse_lora_names=["concept1", "concept2"],
        ...     freefuse_concept_tokens=["<sks1>", "<tok2>"],
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("freefuse_output.png")
        ```
"""


# Copied from diffusers.pipelines.flux2.pipeline_flux2.compute_empirical_mu
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class FreeFuseFlux2KleinPipeline(DiffusionPipeline, Flux2LoraLoaderMixin):
    r"""
    FreeFuse Flux2 Klein pipeline for multi-concept text-to-image generation.

    Extends Flux2KleinPipeline with FreeFuse capabilities for better concept isolation
    when using multiple LoRAs.

    Reference:
    [https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)

    Args:
        transformer ([`Flux2Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLFlux2`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen3ForCausalLM`]):
            [Qwen3ForCausalLM](https://huggingface.co/docs/transformers/en/model_doc/qwen3#transformers.Qwen3ForCausalLM)
        tokenizer (`Qwen2TokenizerFast`):
            Tokenizer of class
            [Qwen2TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        transformer: Flux2Transformer2DModel,
        is_distilled: bool = False,
    ):
        super().__init__()

        if not isinstance(transformer, FreeFuseFlux2Transformer2DModel):
            transformer.__class__ = FreeFuseFlux2Transformer2DModel

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )

        self.register_to_config(is_distilled=is_distilled)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

    @staticmethod
    def _get_qwen3_prompt_embeds(
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: List[int] = (9, 18, 27),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        # Forward pass through the model
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_text_ids
    def _prepare_text_ids(
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: Optional[torch.Tensor] = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_latent_ids
    def _prepare_latent_ids(
        latents: torch.Tensor,  # (B, C, H, W)
    ):
        r"""
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents (torch.Tensor):
                Latent tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
                H=[0..H-1], W=[0..W-1], L=0
        """

        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_image_ids
    def _prepare_image_ids(
        image_latents: List[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
        scale: int = 10,
    ):
        r"""
        Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.

        This function creates a unique coordinate for every pixel/patch across all input latent with different
        dimensions.

        Args:
            image_latents (List[torch.Tensor]):
                A list of image latent feature tensors, typically of shape (C, H, W).
            scale (int, optional):
                A factor used to define the time separation (T-coordinate) between latents. T-coordinate for the i-th
                latent is: 'scale + scale * i'. Defaults to 10.

        Returns:
            torch.Tensor:
                The combined coordinate tensor. Shape: (1, N_total, 4) Where N_total is the sum of (H * W) for all
                input latents.

        Coordinate Components (Dimension 4):
            - T (Time): The unique index indicating which latent image the coordinate belongs to.
            - H (Height): The row index within that latent image.
            - W (Width): The column index within that latent image.
            - L (Seq. Length): A sequence length dimension, which is always fixed at 0 (size 1)
        """

        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

        # create time offset for each reference image
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._patchify_latents
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpatchify_latents
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._pack_latents
    def _pack_latents(latents):
        """
        pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
        """

        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpack_latents_with_ids
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        using position ids to scatter tokens into place
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape  # noqa: F841
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (9, 18, 27),
    ):
        device = device or self._execution_device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = self._patchify_latents(image_latents)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)

        latents = self._pack_latents(latents)  # [B, C, H, W] -> [B, H*W, C]
        return latents, latent_ids

    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.prepare_image_latents
    def prepare_image_latents(
        self,
        images: List[torch.Tensor],
        batch_size,
        generator: torch.Generator,
        device,
        dtype,
    ):
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            imagge_latent = self._encode_vae_image(image=image, generator=generator)
            image_latents.append(imagge_latent)  # (1, 128, 32, 32)

        image_latent_ids = self._prepare_image_ids(image_latents)

        # Pack each latent and concatenate
        packed_latents = []
        for latent in image_latents:
            # latent: (1, 128, 32, 32)
            packed = self._pack_latents(latent)  # (1, 1024, 128)
            packed = packed.squeeze(0)  # (1024, 128) - remove batch dim
            packed_latents.append(packed)

        # Concatenate all reference tokens along sequence dimension
        image_latents = torch.cat(packed_latents, dim=0)  # (N*1024, 128)
        image_latents = image_latents.unsqueeze(0)  # (1, N*1024, 128)

        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale=None,
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * 2) != 0
            or width is not None
            and width % (self.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if guidance_scale > 1.0 and self.config.is_distilled:
            logger.warning(f"Guidance scale {guidance_scale} is ignored for step-wise distilled models.")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and not self.config.is_distilled

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    # ──────────────────────────────────────────────────────────────────
    # FreeFuse Methods
    # ──────────────────────────────────────────────────────────────────
    def find_concept_token_positions(
        self,
        prompt: str,
        concept_tokens: List[str],
    ) -> Dict[str, List[int]]:
        """
        Find positions of concept tokens in the Qwen3 tokenized prompt.
        
        Args:
            prompt: The prompt string
            concept_tokens: List of concept token strings (e.g., ["<sks1>", "<tok2>"])
            
        Returns:
            Dict mapping concept token -> list of positions in tokenized sequence
        """
        # Apply chat template as done in encode_prompt
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
        )
        input_ids = inputs["input_ids"][0]
        
        token_positions = {}
        for concept in concept_tokens:
            # Tokenize the concept alone
            concept_ids = self.tokenizer.encode(concept, add_special_tokens=False)
            concept_len = len(concept_ids)
            
            positions = []
            for i in range(len(input_ids) - concept_len + 1):
                if input_ids[i:i+concept_len].tolist() == concept_ids:
                    positions.extend(range(i, i + concept_len))
            
            token_positions[concept] = positions
        
        return token_positions

    def find_eos_token_index(self, prompt: str) -> Optional[int]:
        """Find the first EOS token position in the prompt."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
        )
        input_ids = inputs["input_ids"][0]
        
        eos_id = self.tokenizer.eos_token_id
        for i, tok in enumerate(input_ids):
            if tok == eos_id:
                return i
        return None

    @staticmethod
    def _infer_sim_map_hw(
        img_len: int,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Infer 2D shape from flattened token length."""
        if img_len <= 0:
            raise ValueError(f"Invalid sim-map token length: {img_len}")

        if (
            target_height is not None
            and target_width is not None
            and img_len == target_height * target_width
        ):
            return target_height, target_width

        h_lat = int(img_len**0.5)
        while h_lat > 1 and img_len % h_lat != 0:
            h_lat -= 1
        w_lat = img_len // h_lat
        return h_lat, w_lat

    def _to_debug_2d_map(
        self,
        tensor: torch.Tensor,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert a map-like tensor into a 2D numpy array for debug visualization.

        Supported shapes:
        - (B, 1, H, W)
        - (B, N, 1)
        - (B, N)
        """
        if tensor is None:
            raise ValueError("Expected a tensor, got None.")

        if tensor.dim() == 4:
            map_2d = tensor[0, 0]
        elif tensor.dim() == 3:
            if tensor.shape[-1] == 1:
                flat = tensor[0, :, 0]
            elif tensor.shape[1] == 1:
                flat = tensor[0, 0, :]
            else:
                raise ValueError(f"Unsupported 3D tensor shape: {tuple(tensor.shape)}")
            h_lat, w_lat = self._infer_sim_map_hw(flat.shape[0], target_height, target_width)
            map_2d = flat.reshape(h_lat, w_lat)
        elif tensor.dim() == 2:
            if tensor.shape[0] <= 4:
                flat = tensor[0]
                h_lat, w_lat = self._infer_sim_map_hw(flat.shape[0], target_height, target_width)
                map_2d = flat.reshape(h_lat, w_lat)
            else:
                map_2d = tensor
        else:
            raise ValueError(f"Unsupported tensor shape for debug map: {tuple(tensor.shape)}")

        return map_2d.detach().cpu().float().numpy()

    def _save_debug_visualizations(
        self,
        debug_save_path: str,
        step_idx: int = 0,
        sim_maps: Optional[Dict[str, torch.Tensor]] = None,
        lora_masks: Optional[Dict[str, torch.Tensor]] = None,
        attention_bias: Optional[torch.Tensor] = None,
        attention_bias_vmax: Optional[float] = None,
    ) -> None:
        """Save FreeFuse intermediate maps/masks/attention-bias visualizations."""
        if debug_save_path is None:
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning(
                "`debug_save_path` is set but matplotlib is unavailable (%s). Skip debug visualizations.",
                exc,
            )
            return

        os.makedirs(debug_save_path, exist_ok=True)

        target_height = None
        target_width = None
        if lora_masks:
            first_mask = next(iter(lora_masks.values()))
            if first_mask.dim() == 4:
                target_height, target_width = first_mask.shape[-2], first_mask.shape[-1]

        if sim_maps:
            for concept_name, sim_map in sim_maps.items():
                try:
                    sim_map_2d = self._to_debug_2d_map(sim_map, target_height=target_height, target_width=target_width)
                except ValueError as err:
                    logger.warning("Skip sim-map debug for `%s`: %s", concept_name, err)
                    continue

                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(sim_map_2d, cmap="viridis")
                ax.set_title(f"Concept Sim Map: {concept_name} (step {step_idx})")
                plt.colorbar(im, ax=ax)
                plt.savefig(
                    os.path.join(debug_save_path, f"concept_sim_map_{concept_name}_step{step_idx}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

        if lora_masks:
            for lora_name, mask in lora_masks.items():
                try:
                    mask_2d = self._to_debug_2d_map(mask)
                except ValueError as err:
                    logger.warning("Skip LoRA-mask debug for `%s`: %s", lora_name, err)
                    continue

                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(mask_2d, cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_title(f"LoRA Mask: {lora_name} (step {step_idx})")
                plt.colorbar(im, ax=ax)
                plt.savefig(
                    os.path.join(debug_save_path, f"lora_mask_{lora_name}_step{step_idx}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

        if attention_bias is not None:
            bias_2d = attention_bias[0].detach().cpu().float().numpy()
            if attention_bias_vmax is None:
                vmax = float(np.max(np.abs(bias_2d)))
                attention_bias_vmax = max(vmax, 1.0)
            else:
                attention_bias_vmax = max(abs(float(attention_bias_vmax)), 1.0)

            fig, ax = plt.subplots(figsize=(12, 12))
            im = ax.imshow(
                bias_2d,
                cmap="RdBu",
                vmin=-attention_bias_vmax,
                vmax=attention_bias_vmax,
            )
            ax.set_title(f"Attention Bias Matrix (step {step_idx})")
            ax.set_xlabel("Key positions (text | image)")
            ax.set_ylabel("Query positions (text | image)")
            plt.colorbar(im, ax=ax)
            plt.savefig(
                os.path.join(debug_save_path, f"attention_bias_step{step_idx}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    def stabilized_balanced_argmax(
        self,
        logits: torch.Tensor,
        h: int,
        w: int,
        target_count: Optional[float] = None,
        max_iter: int = 15,
        lr: float = 0.01,
        gravity_weight: float = 0.000004,
        spatial_weight: float = 0.000004,
        momentum: float = 0.2,
        centroid_margin: float = 0.0,
        border_penalty: float = 0.0,
        anisotropy: float = 1.1,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Balanced argmax with iterative spatial/centroid regularization.

        This is transplanted from the flux.dev implementation to stabilize concept
        partitioning compared with plain argmax.
        """
        B, C, N = logits.shape
        device = logits.device

        max_dim = max(h, w)
        scale_h = 1.0
        scale_w = 1.0

        y_range = torch.linspace(-scale_h, scale_h, steps=h, device=device)
        x_range = torch.linspace(-scale_w, scale_w, steps=w, device=device)
        x_range = x_range * anisotropy

        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
        flat_y = grid_y.reshape(1, 1, N)
        flat_x = grid_x.reshape(1, 1, N)

        pixel_size = 2.0 / max_dim
        is_border = (flat_y.abs() > (scale_h - pixel_size * 1.5)) | (
            flat_x.abs() > (scale_w - pixel_size * 1.5)
        )
        border_mask = is_border.float()

        if target_count is None:
            target_count = N / C

        bias = torch.zeros(B, C, 1, device=device)

        def linear_normalize(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
            x_min = x.min(dim=dim, keepdim=True)[0]
            x_max = x.max(dim=dim, keepdim=True)[0]
            return (x - x_min) / (x_max - x_min + 1e-8)

        running_probs = linear_normalize(logits, dim=1)

        logit_range = (logits.max() - logits.min()).item()
        logit_scale = max(logit_range, 1e-4)
        effective_lr = lr * logit_scale
        max_bias = logit_scale * 10.0

        if debug:
            print(f"\n[ArgmaxV4 Debug] Start. B={B}, C={C}, N={N}, Target={target_count:.1f}")
            print(f"Logits range: min={logits.min().item():.5f}, max={logits.max().item():.5f}")
            print(f"Logit scale: {logit_scale:.6f}, Effective LR: {effective_lr:.6f}")

        neighbor_kernel = torch.ones(C, 1, 3, 3, device=device, dtype=logits.dtype) / 8.0
        neighbor_kernel[:, :, 1, 1] = 0

        current_logits = logits.clone()

        for i in range(max_iter):
            probs = linear_normalize(current_logits - bias, dim=1)

            running_probs = (1 - momentum) * probs + momentum * running_probs

            mass = running_probs.sum(dim=2, keepdim=True) + 1e-6
            center_y = (running_probs * flat_y).sum(dim=2, keepdim=True) / mass
            center_x = (running_probs * flat_x).sum(dim=2, keepdim=True) / mass

            if centroid_margin > 0:
                limit_y = scale_h * (1.0 - centroid_margin)
                limit_x = scale_w * (1.0 - centroid_margin)
                center_y = torch.clamp(center_y, -limit_y, limit_y)
                center_x = torch.clamp(center_x, -limit_x, limit_x)

            dist_sq = (flat_y - center_y) ** 2 + (flat_x - center_x) ** 2

            hard_indices = torch.argmax(current_logits - bias, dim=1)
            hard_counts = F.one_hot(hard_indices, num_classes=C).float().sum(dim=1)

            diff = hard_counts - target_count
            cur_lr = effective_lr * (0.95**i)
            bias += torch.sign(diff).unsqueeze(2) * cur_lr
            bias = torch.clamp(bias, -max_bias, max_bias)

            if spatial_weight > 0:
                probs_img = running_probs.view(B, C, h, w)
                probs_img_f32 = probs_img.float()
                kernel_f32 = neighbor_kernel.float()
                neighbor_votes = F.conv2d(probs_img_f32, kernel_f32, padding=1, groups=C)
                neighbor_votes = neighbor_votes.to(logits.dtype).view(B, C, N)
            else:
                neighbor_votes = torch.zeros_like(logits)

            gravity_term = dist_sq * gravity_weight
            border_term = border_mask * border_penalty

            current_logits = (
                logits
                - bias
                + (neighbor_votes * spatial_weight)
                - gravity_term
                - border_term
            )

            if debug and (i == 0 or i == max_iter - 1 or i % 10 == 0):
                hard_counts_list = hard_counts[0].int().tolist()
                print(f"Iter {i:02d}: Counts={hard_counts_list}")
                print(f"  Bias range: {bias.min().item():.5f} ~ {bias.max().item():.5f}")
                print(
                    f"  Gravity Mean: {gravity_term.mean().item():.3f}, Max: {gravity_term.max().item():.3f}"
                )
                print(f"  Spatial Mean: {(neighbor_votes * spatial_weight).mean().item():.3f}")

        if debug:
            final_assignment = torch.argmax(current_logits, dim=1)[0]
            final_counts = final_assignment.bincount(minlength=C).tolist()
            print(f"[Argmax Debug] Final Counts: {final_counts}\n")

        return torch.argmax(current_logits, dim=1)

    def sim_maps_to_masks(
        self,
        sim_maps: Dict[str, torch.Tensor],
        height: int,
        width: int,
        exclude_background: bool = True,
        normalize: bool = True,
        debug_save_path: Optional[str] = None,
        debug_step_idx: int = 0,
        eos_bg_scale: float = 0.95,
        use_morphological_clean: bool = True,
        opening_kernel_size: int = 2,
        closing_kernel_size: int = 2,
        use_balanced_argmax: bool = True,
        balanced_argmax_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert similarity maps to spatial masks.
        
        Args:
            sim_maps: Dict from lora_name -> (B, img_len, 1) sim map
            height: Target height (in latent space)
            width: Target width (in latent space)
            exclude_background: If True, set background mask to 0 for all concepts
            normalize: If True, normalize masks to sum to 1 at each position
            debug_save_path: If set, save sim-map/mask debug visualizations to this folder
            debug_step_idx: Step index used in debug output file names
            eos_bg_scale: Scale applied to bg/eos channel before bg-vs-fg argmax
            use_morphological_clean: Whether to apply opening+closing for fg/bg mask cleanup
            opening_kernel_size: Kernel size for foreground opening
            closing_kernel_size: Kernel size for foreground closing
            use_balanced_argmax: If True, use stabilized_balanced_argmax for concept assignment
            balanced_argmax_debug: Enable debug logs in stabilized_balanced_argmax
            
        Returns:
            Dict of lora_name -> (B, 1, H, W) spatial mask tensors
        """
        if len(sim_maps) == 0:
            return {}

        def _squeeze_sim_map(x: torch.Tensor, name: str) -> torch.Tensor:
            # normalize to shape (B, N)
            if x.dim() == 4 and x.shape[1] == 1:
                return x[:, 0].reshape(x.shape[0], -1)
            if x.dim() == 3:
                if x.shape[-1] == 1:
                    return x[:, :, 0]
                if x.shape[1] == 1:
                    return x[:, 0, :]
            if x.dim() == 2:
                return x
            raise ValueError(f"Unsupported sim-map shape for `{name}`: {tuple(x.shape)}")

        # Avoid mutating caller data
        local_maps = dict(sim_maps)
        bg_sim_map = local_maps.pop("__bg__", None)
        eos_sim_map = local_maps.pop("__eos__", None)
        if bg_sim_map is None:
            bg_sim_map = eos_sim_map

        concept_items = list(local_maps.items())
        if len(concept_items) == 0:
            return {}

        concept_names = [name for name, _ in concept_items]
        concept_tensor = torch.stack(
            [_squeeze_sim_map(sim_map, name) for name, sim_map in concept_items],
            dim=1,
        )  # (B, C, N)

        B, C, img_len = concept_tensor.shape
        h_lat, w_lat = self._infer_sim_map_hw(img_len, height, width)

        # Foreground mask via bg-vs-concept argmax (reference: direct_extract pipeline)
        if exclude_background:
            if bg_sim_map is not None:
                bg_channel = _squeeze_sim_map(bg_sim_map, "__bg__") * eos_bg_scale  # (B, N)
            else:
                # fallback when no explicit bg/eos token map exists
                bg_channel = concept_tensor.mean(dim=1)

            sim_with_bg = torch.cat([concept_tensor, bg_channel.unsqueeze(1)], dim=1)  # (B, C+1, N)
            raw_not_background_mask = (sim_with_bg.argmax(dim=1) != C).float()  # (B, N)
        else:
            raw_not_background_mask = torch.ones((B, img_len), device=concept_tensor.device, dtype=concept_tensor.dtype)

        if use_morphological_clean:
            cleaned_not_background = self.morphological_clean_mask(
                raw_not_background_mask,
                h=h_lat,
                w=w_lat,
                opening_kernel_size=opening_kernel_size,
                closing_kernel_size=closing_kernel_size,
            )
            not_background_mask = cleaned_not_background > 0.5
        else:
            not_background_mask = raw_not_background_mask > 0.5

        # Hard assignment to concept channels (01 mask)
        if use_balanced_argmax:
            max_indices = self.stabilized_balanced_argmax(
                concept_tensor,
                h_lat,
                w_lat,
                debug=balanced_argmax_debug,
            )
        else:
            max_indices = torch.argmax(concept_tensor, dim=1)  # (B, N)

        masks: Dict[str, torch.Tensor] = {}
        for idx, lora_name in enumerate(concept_names):
            hard_mask_flat = ((max_indices == idx) & not_background_mask).float()  # (B, N), binary
            hard_mask_2d = hard_mask_flat.reshape(B, 1, h_lat, w_lat)

            if h_lat != height or w_lat != width:
                # nearest keeps the mask binary
                hard_mask_2d = F.interpolate(hard_mask_2d, size=(height, width), mode="nearest")
            masks[lora_name] = hard_mask_2d

        # Keep normalization option for compatibility. For hard one-hot masks this stays binary.
        if normalize and len(masks) > 0:
            all_masks = torch.stack(list(masks.values()), dim=0)  # (num_concepts, B, 1, H, W)
            mask_sum = all_masks.sum(dim=0, keepdim=True).clamp(min=1e-6)
            all_masks = all_masks / mask_sum

            for i, lora_name in enumerate(masks.keys()):
                masks[lora_name] = all_masks[i]

        if debug_save_path is not None:
            self._save_debug_visualizations(
                debug_save_path=debug_save_path,
                step_idx=debug_step_idx,
                sim_maps=sim_maps,
                lora_masks=masks,
            )
            logger.info(
                "[FreeFuse Debug] Saved sim-map/mask debug outputs to `%s` (step %d).",
                debug_save_path,
                debug_step_idx,
            )
        
        return masks

    def build_attention_bias(
        self,
        lora_masks: Dict[str, torch.Tensor],
        freefuse_token_pos_maps: Dict[str, List[List[int]]],
        txt_len: int,
        img_len: int,
        suppress_strength: float = -1e4,
        debug_save_path: Optional[str] = None,
        debug_step_idx: int = 0,
        positive_bias_scale: float = 1.0,
        bidirectional: bool = True,
        use_positive_bias: bool = True,
    ) -> torch.Tensor:
        """
        Build additive attention bias for cross-LoRA suppression.
        
        For each concept, we want to suppress attention from image regions that
        do NOT belong to that concept to the concept's text tokens.
        
        Args:
            lora_masks: Dict of lora_name -> (B, 1, H, W) spatial masks
            freefuse_token_pos_maps: Dict of lora_name -> [[positions], ...]
            txt_len: Text sequence length
            img_len: Image sequence length
            suppress_strength: Negative bias to apply for suppression
            debug_save_path: If set, save attention-bias debug visualization to this folder
            debug_step_idx: Step index used in debug output file names
            positive_bias_scale: Positive bias for same-LoRA image/text pairs
            bidirectional: If True, also apply text->image bias
            use_positive_bias: If True, add positive same-LoRA bias terms
            
        Returns:
            (B, txt_len+img_len, txt_len+img_len) additive attention bias
        """
        if len(lora_masks) == 0:
            return None

        first_mask = next(iter(lora_masks.values()))
        B = first_mask.shape[0]
        device = first_mask.device
        dtype = first_mask.dtype

        total_len = txt_len + img_len
        bias = torch.zeros((B, total_len, total_len), device=device, dtype=dtype)

        lora_names = list(lora_masks.keys())
        lora_name_to_idx = {name: idx for idx, name in enumerate(lora_names)}

        # Map each text token to concept idx; -1 means shared/unmapped token
        text_token_to_lora = torch.full((txt_len,), -1, device=device, dtype=torch.long)
        for lora_name, positions_list in freefuse_token_pos_maps.items():
            if lora_name not in lora_name_to_idx:
                continue

            lora_idx = lora_name_to_idx[lora_name]
            positions = positions_list[0] if len(positions_list) > 0 else []
            for pos in positions:
                if 0 <= pos < txt_len:
                    text_token_to_lora[pos] = lora_idx

        neg_scale = abs(float(suppress_strength))

        # Flatten/reshape masks to (B, img_len)
        flat_masks: Dict[str, torch.Tensor] = {}
        for lora_name, mask in lora_masks.items():
            if mask.dim() == 4:
                flat_mask = mask.reshape(B, -1)
            elif mask.dim() == 3:
                if mask.shape[-1] == 1:
                    flat_mask = mask[:, :, 0]
                elif mask.shape[1] == 1:
                    flat_mask = mask[:, 0, :]
                else:
                    flat_mask = mask.reshape(B, -1)
            elif mask.dim() == 2:
                flat_mask = mask
            else:
                raise ValueError(f"Unsupported mask shape for `{lora_name}`: {tuple(mask.shape)}")

            if flat_mask.shape[1] != img_len:
                if flat_mask.shape[1] > img_len:
                    flat_mask = flat_mask[:, :img_len]
                else:
                    flat_mask = F.pad(flat_mask, (0, img_len - flat_mask.shape[1]), value=0.0)

            flat_masks[lora_name] = flat_mask.to(device=device, dtype=dtype)

        for lora_name, img_mask in flat_masks.items():
            lora_idx = lora_name_to_idx[lora_name]

            other_lora_text_mask = (text_token_to_lora != lora_idx) & (text_token_to_lora != -1)
            other_lora_text_mask = other_lora_text_mask.to(dtype=dtype)  # (txt_len,)

            img_to_txt_bias = img_mask.unsqueeze(-1) * other_lora_text_mask.unsqueeze(0).unsqueeze(0)
            bias[:, txt_len:, :txt_len] += img_to_txt_bias * (-neg_scale)

            this_lora_text_mask = (text_token_to_lora == lora_idx).to(dtype=dtype)  # (txt_len,)
            if use_positive_bias:
                img_to_txt_positive = img_mask.unsqueeze(-1) * this_lora_text_mask.unsqueeze(0).unsqueeze(0)
                bias[:, txt_len:, :txt_len] += img_to_txt_positive * float(positive_bias_scale)

            if bidirectional:
                not_this_lora_img_mask = 1.0 - img_mask
                txt_to_img_bias = this_lora_text_mask.unsqueeze(0).unsqueeze(-1) * not_this_lora_img_mask.unsqueeze(1)
                bias[:, :txt_len, txt_len:] += txt_to_img_bias * (-neg_scale)

                if use_positive_bias:
                    txt_to_img_positive = this_lora_text_mask.unsqueeze(0).unsqueeze(-1) * img_mask.unsqueeze(1)
                    bias[:, :txt_len, txt_len:] += txt_to_img_positive * float(positive_bias_scale)

        if debug_save_path is not None:
            self._save_debug_visualizations(
                debug_save_path=debug_save_path,
                step_idx=debug_step_idx,
                attention_bias=bias,
                attention_bias_vmax=abs(float(suppress_strength)),
            )
            logger.info(
                "[FreeFuse Debug] Saved attention-bias debug output to `%s` (step %d).",
                debug_save_path,
                debug_step_idx,
            )
        
        return bias

    def morphological_clean_mask(
        self,
        mask: torch.Tensor,
        h: int,
        w: int,
        opening_kernel_size: int = 3,
        closing_kernel_size: int = 3,
    ) -> torch.Tensor:
        """
        Clean binary fg/bg mask using opening then closing.

        Input shape: (B, N) where N = h * w
        Output shape: (B, N)
        """
        if mask.dim() != 2:
            raise ValueError(f"`mask` must be 2D (B, N), got {tuple(mask.shape)}")
        if mask.shape[1] != h * w:
            raise ValueError(f"`mask` length mismatch: got N={mask.shape[1]}, expected {h*w} from h={h}, w={w}")

        mask_2d = mask.view(mask.shape[0], 1, h, w)

        def dilate(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
            padding = kernel_size // 2
            out = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
            if out.shape[-2:] != x.shape[-2:]:
                out = F.interpolate(out, size=x.shape[-2:], mode="nearest")
            return out

        def erode(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
            padding = kernel_size // 2
            out = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel_size, stride=1, padding=padding)
            if out.shape[-2:] != x.shape[-2:]:
                out = F.interpolate(out, size=x.shape[-2:], mode="nearest")
            return out

        if opening_kernel_size > 1:
            opened = dilate(erode(mask_2d, opening_kernel_size), opening_kernel_size)
        else:
            opened = mask_2d

        if closing_kernel_size > 1:
            closed = erode(dilate(opened, closing_kernel_size), closing_kernel_size)
        else:
            closed = opened

        return closed.view(mask.shape[0], -1)

    def setup_freefuse_attention_processors(self) -> None:
        """Replace attention processors with FreeFuse variants."""
        if not isinstance(self.transformer, FreeFuseFlux2Transformer2DModel):
            self.transformer.__class__ = FreeFuseFlux2Transformer2DModel

        for name, module in self.transformer.named_modules():
            if hasattr(module, 'processor'):
                if 'single_transformer_blocks' in name:
                    module.set_processor(FreeFuseFlux2SingleAttnProcessor())
                elif 'transformer_blocks' in name:
                    module.set_processor(FreeFuseFlux2AttnProcessor())

    def convert_lora_layers(self, lora_names: List[str]) -> None:
        """Convert PEFT LoRA layers to FreeFuseLinear layers."""
        convert_peft_lora_to_freefuse_lora(self.transformer, lora_names)


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[Union[str, List[str]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (9, 18, 27),
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality. For step-wise distilled models,
                `guidance_scale` is ignored.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Note that "" is used as the negative prompt in this pipeline.
                If not provided, will be generated from "".
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            text_encoder_out_layers (`Tuple[int]`):
                Layer indices to use in the `text_encoder` to derive the final prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.flux2.Flux2PipelineOutput`] or `tuple`: [`~pipelines.flux2.Flux2PipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. prepare text embeddings
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if hasattr(self.transformer, "set_freefuse_txt_len"):
            self.transformer.set_freefuse_txt_len(prompt_embeds.shape[1])

        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        # 4. process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 7. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,  # (B, image_seq_len, C)
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,  # B, text_seq_len, 4
                        img_ids=latent_image_ids,  # B, image_seq_len, 4
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred[:, : latents.size(1) :]

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self._attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)
