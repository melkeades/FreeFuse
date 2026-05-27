"""
FreeFuse Attention Bias Patches for ComfyUI

Patches to inject attention bias into transformer blocks during generation.
Supports both Flux (joint attention) and SDXL (cross-attention) architectures.

For Flux:
- Uses block_replace to intercept the entire DoubleStreamBlock
- Modifies the attention computation to include bias

For SDXL:
- Uses attn2_replace to intercept cross-attention
- Applies bias to the Q-K attention scores
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Callable, List, Tuple
import logging
import os

# Z-Image (Lumina2 / NextDiT) helpers
import comfy.ops
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.lumina.model import modulate, apply_gate, clamp_fp16

from .attention_bias import (
    construct_attention_bias,
    construct_attention_bias_sdxl,
    apply_attention_bias_to_weights,
    get_attention_bias_for_layer,
    AttentionBiasConfig,
)


def _prepare_additive_attn_mask(
    mask: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Normalize bool/additive masks to SDPA's broadcastable additive layout."""
    if mask is None:
        return None

    if mask.dtype == torch.bool:
        bool_mask = mask.to(device=device)
        additive = torch.zeros(mask.shape, device=device, dtype=dtype)
        additive.masked_fill_(~bool_mask, torch.finfo(dtype).min)
        mask = additive
    elif mask.device != device or mask.dtype != dtype:
        mask = mask.to(device=device, dtype=dtype)

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask


def _freefuse_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor] = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Run attention through PyTorch/Comfy SDPA instead of xFormers.

    xFormers rejects dense tensor attention bias on some GPUs/drivers. FreeFuse
    attention bias is an additive dense mask, so route only biased calls through
    SDPA. Comfy/PyTorch can still choose FlashAttention-class kernels whenever
    that is valid for the shape/mask.
    """
    if skip_reshape:
        if q.dim() != 4:
            raise RuntimeError(
                f"Expected 4D q/k/v with skip_reshape=True, got q.shape={tuple(q.shape)}"
            )
        b = q.shape[0]
        dim_head = q.shape[-1]
        input_layout = "bhsd"
        if q.shape[1] == heads:
            q_4d, k_4d, v_4d = q, k, v
        elif q.shape[2] == heads:
            q_4d = q.transpose(1, 2)
            k_4d = k.transpose(1, 2)
            v_4d = v.transpose(1, 2)
            input_layout = "bshd"
        else:
            raise RuntimeError(
                f"Cannot infer head dimension for q.shape={tuple(q.shape)}, heads={heads}"
            )
    else:
        b, _, inner_dim = q.shape
        dim_head = inner_dim // heads
        q_4d, k_4d, v_4d = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )
        input_layout = "bhsd"

    mask = _prepare_additive_attn_mask(mask, q_4d.device, q_4d.dtype)

    sdpa_kwargs = {
        "attn_mask": mask,
        "dropout_p": 0.0,
        "is_causal": False,
    }
    if "scale" in kwargs:
        sdpa_kwargs["scale"] = kwargs["scale"]
    if "enable_gqa" in kwargs:
        sdpa_kwargs["enable_gqa"] = kwargs["enable_gqa"]

    try:
        out = comfy.ops.scaled_dot_product_attention(q_4d, k_4d, v_4d, **sdpa_kwargs)
    except TypeError:
        # Older wrappers may not expose newer SDPA kwargs such as enable_gqa.
        sdpa_kwargs.pop("enable_gqa", None)
        try:
            out = comfy.ops.scaled_dot_product_attention(q_4d, k_4d, v_4d, **sdpa_kwargs)
        except Exception:
            out = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, **sdpa_kwargs)
    except Exception:
        out = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, **sdpa_kwargs)

    if skip_output_reshape:
        if input_layout == "bshd":
            return out.transpose(1, 2)
        return out

    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def _make_freefuse_sdpa_attention_override(
    previous_override: Optional[Callable] = None,
) -> Callable:
    """Create an optimized_attention override that handles dense FreeFuse bias."""
    def attention_override(func: Callable, q, k, v, heads, *args, **kwargs):
        mask = kwargs.get("mask")
        if mask is None:
            if previous_override is not None:
                return previous_override(func, q, k, v, heads, *args, **kwargs)
            return func(q, k, v, heads, *args, **kwargs)

        sdpa_extra = {}
        if "scale" in kwargs:
            sdpa_extra["scale"] = kwargs["scale"]
        if "enable_gqa" in kwargs:
            sdpa_extra["enable_gqa"] = kwargs["enable_gqa"]

        return _freefuse_sdpa_attention(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=kwargs.get("skip_reshape", False),
            skip_output_reshape=kwargs.get("skip_output_reshape", False),
            **sdpa_extra,
        )

    return attention_override


def _call_block_with_attention_override(
    original_block: Callable,
    patched_args: Dict[str, Any],
    source_transformer_options: Any,
    attention_override: Callable,
):
    """Run a block with the override visible to copied args and closure-held options."""
    if not isinstance(source_transformer_options, dict):
        return original_block(patched_args)

    had_previous = "optimized_attention_override" in source_transformer_options
    previous_value = source_transformer_options.get("optimized_attention_override")
    source_transformer_options["optimized_attention_override"] = attention_override
    try:
        return original_block(patched_args)
    finally:
        if had_previous:
            source_transformer_options["optimized_attention_override"] = previous_value
        else:
            source_transformer_options.pop("optimized_attention_override", None)


class FreeFuseFluxBiasBlockReplace:
    """
    Block replace patch for Flux that applies attention bias during generation.
    
    This patch intercepts the attention computation in DoubleStreamBlock and adds
    the FreeFuse attention bias to guide cross-modal attention.
    
    The bias is applied to the pre-softmax attention scores:
        attn_weights = Q @ K^T / sqrt(d_k) + attention_bias
        attn_output = softmax(attn_weights) @ V
    
    IMPORTANT: Bias is constructed dynamically at runtime based on actual sequence
    lengths, as txt_seq_len can vary between tokenizers and models.
    """
    
    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List],
        config: AttentionBiasConfig,
        block_index: int,
        block=None,  # Actual DoubleStreamBlock reference
    ):
        """
        Args:
            lora_masks: Dict of LoRA name -> (B, img_seq_len) binary mask
            token_pos_maps: Dict of LoRA name -> [[token positions], ...]
            config: Attention bias configuration
            block_index: Index of this transformer block
            block: The actual DoubleStreamBlock reference (needed for modulation/FFN)
        """
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.block_index = block_index
        self.block = block  # Capture block reference
        # Cache for constructed bias (keyed by (txt_len, img_len))
        self._bias_cache = {}
        
    def _get_or_build_bias(
        self,
        txt_len: int,
        img_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Get cached bias or build new one for the given dimensions."""
        cache_key = (txt_len, img_len)
        if cache_key in self._bias_cache:
            bias = self._bias_cache[cache_key]
            if bias.device != device:
                bias = bias.to(device)
            if bias.dtype != dtype:
                bias = bias.to(dtype)
            return bias
        
        # Build new bias
        bias = construct_attention_bias(
            lora_masks=self.lora_masks,
            token_pos_maps=self.token_pos_maps,
            txt_seq_len=txt_len,
            img_seq_len=img_len,
            bias_scale=self.config.bias_scale,
            positive_bias_scale=self.config.positive_bias_scale,
            bidirectional=self.config.bidirectional,
            use_positive_bias=self.config.use_positive_bias,
            device=device,
            dtype=dtype,
        )
        
        if bias is not None:
            self._bias_cache[cache_key] = bias
            
        return bias
        
    def create_block_replace(self) -> Callable:
        """Create a block replace function that applies attention bias."""
        config = self.config
        block_index = self.block_index
        block = self.block  # Capture block reference from self
        # Capture self for dynamic bias construction
        bias_builder = self
        
        def block_replace(args: Dict, extra_args: Dict) -> Dict:
            """
            Replace function for DoubleStreamBlock with attention bias.
            
            We intercept the block, apply attention bias, and return modified output.
            """
            img = args["img"]
            txt = args["txt"]
            vec = args["vec"]
            pe = args["pe"]
            attn_mask = args.get("attn_mask")
            transformer_options = args.get("transformer_options", {})
            
            original_block = extra_args["original_block"]
            # block is captured from self.block, not from extra_args
            
            # Check if we should apply bias to this block
            block_name = f"transformer_blocks.{block_index}"
            if not config.should_apply_to_block(block_name):
                return original_block(args)
            
            # Check if block reference is valid
            if block is None:
                logging.warning(f"[FreeFuse] Block {block_index} is None, falling back to original")
                return original_block(args)
            
            # Get actual sequence lengths
            txt_len = txt.shape[1]
            img_len = img.shape[1]

            # Expose exact txt_len for downstream single blocks (no guessing)
            if isinstance(transformer_options, dict) and "txt_len" not in transformer_options:
                transformer_options["txt_len"] = txt_len
            
            # Dynamically build bias for actual dimensions
            attention_bias = bias_builder._get_or_build_bias(
                txt_len=txt_len,
                img_len=img_len,
                device=img.device,
                dtype=img.dtype,
            )
            
            if attention_bias is None:
                return original_block(args)
            
            try:
                bias_for_attn = attention_bias
                if bias_for_attn.device != img.device:
                    bias_for_attn = bias_for_attn.to(img.device)
                if bias_for_attn.dtype != img.dtype:
                    bias_for_attn = bias_for_attn.to(img.dtype)

                # If block expects [img, txt], reorder bias accordingly
                if getattr(block, 'flipped_img_txt', False):
                    img_len = img.shape[1]
                    txt_len = txt.shape[1]
                    perm = torch.cat([
                        torch.arange(txt_len, txt_len + img_len, device=bias_for_attn.device),
                        torch.arange(txt_len, device=bias_for_attn.device),
                    ])
                    bias_for_attn = bias_for_attn[:, perm][:, :, perm]

                # Ensure bias has heads dimension
                if bias_for_attn.dim() == 3:
                    bias_for_attn = bias_for_attn.unsqueeze(1)

                # Combine with attention mask if provided
                if attn_mask is not None:
                    mask = attn_mask
                    if mask.dtype == torch.bool:
                        float_mask = torch.zeros_like(mask, dtype=img.dtype)
                        float_mask.masked_fill_(~mask, torch.finfo(img.dtype).min)
                        mask = float_mask
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(1).unsqueeze(1)
                    elif mask.dim() == 3:
                        mask = mask.unsqueeze(1)
                    if mask.device != bias_for_attn.device:
                        mask = mask.to(bias_for_attn.device)
                    if mask.dtype != bias_for_attn.dtype:
                        mask = mask.to(bias_for_attn.dtype)
                    bias_for_attn = bias_for_attn + mask

                new_args = dict(args)
                new_args["attn_mask"] = bias_for_attn
                new_transformer_options = dict(transformer_options) if isinstance(transformer_options, dict) else {}
                previous_override = new_transformer_options.get("optimized_attention_override")
                attention_override = _make_freefuse_sdpa_attention_override(previous_override)
                new_transformer_options["optimized_attention_override"] = attention_override
                new_args["transformer_options"] = new_transformer_options
                return _call_block_with_attention_override(
                    original_block,
                    new_args,
                    transformer_options,
                    attention_override,
                )

            except Exception as e:
                logging.warning(f"[FreeFuse] Failed to apply attention bias at block {block_index}: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to original block
                return original_block(args)
        
        return block_replace


class FreeFuseFluxBiasSingleBlockReplace:
    """
    Block replace patch for Flux SingleStreamBlock with attention bias.
    
    SingleStreamBlock processes concatenated txt+img as a single sequence.
    The bias is applied to the full sequence attention.
    
    Key difference from DoubleStreamBlock:
    - Input x is already [txt, img] concatenated
    - We need to know txt_len to correctly apply the bias
    - txt_len is inferred from total_seq_len - img_seq_len (from masks)
    """
    
    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        config: AttentionBiasConfig,
        block_index: int,
        block=None,  # Actual SingleStreamBlock reference
    ):
        """
        Args:
            lora_masks: Spatial masks for each LoRA concept {name: (H, W)}
            token_pos_maps: Token positions for each concept {name: [[positions]]}
            config: Attention bias configuration
            block_index: Index of this transformer block
            block: The actual SingleStreamBlock reference
        """
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.block_index = block_index
        self.block = block
        
        # Cache for constructed bias (keyed by sequence length)
        self._bias_cache = {}
        
    def create_block_replace(self) -> Callable:
        """Create a block replace function for SingleStreamBlock."""
        lora_masks = self.lora_masks
        token_pos_maps = self.token_pos_maps
        config = self.config
        block_index = self.block_index
        block = self.block
        bias_cache = self._bias_cache
        
        def block_replace(args: Dict, extra_args: Dict) -> Dict:
            """
            Replace function for SingleStreamBlock with attention bias.
            
            Note: In ComfyUI's Flux model, single blocks receive the concatenated
            txt+img sequence via the 'img' key (not 'x'), and return {'img': output}.
            """
            # In ComfyUI Flux single block replace, the input is passed as 'img'
            # even though it's actually the concatenated [txt, img] sequence
            x = args["img"]  # This is actually [txt, img] concatenated
            vec = args["vec"]
            pe = args["pe"]
            attn_mask = args.get("attn_mask")
            transformer_options = args.get("transformer_options", {})
            
            original_block = extra_args["original_block"]
            
            block_name = f"single_transformer_blocks.{block_index}"
            if not config.should_apply_to_block(block_name):
                return original_block(args)
            
            if block is None:
                logging.warning(f"[FreeFuse] Single block {block_index} is None, falling back to original")
                return original_block(args)
            
            if not lora_masks:
                return original_block(args)
            
            try:
                B, total_seq_len, _ = x.shape

                # Require exact txt_len passed via transformer_options (no guessing).
                txt_len = transformer_options.get("txt_len") if isinstance(transformer_options, dict) else None

                if txt_len is None:
                    logging.warning(
                        f"[FreeFuse] Single block {block_index} missing txt_len in transformer_options; "
                        "skipping bias to avoid guessing."
                    )
                    return original_block(args)

                img_seq_len = total_seq_len - txt_len
                if img_seq_len <= 0:
                    logging.warning(
                        f"[FreeFuse] Single block {block_index} invalid lengths "
                        f"(total_seq_len={total_seq_len}, txt_len={txt_len}). Skipping bias."
                    )
                    return original_block(args)

                # Optional sanity check: log if mask length doesn't match derived img length
                first_mask = next(iter(lora_masks.values()))
                mask_seq_len = None
                if first_mask.dim() == 2:
                    mask_seq_len = first_mask.shape[1]
                elif first_mask.dim() == 1:
                    mask_seq_len = first_mask.shape[0]
                elif first_mask.dim() == 3:
                    mask_seq_len = first_mask.shape[1] * first_mask.shape[2]
                if mask_seq_len is not None and mask_seq_len != img_seq_len and block_index == 0:
                    logging.info(
                        f"[FreeFuse] Single block {block_index} mask len {mask_seq_len} "
                        f"!= img_seq_len {img_seq_len}; masks will be resized in bias construction."
                    )

                # Construct attention bias dynamically
                cache_key = (total_seq_len, txt_len, img_seq_len)
                if cache_key not in bias_cache:
                    attention_bias = construct_attention_bias(
                        lora_masks=lora_masks,
                        token_pos_maps=token_pos_maps,
                        txt_seq_len=txt_len,
                        img_seq_len=img_seq_len,
                        bias_scale=config.bias_scale,
                        positive_bias_scale=config.positive_bias_scale if config.use_positive_bias else 0.0,
                        bidirectional=config.bidirectional,
                        use_positive_bias=config.use_positive_bias,
                        device=x.device,
                        dtype=x.dtype,
                    )
                    if attention_bias is not None:
                        bias_cache[cache_key] = attention_bias
                        if block_index == 0:
                            logging.info(f"[FreeFuse] Single block bias: shape={attention_bias.shape}")
                    else:
                        bias_cache[cache_key] = None

                attention_bias = bias_cache.get(cache_key)
                if attention_bias is None:
                    return original_block(args)

                bias_for_attn = attention_bias
                if bias_for_attn.device != x.device:
                    bias_for_attn = bias_for_attn.to(x.device)
                if bias_for_attn.dtype != x.dtype:
                    bias_for_attn = bias_for_attn.to(x.dtype)

                # Handle size mismatch: pad or crop to total_seq_len
                if bias_for_attn.shape[-1] != total_seq_len:
                    if bias_for_attn.shape[-1] < total_seq_len:
                        pad_s = total_seq_len - bias_for_attn.shape[-1]
                        bias_for_attn = F.pad(bias_for_attn, (0, pad_s, 0, pad_s), value=0.0)
                    else:
                        bias_for_attn = bias_for_attn[:, :total_seq_len, :total_seq_len]

                # Ensure bias has heads dimension for broadcasting
                if bias_for_attn.dim() == 3:
                    bias_for_attn = bias_for_attn.unsqueeze(1)

                # Combine with attention mask if provided (keep additive semantics)
                if attn_mask is not None:
                    mask = attn_mask
                    if mask.dtype == torch.bool:
                        float_mask = torch.zeros_like(mask, dtype=x.dtype)
                        float_mask.masked_fill_(~mask, torch.finfo(x.dtype).min)
                        mask = float_mask
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(1).unsqueeze(1)
                    elif mask.dim() == 3:
                        mask = mask.unsqueeze(1)
                    if mask.device != bias_for_attn.device:
                        mask = mask.to(bias_for_attn.device)
                    if mask.dtype != bias_for_attn.dtype:
                        mask = mask.to(bias_for_attn.dtype)
                    bias_for_attn = bias_for_attn + mask

                new_args = dict(args)
                new_args["attn_mask"] = bias_for_attn
                new_transformer_options = dict(transformer_options) if isinstance(transformer_options, dict) else {}
                previous_override = new_transformer_options.get("optimized_attention_override")
                attention_override = _make_freefuse_sdpa_attention_override(previous_override)
                new_transformer_options["optimized_attention_override"] = attention_override
                new_args["transformer_options"] = new_transformer_options
                return _call_block_with_attention_override(
                    original_block,
                    new_args,
                    transformer_options,
                    attention_override,
                )

            except Exception as e:
                logging.warning(f"[FreeFuse] Failed to apply bias to single block {block_index}: {e}")
                import traceback
                traceback.print_exc()
                return original_block(args)
        
        return block_replace


class FreeFuseSDXLBiasAttnReplace:
    """
    Cross-attention replace patch for SDXL that applies attention bias.
    
    SDXL uses separate cross-attention where Q comes from image features
    and K/V come from text. The bias is applied to Q @ K^T scores.
    """
    
    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        config: AttentionBiasConfig,
        latent_size: Tuple[int, int],
    ):
        """
        Args:
            lora_masks: Spatial masks for each LoRA
            token_pos_maps: Token positions for each LoRA
            config: Attention bias configuration
            latent_size: (H, W) of the latent space
        """
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.latent_size = latent_size
        
        # Cache computed biases for different resolutions
        self._bias_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _infer_spatial_size(self, img_len: int) -> Tuple[int, int]:
        """Infer (img_h, img_w) from flattened img_len using latent_size aspect ratio.

        SDXL UNet preserves aspect ratio through all downsample stages.
        Each layer is (latent_h // d, latent_w // d) for d in {1, 2, 4, 8, ...}.
        We try these factors against the known latent_size first, then fall back
        to a generic aspect-ratio-preserving factorisation.
        """
        lat_h, lat_w = self.latent_size

        # Strategy 1: try power-of-2 downscale factors of latent_size
        for d in (1, 2, 4, 8, 16):
            h, w = lat_h // d, lat_w // d
            if h > 0 and w > 0 and h * w == img_len:
                return (h, w)

        # Strategy 2: aspect-ratio-preserving decomposition
        if lat_h > 0 and lat_w > 0:
            ratio = lat_w / lat_h
            h = max(int(round((img_len / max(ratio, 1e-8)) ** 0.5)), 1)
            w = img_len // h
            if h * w == img_len:
                return (h, w)

        # Strategy 3: try square
        side = int(round(img_len ** 0.5))
        if side * side == img_len:
            return (side, side)

        # Strategy 4: closest-to-square factor search (general fallback)
        import math as _math
        best_h, best_w = 1, img_len
        for h in range(1, int(_math.isqrt(img_len)) + 1):
            if img_len % h == 0:
                w = img_len // h
                if abs(w - h) < abs(best_w - best_h):
                    best_h, best_w = h, w
        return (best_h, best_w)

    def _get_bias_for_size(
        self,
        img_h: int,
        img_w: int,
        txt_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Get or compute attention bias for a specific feature map size."""
        cache_key = (img_h, img_w, txt_len)
        
        if cache_key not in self._bias_cache:
            bias = construct_attention_bias_sdxl(
                lora_masks=self.lora_masks,
                token_pos_maps=self.token_pos_maps,
                txt_seq_len=txt_len,
                img_h=img_h,
                img_w=img_w,
                bias_scale=self.config.bias_scale,
                positive_bias_scale=self.config.positive_bias_scale,
                use_positive_bias=self.config.use_positive_bias,
                device=device,
                dtype=dtype,
            )
            self._bias_cache[cache_key] = bias
        
        return self._bias_cache[cache_key]
    
    def create_attn_replace(self, block_name: str, block_num: int) -> Callable:
        """Create an attn2 replace function for a specific SDXL block."""
        config = self.config
        
        full_block_name = f"{block_name}.{block_num}"
        
        def attn2_replace(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            extra_options: Dict,
        ) -> torch.Tensor:
            """
            Replace function for SDXL cross-attention with bias.
            
            Args:
                q: Query from image (B*heads, img_len, head_dim) or (B, heads, img_len, head_dim)
                k: Key from text
                v: Value from text
                extra_options: Block info including n_heads, dim_head
            """
            if not config.should_apply_to_block(full_block_name):
                # Use standard attention without bias
                return _compute_standard_attention(q, k, v, extra_options)
            
            n_heads = extra_options.get("n_heads", 8)
            dim_head = extra_options.get("dim_head", 64)
            
            # Reshape if needed
            if q.dim() == 3:
                # ComfyUI passes q,k,v as (B, seq_len, n_heads*dim_head)
                # Reshape to (B, n_heads, seq_len, dim_head)
                batch_size = q.shape[0]
                img_len = q.shape[1]
                txt_len = k.shape[1]
                
                q = q.view(batch_size, img_len, n_heads, dim_head).transpose(1, 2)
                k = k.view(batch_size, txt_len, n_heads, dim_head).transpose(1, 2)
                v = v.view(batch_size, txt_len, n_heads, dim_head).transpose(1, 2)
            else:
                batch_size, _, img_len, _ = q.shape
                txt_len = k.shape[2]
            
            # Infer spatial size from img_len using latent_size aspect ratio
            img_h, img_w = self._infer_spatial_size(img_len)
            
            # Get bias for this resolution
            bias = self._get_bias_for_size(img_h, img_w, txt_len, q.device, q.dtype)
            
            # Compute attention
            scale = dim_head ** -0.5
            attn_weights = torch.einsum('bhid,bhjd->bhij', q, k) * scale
            
            # Apply bias
            if bias is not None:
                if bias.dim() == 3:
                    bias = bias.unsqueeze(1)  # Add heads dim
                attn_weights = attn_weights + bias
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.einsum('bhij,bhjd->bhid', attn_weights, v)
            
            # Reshape back
            out = out.transpose(1, 2).reshape(batch_size, img_len, n_heads * dim_head)
            
            return out
        
        return attn2_replace


def _compute_standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    extra_options: Dict,
) -> torch.Tensor:
    """Compute standard attention without bias (for blocks where bias is disabled)."""
    n_heads = extra_options.get("n_heads", 8)
    dim_head = extra_options.get("dim_head", 64)
    
    if q.dim() == 3:
        # ComfyUI passes q,k,v as (B, seq_len, n_heads*dim_head)
        # Reshape to (B, n_heads, seq_len, dim_head)
        batch_size = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_kv = k.shape[1]
        
        q = q.view(batch_size, seq_len_q, n_heads, dim_head).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, n_heads, dim_head).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, n_heads, dim_head).transpose(1, 2)
    else:
        batch_size = q.shape[0]
        seq_len_q = q.shape[2]
    
    scale = dim_head ** -0.5
    attn_weights = torch.einsum('bhid,bhjd->bhij', q, k) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    out = torch.einsum('bhij,bhjd->bhid', attn_weights, v)
    out = out.transpose(1, 2).reshape(batch_size, seq_len_q, n_heads * dim_head)
    
    return out


class FreeFuseZImageBiasBlockReplace:
    """
    Block replace patch for Z-Image (Lumina2/NextDiT) that applies attention bias.
    
    In ComfyUI's NextDiT, the unified sequence is [txt, img] (text FIRST, then image).
    This matches the standard construct_attention_bias output layout [txt, img],
    so NO permutation is needed.
    """
    
    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        config: AttentionBiasConfig,
        block_index: int,
        block=None,  # Actual JointTransformerBlock reference
    ):
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.block_index = block_index
        self.block = block
        self._bias_cache = {}
    
    def _get_or_build_bias(
        self,
        img_len: int,
        cap_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Get cached bias or build new one in [txt, img] layout.
        
        In ComfyUI's NextDiT, the unified sequence is [txt, img] (text FIRST),
        which matches construct_attention_bias's output layout directly.
        No permutation is needed.
        """
        cache_key = (img_len, cap_len)
        if cache_key in self._bias_cache:
            bias = self._bias_cache[cache_key]
            return bias.to(device=device, dtype=dtype)
        
        # Build in [txt, img] layout — matches ComfyUI's NextDiT sequence order
        bias = construct_attention_bias(
            lora_masks=self.lora_masks,
            token_pos_maps=self.token_pos_maps,
            txt_seq_len=cap_len,
            img_seq_len=img_len,
            bias_scale=self.config.bias_scale,
            positive_bias_scale=self.config.positive_bias_scale,
            bidirectional=self.config.bidirectional,
            use_positive_bias=self.config.use_positive_bias,
            device=device,
            dtype=dtype,
        )
        
        if bias is None:
            self._bias_cache[cache_key] = None
            return None
        
        # No permutation needed: ComfyUI's NextDiT uses [txt, img] order
        # which matches construct_attention_bias's output layout directly
        self._bias_cache[cache_key] = bias

        if os.environ.get("FREEFUSE_DEBUG_ZIMAGE") == "1":
            try:
                print(
                    "[FreeFuse Z-Image Debug] attention_bias "
                    f"(cap_len={cap_len}, img_len={img_len}) "
                    f"min={bias.min().item():.4f} max={bias.max().item():.4f} "
                    f"mean={bias.mean().item():.4f}"
                )
            except Exception:
                pass
        return bias

    def _attention_with_bias(
        self,
        attention,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        transformer_options: Dict[str, Any],
        attention_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute JointAttention with additive bias, mirroring ComfyUI's Lumina2 implementation.

        This reproduces:
          - qkv projection
          - q_norm / k_norm
          - complex RoPE (apply_rope)
          - GQA expansion
          - scaled dot-product attention with optional additive bias + key mask
          - out projection
        """
        bsz, seqlen, _ = x.shape

        # QKV
        qkv = attention.qkv(x)
        n_heads = attention.n_local_heads
        n_kv_heads = attention.n_local_kv_heads
        head_dim = attention.head_dim

        xq, xk, xv = torch.split(
            qkv,
            [n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, n_heads, head_dim)
        xk = xk.view(bsz, seqlen, n_kv_heads, head_dim)
        xv = xv.view(bsz, seqlen, n_kv_heads, head_dim)

        # QK norm
        xq = attention.q_norm(xq)
        xk = attention.k_norm(xk)

        # RoPE (complex-valued)
        if freqs_cis is not None:
            xq, xk = apply_rope(xq, xk, freqs_cis)

        # GQA expansion
        n_rep = n_heads // n_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # (B, heads, seq, head_dim)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # Build attention mask: additive bias (+ optional key mask)
        attn_mask = None
        if attention_bias is not None:
            bias = attention_bias
            if bias.dim() == 3:
                bias = bias.unsqueeze(1)
            bias = bias.to(device=q.device, dtype=q.dtype)
            attn_mask = bias

        if x_mask is not None:
            # key mask: (B, seq) -> (B, 1, 1, seq) with -inf for padding
            key_mask = torch.zeros(
                bsz, 1, 1, seqlen, device=q.device, dtype=q.dtype
            )
            key_mask.masked_fill_(
                ~x_mask.bool().unsqueeze(1).unsqueeze(1),
                torch.finfo(q.dtype).min,
            )
            attn_mask = key_mask if attn_mask is None else attn_mask + key_mask

        # Use SDPA with additive mask (float)
        attn_out = comfy.ops.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # (B, seq, dim)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seqlen, n_heads * head_dim)
        return attention.out(attn_out)
    
    def create_block_replace(self) -> Callable:
        """Create a block replace function for Z-Image JointTransformerBlock."""
        config = self.config
        block_index = self.block_index
        block = self.block
        bias_builder = self
        
        def block_replace(args: Dict, extra_args: Dict) -> Dict:
            """
            Replace function for Z-Image JointTransformerBlock with attention bias.
            """
            x = args["x"]  # (B, seq_len, dim) = [txt + img] in ComfyUI Lumina2
            x_mask = args.get("x_mask")
            freqs_cis = args["freqs_cis"]
            adaln_input = args.get("adaln_input")
            timestep_zero_index = args.get("timestep_zero_index")
            transformer_options = args.get("transformer_options", {})
            
            original_block = extra_args["original_block"]
            
            # Check if we should apply bias to this block
            block_name = f"layers.{block_index}"
            if not config.should_apply_to_block(block_name):
                return original_block(args)
            
            if block is None:
                logging.warning(f"[FreeFuse Z-Image] Block {block_index} is None, falling back")
                return original_block(args)
            
            try:
                bsz, seqlen, dim = x.shape
                
                # Get sequence lengths from transformer_options or infer
                img_seq_len = transformer_options.get("img_seq_len")
                cap_seq_len = transformer_options.get("cap_seq_len")
                
                if img_seq_len is None or cap_seq_len is None:
                    # Infer from lora_masks
                    if bias_builder.lora_masks:
                        first_mask = next(iter(bias_builder.lora_masks.values()))
                        if first_mask.dim() == 2:
                            img_seq_len = first_mask.shape[1]
                        else:
                            img_seq_len = first_mask.numel()
                        cap_seq_len = seqlen - img_seq_len
                    else:
                        logging.warning(f"[FreeFuse Z-Image] Cannot infer sequence lengths at block {block_index}")
                        return original_block(args)
                
                if cap_seq_len <= 0:
                    return original_block(args)
                
                # Build attention bias in [txt, img] layout (matches ComfyUI's NextDiT sequence order)
                attention_bias = bias_builder._get_or_build_bias(
                    img_len=img_seq_len,
                    cap_len=cap_seq_len,
                    device=x.device,
                    dtype=x.dtype,
                )
                
                if attention_bias is None:
                    return original_block(args)
                
                # === Execute block with modified attention (match ComfyUI Lumina2) ===
                attention = block.attention

                if block.modulation:
                    if adaln_input is None:
                        return original_block(args)
                    scale_msa, gate_msa, scale_mlp, gate_mlp = block.adaLN_modulation(adaln_input).chunk(4, dim=1)

                    attn_in = modulate(block.attention_norm1(x), scale_msa, timestep_zero_index=timestep_zero_index)
                    attn_out = self._attention_with_bias(
                        attention, attn_in, x_mask, freqs_cis, transformer_options, attention_bias
                    )
                    attn_out = block.attention_norm2(clamp_fp16(attn_out))
                    x = x + apply_gate(gate_msa.unsqueeze(1).tanh(), attn_out, timestep_zero_index=timestep_zero_index)

                    ff_in = modulate(block.ffn_norm1(x), scale_mlp, timestep_zero_index=timestep_zero_index)
                    ff_out = block.feed_forward(ff_in)
                    ff_out = block.ffn_norm2(clamp_fp16(ff_out))
                    x = x + apply_gate(gate_mlp.unsqueeze(1).tanh(), ff_out, timestep_zero_index=timestep_zero_index)
                else:
                    # No modulation: match original non-modulated path
                    attn_in = block.attention_norm1(x)
                    attn_out = self._attention_with_bias(
                        attention, attn_in, x_mask, freqs_cis, transformer_options, attention_bias
                    )
                    x = x + block.attention_norm2(clamp_fp16(attn_out))

                    ff_out = block.feed_forward(block.ffn_norm1(x))
                    x = x + block.ffn_norm2(clamp_fp16(ff_out))

                return {"x": x}
                
            except Exception as e:
                logging.warning(f"[FreeFuse Z-Image] Failed to apply attention bias at block {block_index}: {e}")
                import traceback
                traceback.print_exc()
                return original_block(args)
        
        return block_replace


class FreeFuseQwenImageBiasBlockReplace:
    """
    Block replace patch for ComfyUI native Qwen-Image attention bias.

    Qwen-Image transformer blocks use dual streams but joint attention is built
    in [txt, img] order, matching construct_attention_bias() directly.
    """

    def __init__(
        self,
        lora_masks: Dict[str, torch.Tensor],
        token_pos_maps: Dict[str, List[List[int]]],
        config: AttentionBiasConfig,
        block_index: int,
        block=None,
    ):
        self.lora_masks = lora_masks
        self.token_pos_maps = token_pos_maps
        self.config = config
        self.block_index = block_index
        self.block = block
        self._bias_cache = {}

    def _get_or_build_bias(
        self,
        txt_len: int,
        img_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        cache_key = (txt_len, img_len)
        if cache_key in self._bias_cache:
            bias = self._bias_cache[cache_key]
            if bias is None:
                return None
            return bias.to(device=device, dtype=dtype)

        bias = construct_attention_bias(
            lora_masks=self.lora_masks,
            token_pos_maps=self.token_pos_maps,
            txt_seq_len=txt_len,
            img_seq_len=img_len,
            bias_scale=self.config.bias_scale,
            positive_bias_scale=self.config.positive_bias_scale,
            bidirectional=self.config.bidirectional,
            use_positive_bias=self.config.use_positive_bias,
            device=device,
            dtype=dtype,
        )
        self._bias_cache[cache_key] = bias
        return bias

    @staticmethod
    def _attention_with_bias(
        attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        image_rotary_emb: Optional[torch.Tensor],
        transformer_options: Dict[str, Any],
        attention_bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from comfy.ldm.flux.math import apply_rope1

        batch_size = hidden_states.shape[0]
        seq_img = hidden_states.shape[1]
        seq_txt = encoder_hidden_states.shape[1]
        heads = attention.heads

        img_query = attention.to_q(hidden_states).view(batch_size, seq_img, heads, -1).transpose(1, 2).contiguous()
        img_key = attention.to_k(hidden_states).view(batch_size, seq_img, heads, -1).transpose(1, 2).contiguous()
        img_value = attention.to_v(hidden_states).view(batch_size, seq_img, heads, -1).transpose(1, 2)

        txt_query = attention.add_q_proj(encoder_hidden_states).view(batch_size, seq_txt, heads, -1).transpose(1, 2).contiguous()
        txt_key = attention.add_k_proj(encoder_hidden_states).view(batch_size, seq_txt, heads, -1).transpose(1, 2).contiguous()
        txt_value = attention.add_v_proj(encoder_hidden_states).view(batch_size, seq_txt, heads, -1).transpose(1, 2)

        img_query = attention.norm_q(img_query)
        img_key = attention.norm_k(img_key)
        txt_query = attention.norm_added_q(txt_query)
        txt_key = attention.norm_added_k(txt_key)

        joint_query = torch.cat([txt_query, img_query], dim=2)
        joint_key = torch.cat([txt_key, img_key], dim=2)
        joint_value = torch.cat([txt_value, img_value], dim=2)

        if image_rotary_emb is not None:
            joint_query = apply_rope1(joint_query, image_rotary_emb)
            joint_key = apply_rope1(joint_key, image_rotary_emb)

        attn_mask = None
        if attention_bias is not None:
            attn_mask = attention_bias
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.to(device=joint_query.device, dtype=joint_query.dtype)

        if encoder_hidden_states_mask is not None:
            key_mask = torch.zeros(
                batch_size,
                1,
                1,
                seq_txt + seq_img,
                device=joint_query.device,
                dtype=joint_query.dtype,
            )
            mask = encoder_hidden_states_mask
            if mask.dtype == torch.bool:
                key_mask[:, :, :, :seq_txt].masked_fill_(
                    ~mask.unsqueeze(1).unsqueeze(1),
                    torch.finfo(joint_query.dtype).min,
                )
            else:
                key_mask[:, :, :, :seq_txt] = mask.to(
                    device=joint_query.device,
                    dtype=joint_query.dtype,
                ).unsqueeze(1).unsqueeze(1)
            attn_mask = key_mask if attn_mask is None else attn_mask + key_mask

        try:
            joint_hidden_states = comfy.ops.scaled_dot_product_attention(
                joint_query,
                joint_key,
                joint_value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        except Exception:
            joint_hidden_states = F.scaled_dot_product_attention(
                joint_query,
                joint_key,
                joint_value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        joint_hidden_states = joint_hidden_states.transpose(1, 2).reshape(
            batch_size, seq_txt + seq_img, heads * joint_value.shape[-1]
        )
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attention.to_out[0](img_attn_output)
        img_attn_output = attention.to_out[1](img_attn_output)
        txt_attn_output = attention.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    def create_block_replace(self) -> Callable:
        config = self.config
        block_index = self.block_index
        block = self.block
        bias_builder = self

        def block_replace(args: Dict, extra_args: Dict) -> Dict:
            hidden_states = args["img"]
            encoder_hidden_states = args["txt"]
            temb = args["vec"]
            image_rotary_emb = args["pe"]
            timestep_zero_index = args.get("timestep_zero_index")
            transformer_options = args.get("transformer_options", {})
            encoder_hidden_states_mask = args.get("encoder_hidden_states_mask")
            original_block = extra_args["original_block"]

            block_name = f"transformer_blocks.{block_index}"
            if not config.should_apply_to_block(block_name):
                return original_block(args)

            if block is None:
                logging.warning(f"[FreeFuse Qwen-Image] Block {block_index} is None, falling back")
                return original_block(args)

            try:
                txt_len = encoder_hidden_states.shape[1]
                img_len = hidden_states.shape[1]
                attention_bias = bias_builder._get_or_build_bias(
                    txt_len=txt_len,
                    img_len=img_len,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                if attention_bias is None:
                    return original_block(args)

                img_mod_params = block.img_mod(temb)
                txt_temb = temb.chunk(2, dim=0)[0] if timestep_zero_index is not None else temb
                txt_mod_params = block.txt_mod(txt_temb)
                img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
                txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

                img_modulated, img_gate1 = block._modulate(
                    block.img_norm1(hidden_states), img_mod1, timestep_zero_index
                )
                txt_modulated, txt_gate1 = block._modulate(
                    block.txt_norm1(encoder_hidden_states), txt_mod1
                )

                img_attn_output, txt_attn_output = self._attention_with_bias(
                    block.attn,
                    img_modulated,
                    txt_modulated,
                    encoder_hidden_states_mask,
                    image_rotary_emb,
                    transformer_options,
                    attention_bias,
                )

                hidden_states = block._apply_gate(
                    img_attn_output, hidden_states, img_gate1, timestep_zero_index
                )
                encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

                img_modulated2, img_gate2 = block._modulate(
                    block.img_norm2(hidden_states), img_mod2, timestep_zero_index
                )
                hidden_states = block._apply_gate(
                    block.img_mlp(img_modulated2), hidden_states, img_gate2, timestep_zero_index
                )

                txt_modulated2, txt_gate2 = block._modulate(
                    block.txt_norm2(encoder_hidden_states), txt_mod2
                )
                encoder_hidden_states = torch.addcmul(
                    encoder_hidden_states, txt_gate2, block.txt_mlp(txt_modulated2)
                )

                return {"img": hidden_states, "txt": encoder_hidden_states}

            except Exception as e:
                logging.warning(f"[FreeFuse Qwen-Image] Failed to apply attention bias at block {block_index}: {e}")
                import traceback
                traceback.print_exc()
                return original_block(args)

        return block_replace


def apply_attention_bias_patches(
    model_patcher,
    attention_bias: torch.Tensor,
    config: AttentionBiasConfig,
    txt_seq_len: int,
    model_type: str = "flux",
    lora_masks: Dict[str, torch.Tensor] = None,
    token_pos_maps: Dict[str, List[List[int]]] = None,
    latent_size: Tuple[int, int] = None,
):
    """
    Apply attention bias patches to a model.
    
    Args:
        model_patcher: ComfyUI ModelPatcher instance
        attention_bias: Pre-computed attention bias for Flux (ignored - use lora_masks instead)
        config: Attention bias configuration
        txt_seq_len: Text sequence length estimate (not used for Flux, actual length determined at runtime)
        model_type: "flux", "flux2", "z_image", "qwen_image", or "sdxl"
        lora_masks: Spatial masks for each LoRA (flattened to img_seq_len)
        token_pos_maps: Token positions for each LoRA
        latent_size: Required for SDXL - (H, W) of latent space
    """
    if not config.enabled:
        logging.info("[FreeFuse] Attention bias disabled, skipping patches")
        return
    
    if model_type in ("flux", "flux2"):
        _apply_flux_bias_patches(model_patcher, lora_masks, token_pos_maps, config)
    elif model_type == "qwen_image":
        if lora_masks is None or token_pos_maps is None:
            logging.warning("[FreeFuse] Qwen-Image attention bias requires lora_masks and token_pos_maps")
            return
        _apply_qwen_image_bias_patches(model_patcher, lora_masks, token_pos_maps, config)
    elif model_type == "z_image":
        if lora_masks is None or token_pos_maps is None:
            logging.warning("[FreeFuse] Z-Image attention bias requires lora_masks and token_pos_maps")
            return
        _apply_z_image_bias_patches(model_patcher, lora_masks, token_pos_maps, config)
    elif model_type == "sdxl":
        if lora_masks is None or token_pos_maps is None or latent_size is None:
            logging.warning("[FreeFuse] SDXL attention bias requires lora_masks, token_pos_maps, and latent_size")
            return
        _apply_sdxl_bias_patches(model_patcher, lora_masks, token_pos_maps, config, latent_size)
    else:
        logging.warning(f"[FreeFuse] Unknown model type for attention bias: {model_type}")


def _apply_flux_bias_patches(
    model_patcher,
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    config: AttentionBiasConfig,
):
    """Apply attention bias patches for Flux model."""
    if lora_masks is None or not lora_masks:
        logging.warning("[FreeFuse] No LoRA masks provided for Flux attention bias")
        return
    
    if token_pos_maps is None or not token_pos_maps:
        logging.warning("[FreeFuse] No token position maps provided for Flux attention bias")
        return
    
    # Get model structure
    try:
        diffusion_model = model_patcher.model.diffusion_model
        num_double_blocks = len(diffusion_model.double_blocks)
        num_single_blocks = len(diffusion_model.single_blocks)
    except AttributeError:
        logging.warning("[FreeFuse] Could not access Flux model structure")
        return
    
    double_patches = 0
    single_patches = 0
    
    # Apply to double blocks
    for i in range(num_double_blocks):
        block_name = f"transformer_blocks.{i}"
        if config.should_apply_to_block(block_name):
            block = diffusion_model.double_blocks[i]
            replacer = FreeFuseFluxBiasBlockReplace(
                lora_masks=lora_masks,
                token_pos_maps=token_pos_maps,
                config=config,
                block_index=i,
                block=block,  # Pass actual block reference
            )
            model_patcher.set_model_patch_replace(
                replacer.create_block_replace(),
                "dit",
                "double_block",
                i,
            )
            double_patches += 1
    
    # Apply to single blocks
    for i in range(num_single_blocks):
        block_name = f"single_transformer_blocks.{i}"
        if config.should_apply_to_block(block_name):
            block = diffusion_model.single_blocks[i]
            replacer = FreeFuseFluxBiasSingleBlockReplace(
                lora_masks=lora_masks,
                token_pos_maps=token_pos_maps,
                config=config,
                block_index=i,
                block=block,  # Pass actual block reference
            )
            model_patcher.set_model_patch_replace(
                replacer.create_block_replace(),
                "dit",
                "single_block",
                i,
            )
            single_patches += 1
    
    logging.info(f"[FreeFuse] Applied attention bias to {double_patches} double blocks + {single_patches} single blocks")


def _apply_sdxl_bias_patches(
    model_patcher,
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    config: AttentionBiasConfig,
    latent_size: Tuple[int, int],
):
    """Apply attention bias patches for SDXL model."""
    replacer = FreeFuseSDXLBiasAttnReplace(
        lora_masks=lora_masks,
        token_pos_maps=token_pos_maps,
        config=config,
        latent_size=latent_size,
    )
    
    patches_applied = 0
    
    # SDXL UNet structure: input_blocks, middle_block, output_blocks
    # We apply to cross-attention (attn2) in each
    
    # Define SDXL block structure (typical for SDXL)
    sdxl_blocks = [
        # Input blocks with cross-attention
        ("input", 1), ("input", 2),
        ("input", 4), ("input", 5),
        ("input", 7), ("input", 8),
        # Middle block
        ("middle", 0),
        # Output blocks with cross-attention
        ("output", 0), ("output", 1), ("output", 2),
        ("output", 3), ("output", 4), ("output", 5),
        ("output", 6), ("output", 7), ("output", 8),
    ]
    
    for block_name, block_num in sdxl_blocks:
        full_name = f"{block_name}.{block_num}"
        if config.should_apply_to_block(full_name):
            model_patcher.set_model_attn2_replace(
                replacer.create_attn_replace(block_name, block_num),
                block_name,
                block_num,
            )
            patches_applied += 1
    
    logging.info(f"[FreeFuse] Applied attention bias to {patches_applied} SDXL cross-attention blocks")


def _apply_z_image_bias_patches(
    model_patcher,
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    config: AttentionBiasConfig,
):
    """Apply attention bias patches for Z-Image (Lumina2/NextDiT) model."""
    if not lora_masks:
        logging.warning("[FreeFuse] No LoRA masks provided for Z-Image attention bias")
        return
    
    if not token_pos_maps:
        logging.warning("[FreeFuse] No token position maps provided for Z-Image attention bias")
        return
    
    try:
        diffusion_model = model_patcher.model.diffusion_model
        num_layers = len(diffusion_model.layers)
    except AttributeError:
        logging.warning("[FreeFuse] Could not access Z-Image model structure")
        return

    # Resolve Z-Image-specific block presets
    # ComfyUI Lumina2 blocks are named: layers.{i}
    apply_to_blocks = config.apply_to_blocks
    if isinstance(apply_to_blocks, str):
        preset = apply_to_blocks
        # Flux/SDXL presets don't apply to Z-Image; map to last_half by default
        if preset in ("double_stream_only", "single_stream_only", "last_half_double"):
            preset = "last_half"
        if preset == "all":
            apply_to_blocks = None
        elif preset == "last_half":
            apply_to_blocks = [f"layers.{i}" for i in range(num_layers // 2, num_layers)]
        else:
            apply_to_blocks = [preset]
        config = AttentionBiasConfig(
            enabled=config.enabled,
            bias_scale=config.bias_scale,
            positive_bias_scale=config.positive_bias_scale,
            bidirectional=config.bidirectional,
            use_positive_bias=config.use_positive_bias,
            apply_to_blocks=apply_to_blocks,
        )
    elif isinstance(apply_to_blocks, list):
        if "last_half" in apply_to_blocks:
            apply_to_blocks = [f"layers.{i}" for i in range(num_layers // 2, num_layers)]
            config = AttentionBiasConfig(
                enabled=config.enabled,
                bias_scale=config.bias_scale,
                positive_bias_scale=config.positive_bias_scale,
                bidirectional=config.bidirectional,
                use_positive_bias=config.use_positive_bias,
                apply_to_blocks=apply_to_blocks,
            )
    
    patches_applied = 0
    
    for i in range(num_layers):
        block_name = f"layers.{i}"
        if config.should_apply_to_block(block_name):
            block = diffusion_model.layers[i]
            replacer = FreeFuseZImageBiasBlockReplace(
                lora_masks=lora_masks,
                token_pos_maps=token_pos_maps,
                config=config,
                block_index=i,
                block=block,
            )
            model_patcher.set_model_patch_replace(
                replacer.create_block_replace(),
                "dit",
                "layer",
                i,
            )
            patches_applied += 1
    
    logging.info(f"[FreeFuse] Applied attention bias to {patches_applied} Z-Image transformer layers")


def _resolve_qwen_image_bias_config(
    config: AttentionBiasConfig,
    num_layers: int,
) -> AttentionBiasConfig:
    """Map generic bias block presets onto Qwen transformer_blocks.N names."""
    apply_to_blocks = config.apply_to_blocks

    if isinstance(apply_to_blocks, str):
        preset = apply_to_blocks
        if preset in ("all", "double_stream_only"):
            apply_to_blocks = None
        elif preset in ("last_half_double", "last_half"):
            apply_to_blocks = [
                f"transformer_blocks.{i}"
                for i in range(num_layers // 2, num_layers)
            ]
        elif preset == "single_stream_only":
            apply_to_blocks = []
        else:
            apply_to_blocks = [preset]
    elif isinstance(apply_to_blocks, list):
        if any(p in apply_to_blocks for p in ("all", "double_stream_only")):
            apply_to_blocks = None
        elif any(p in apply_to_blocks for p in ("last_half_double", "last_half")):
            apply_to_blocks = [
                f"transformer_blocks.{i}"
                for i in range(num_layers // 2, num_layers)
            ]
        elif apply_to_blocks == ["single_stream_only"]:
            apply_to_blocks = []

    return AttentionBiasConfig(
        enabled=config.enabled,
        bias_scale=config.bias_scale,
        positive_bias_scale=config.positive_bias_scale,
        bidirectional=config.bidirectional,
        use_positive_bias=config.use_positive_bias,
        apply_to_blocks=apply_to_blocks,
    )


def _apply_qwen_image_bias_patches(
    model_patcher,
    lora_masks: Dict[str, torch.Tensor],
    token_pos_maps: Dict[str, List[List[int]]],
    config: AttentionBiasConfig,
):
    """Apply attention bias patches for native Qwen-Image transformer blocks."""
    if not lora_masks:
        logging.warning("[FreeFuse] No LoRA masks provided for Qwen-Image attention bias")
        return

    if not token_pos_maps:
        logging.warning("[FreeFuse] No token position maps provided for Qwen-Image attention bias")
        return

    try:
        diffusion_model = model_patcher.model.diffusion_model
        transformer_blocks = diffusion_model.transformer_blocks
        num_layers = len(transformer_blocks)
    except AttributeError:
        logging.warning("[FreeFuse] Could not access Qwen-Image transformer_blocks")
        return

    qwen_config = _resolve_qwen_image_bias_config(config, num_layers)

    patches_applied = 0
    for i in range(num_layers):
        block_name = f"transformer_blocks.{i}"
        if qwen_config.should_apply_to_block(block_name):
            block = transformer_blocks[i]
            replacer = FreeFuseQwenImageBiasBlockReplace(
                lora_masks=lora_masks,
                token_pos_maps=token_pos_maps,
                config=qwen_config,
                block_index=i,
                block=block,
            )
            model_patcher.set_model_patch_replace(
                replacer.create_block_replace(),
                "dit",
                "double_block",
                i,
            )
            patches_applied += 1

    logging.info(
        f"[FreeFuse] Applied attention bias to {patches_applied} Qwen-Image transformer blocks"
    )
