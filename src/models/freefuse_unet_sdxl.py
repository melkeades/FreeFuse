# Copyright 2025 The FreeFuse Team. All rights reserved.
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
FreeFuse UNet2DConditionModel for SDXL

This module provides a UNet wrapper that adds FreeFuse mask management methods.
It handles multi-scale masks for different UNet blocks (down/mid/up).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import logging

from src.tuner.freefuse_lora_layer import FreeFuseLinear


logger = logging.get_logger(__name__)


class FreeFuseUNet2DConditionModel(UNet2DConditionModel):
    """
    FreeFuse wrapper for UNet2DConditionModel.
    
    Adds methods for setting FreeFuse masks and token position maps to LoRA layers.
    Handles multi-scale mask interpolation for different UNet resolutions.
    
    SDXL UNet has the following resolution structure (for 1024x1024 input):
    - down_blocks: 128 -> 64 -> 32 -> 32 (no further downsampling in last block)
    - mid_block: 32
    - up_blocks: 32 -> 64 -> 128 -> 128
    
    The mask is typically collected at one scale and then interpolated to match
    each layer's feature map resolution.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freefuse_masks: Optional[Dict[str, torch.Tensor]] = None
        self.freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None
        # Store the original mask height/width for interpolation calculations
        self._mask_original_hw: Optional[Tuple[int, int]] = None
    
    @classmethod
    def from_unet(cls, unet: UNet2DConditionModel) -> "FreeFuseUNet2DConditionModel":
        """
        Create a FreeFuseUNet2DConditionModel from an existing UNet2DConditionModel.
        
        This method works by replacing the __class__ attribute, avoiding expensive
        weight copying or re-initialization.
        
        Args:
            unet: The original UNet2DConditionModel
            
        Returns:
            The same object but with FreeFuseUNet2DConditionModel class and methods
        """
        # Simply change the class - this is safe because we only add methods
        unet.__class__ = cls
        # Initialize FreeFuse-specific attributes
        unet.freefuse_masks = None
        unet.freefuse_token_pos_maps = None
        unet._mask_original_hw = None
        return unet
    
    def set_freefuse_masks(
        self, 
        freefuse_masks: Dict[str, torch.Tensor], 
        mask_height: int,
        mask_width: int,
        h: int,
        w: int,
        set_to_lora: bool = True,
        derive_01_mask: bool = False,
    ):
        """
        Set FreeFuse masks to all FreeFuseLinear layers in the UNet.
        
        Args:
            freefuse_masks: Dict mapping lora_name -> mask tensor of shape (B, 1, H, W) or (B, H, W)
            mask_height: Height of the mask (for interpolation reference)
            mask_width: Width of the mask (for interpolation reference)
            set_to_lora: Whether to set masks to LoRA layers immediately
            derive_01_mask: Whether to convert soft masks to binary 0-1 masks
        """
        self.freefuse_masks = freefuse_masks
        self._mask_original_hw = (mask_height, mask_width)
        
        # Ensure masks are in (B, 1, H, W) format
        normalized_masks = {}
        for name, mask in freefuse_masks.items():
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            normalized_masks[name] = mask
        
        # Convert to 0-1 mask if requested
        if derive_01_mask and normalized_masks:
            normalized_masks = self._convert_to_01_mask(normalized_masks)
        
        if set_to_lora:
            self._distribute_masks_to_lora_layers(normalized_masks, h, w)
    
    def set_freefuse_token_pos_maps(self, freefuse_token_pos_maps: Dict[str, List[List[int]]]):
        """
        Set token position maps to all FreeFuseLinear layers.
        
        Token position maps are used to ensure each LoRA only affects its own
        concept tokens in the text encoder output.
        
        Args:
            freefuse_token_pos_maps: Dict mapping lora_name -> [[positions for prompt 1], ...]
        """
        self.freefuse_token_pos_maps = freefuse_token_pos_maps
        
        for name, module in self.named_modules():
            if isinstance(module, FreeFuseLinear):
                # Apply to cross-attention layers (attn2) that process text
                if 'attn2' in name:
                    module.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
    
    def _distribute_masks_to_lora_layers(self, masks: Dict[str, torch.Tensor], h, w):
        """
        Distribute masks to all FreeFuseLinear layers with appropriate scaling.
        
        SDXL UNet blocks have different feature map resolutions. This method
        interpolates the mask to match each layer's expected resolution.
        
        Block structure (for 1024x1024 image with 8x VAE downsampling = 128x128 latent):
        - down_blocks.0: 128x128 -> attentions work on HxW sequence
        - down_blocks.1: 64x64
        - down_blocks.2: 32x32 (has cross-attention)
        - mid_block: 32x32 (has cross-attention)
        - up_blocks.0: 32x32 (has cross-attention)
        - up_blocks.1: 64x64
        - up_blocks.2: 128x128
        """
        if not masks:
            return
        
        # Get resolution info from one mask
        sample_mask = next(iter(masks.values()))
        mask_h = sample_mask.shape[2]
        mask_w = sample_mask.shape[3]
        
        for name, module in self.named_modules():
            if isinstance(module, FreeFuseLinear):
                # Determine the feature map resolution for this layer
                layer_hw = self._get_layer_resolution(name, h, w)
                if layer_hw is None:
                    continue
                
                layer_h, layer_w = layer_hw
                
                # Interpolate mask to match layer resolution
                scaled_masks = {}
                for lora_name, mask in masks.items():
                    if (mask.shape[2], mask.shape[3]) == (layer_h, layer_w):
                        scaled_mask = mask
                    else:
                        scaled_mask = F.interpolate(
                            mask.float(), 
                            size=(layer_h, layer_w), 
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # Flatten to (B, H*W, 1) for linear layer masking
                    scaled_masks[lora_name] = scaled_mask.reshape(
                        scaled_mask.shape[0], -1, 1
                    )
                
                module.set_freefuse_masks(scaled_masks)
    
    def _get_layer_resolution(self, layer_name: str, latent_h: int, latent_w: int) -> Optional[Tuple[int, int]]:
        """
        Get the feature map resolution for a specific layer.
        
        SDXL UNet resolution structure (for 1024x1024 image, 128x128 latent):
        - down_blocks.0: 128x128 (same as latent)
        - down_blocks.1: 64x64 (latent / 2)
        - down_blocks.2: 32x32 (latent / 4)
        - mid_block: 32x32 (latent / 4)
        - up_blocks.0: 32x32 (latent / 4)
        - up_blocks.1: 64x64 (latent / 2)
        - up_blocks.2: 128x128 (same as latent)
        
        Args:
            layer_name: Full name of the module
            latent_h: Height of latent space (e.g., 128 for 1024x1024 image)
            latent_w: Width of latent space
            
        Returns:
            (height, width) tuple or None if layer shouldn't have mask
        """
        # Skip to_k and to_v in cross-attention (attn2) - they process text tokens (77), not image features
        if 'attn2' in layer_name and ('to_k' in layer_name or 'to_v' in layer_name):
            return None
        
        # Include: attn1 layers, attn2.to_q, attn2.to_out, and FF layers (ff.net)
        # All these process image features with spatial sequence length
        is_attn1 = 'attn1' in layer_name
        is_attn2_img = 'attn2' in layer_name and ('to_q' in layer_name or 'to_out' in layer_name)
        is_ff = 'ff.net' in layer_name or 'ff_norm' in layer_name
        
        if not (is_attn1 or is_attn2_img or is_ff):
            return None
        
        # Parse block and determine resolution
        # down_blocks: resolution decreases (latent -> latent/2 -> latent/4)
        # up_blocks: resolution increases (latent/4 -> latent/2 -> latent)
        
        if 'down_blocks.0' in layer_name:
            # First down block - same as latent resolution
            return (latent_h, latent_w)
        elif 'down_blocks.1' in layer_name:
            # Second down block (latent / 2)
            return (latent_h // 2, latent_w // 2)
        elif 'down_blocks.2' in layer_name:
            # Third down block (latent / 4)
            return (latent_h // 4, latent_w // 4)
        elif 'mid_block' in layer_name:
            # Mid block (latent / 4)
            return (latent_h // 4, latent_w // 4)
        elif 'up_blocks.0' in layer_name:
            # First up block (latent / 4)
            return (latent_h // 4, latent_w // 4)
        elif 'up_blocks.1' in layer_name:
            # Second up block (latent / 2)
            return (latent_h // 2, latent_w // 2)
        elif 'up_blocks.2' in layer_name:
            # Third up block (same as latent)
            return (latent_h, latent_w)
        
        # Default to mid_block resolution for any unrecognized blocks
        return (latent_h // 4, latent_w // 4)
    
    def _convert_to_01_mask(self, freefuse_masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert soft masks to binary 0-1 masks using argmax.
        
        For each spatial position, the LoRA with the highest value gets 1, others get 0.
        
        Args:
            freefuse_masks: Dict of soft masks, each of shape (B, 1, H, W)
            
        Returns:
            Dict of binary masks
        """
        if not freefuse_masks:
            return freefuse_masks
        
        lora_names = list(freefuse_masks.keys())
        
        # Stack all masks: (B, num_loras, H, W)
        stacked = torch.stack([freefuse_masks[name].squeeze(1) for name in lora_names], dim=1)
        
        # Get argmax for each position
        max_indices = stacked.argmax(dim=1)  # (B, H, W)
        
        # Create binary masks
        converted = {}
        for i, lora_name in enumerate(lora_names):
            mask_01 = (max_indices == i).float().unsqueeze(1)  # (B, 1, H, W)
            converted[lora_name] = mask_01
        
        return converted
    
    def enable_freefuse_masks(self):
        """Enable FreeFuse masks for all FreeFuseLinear layers."""
        for module in self.modules():
            if isinstance(module, FreeFuseLinear):
                module.enable_masks()
    
    def disable_freefuse_masks(self):
        """Disable FreeFuse masks for all FreeFuseLinear layers."""
        for module in self.modules():
            if isinstance(module, FreeFuseLinear):
                module.disable_masks()
