#!/usr/bin/env python3
"""
FreeFuse FLUX.2-klein-9B block comparison.
"""

import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.flux2_klein_block_comparison_common import run_flux2_klein_block_comparison


def main() -> None:
    model_path = "black-forest-labs/FLUX.2-klein-9B"

    lora_paths = [
        "loras/harry_klein_9b.safetensors",
        "loras/shalnark_klein_9b.safetensors",
    ]
    lora_names = ["harry_klein", "shalnark_klein"]
    lora_weights = [1.0, 1.0]

    concept_map = {
        "harry_klein": (
            "harry potter, an European photorealistic style teenage wizard boy with "
            "messy black hair, round wire-frame glasses, and bright green eyes, "
            "wearing a white shirt, burgundy and gold striped tie, and dark robes"
        ),
        "shalnark_klein": (
            "shalnark, an anime boy with blonde bob haircut and turquoise eyes, "
            "wearing purple and teal futuristic uniform, determined expression, "
            "digital anime art style"
        ),
    }

    prompt = (
        "A picture of two characters, a starry night scene with northern lights: "
        + " and ".join(concept_map[lora_name] for lora_name in lora_names)
    )

    run_flux2_klein_block_comparison(
        model_path=model_path,
        lora_paths=lora_paths,
        lora_names=lora_names,
        lora_weights=lora_weights,
        concept_map=concept_map,
        prompt=prompt,
        output_base_dir="outputs/flux2_klein_9b_block_comparison",
        sim_map_extraction_step=4,
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        top_k_ratio=0.1,
        exclude_background=True,
        attention_bias_scale=4.0,
        attention_bias_positive=True,
        attention_bias_positive_scale=2.0,
        attention_bias_bidirectional=True,
        seed=77,
        torch_dtype=torch.bfloat16,
    )


if __name__ == "__main__":
    main()
