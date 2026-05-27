#!/usr/bin/env python3
"""
Direct-run FreeFuse script for FLUX.2 klein 9B.

Edit the configuration section in `main()` and run:
    python main_freefuse_flux2_klein_9b.py
"""

import os
import string
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

from src.models.freefuse_transformer_flux2 import (
    Flux2Transformer2DModel as FreeFuseFlux2Transformer2DModel,
)
from src.pipeline.freefuse_flux2_klein_pipeline import FreeFuseFlux2KleinPipeline


def resolve_default_klein_loras(model_path: str, lora_dir: str) -> Tuple[List[str], List[str]]:
    """Resolve default klein LoRAs from `lora_dir` based on model size (4B/9B)."""
    model_key = model_path.lower()
    if "4b" in model_key:
        model_size_key = "4b"
    elif "9b" in model_key:
        model_size_key = "9b"
    else:
        return [], []

    if not os.path.isdir(lora_dir):
        return [], []

    lora_files: List[str] = []
    for filename in sorted(os.listdir(lora_dir)):
        name_lower = filename.lower()
        if "klein" not in name_lower:
            continue
        if model_size_key not in name_lower:
            continue
        if not name_lower.endswith(".safetensors"):
            continue
        lora_files.append(filename)

    lora_paths = [os.path.join(lora_dir, filename) for filename in lora_files]
    lora_names = [os.path.splitext(filename)[0] for filename in lora_files]
    return lora_paths, lora_names


def find_concept_positions(
    pipe: FreeFuseFlux2KleinPipeline,
    prompt: str,
    concept_map: Dict[str, str],
    filter_meaningless: bool = True,
    filter_single_char: bool = True,
) -> Dict[str, List[List[int]]]:
    """
    Find token positions for each concept text by character-span overlap.

    This is more robust than plain token-id subsequence matching for long
    natural-language concept descriptions.
    """
    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
    }

    def normalize_token_text(token_text: str) -> str:
        return (
            token_text.replace("▁", " ")
            .replace("Ġ", " ")
            .replace("_", " ")
            .strip()
            .lower()
        )

    def is_pure_punctuation(token_text: str) -> bool:
        cleaned = normalize_token_text(token_text)
        if not cleaned:
            return True
        return all(ch in string.punctuation for ch in cleaned)

    def is_meaningless_token(token_text: str) -> bool:
        cleaned = normalize_token_text(token_text)
        if not cleaned:
            return True
        if filter_single_char and len(cleaned) == 1:
            return True
        if cleaned in stopwords:
            return True
        if is_pure_punctuation(token_text):
            return True
        return False

    messages = [{"role": "user", "content": prompt}]
    templated_text = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    prompt_inputs = pipe.tokenizer(
        templated_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer_max_length,
        return_offsets_mapping=True,
    )

    prompt_ids = prompt_inputs.input_ids[0].tolist()
    offset_mapping = prompt_inputs.offset_mapping[0].tolist()

    concept_pos_map: Dict[str, List[List[int]]] = {}
    for concept_name, concept_text in concept_map.items():
        positions: List[int] = []
        positions_with_text: List[Tuple[int, str]] = []

        search_start = 0
        while True:
            char_start = templated_text.find(concept_text, search_start)
            if char_start == -1:
                break
            char_end = char_start + len(concept_text)

            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_end <= char_start or token_start >= char_end:
                    continue
                if token_idx in positions:
                    continue
                positions.append(token_idx)
                token_text = pipe.tokenizer.decode(
                    [prompt_ids[token_idx]],
                    skip_special_tokens=False,
                )
                positions_with_text.append((token_idx, token_text))

            search_start = char_start + 1

        if filter_meaningless and positions_with_text:
            filtered_positions = [
                pos
                for pos, token_text in positions_with_text
                if not is_meaningless_token(token_text)
            ]

            if not filtered_positions:
                non_punct_positions = [
                    pos
                    for pos, token_text in positions_with_text
                    if not is_pure_punctuation(token_text)
                ]
                if non_punct_positions:
                    filtered_positions = [non_punct_positions[0]]
                else:
                    filtered_positions = [positions_with_text[0][0]]

            positions = filtered_positions

        concept_pos_map[concept_name] = [positions]

    return concept_pos_map


def create_comparison_image(
    before_image: Image.Image,
    after_image: Image.Image,
    before_label: str = "Before (No FreeFuse)",
    after_label: str = "After (FreeFuse)",
) -> Image.Image:
    """Create side-by-side comparison image with labels."""
    width, height = before_image.size
    label_height = 40
    composite = Image.new("RGB", (width * 2, height + label_height), color="white")

    composite.paste(before_image, (0, label_height))
    composite.paste(after_image, (width, label_height))

    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
    except Exception:
        font = ImageFont.load_default()

    draw.text((width // 2, 10), before_label, fill="black", font=font, anchor="mt")
    draw.text((width + width // 2, 10), after_label, fill="black", font=font, anchor="mt")
    return composite


def save_masks_for_debug(lora_masks: Dict[str, torch.Tensor], debug_dir: str) -> None:
    """Save per-concept masks as grayscale debug images."""
    os.makedirs(debug_dir, exist_ok=True)
    for lora_name, mask in lora_masks.items():
        mask_np = mask[0, 0].detach().cpu().numpy()
        mask_img = Image.fromarray((mask_np * 255).astype("uint8"))
        mask_path = os.path.join(debug_dir, f"mask_{lora_name}.png")
        mask_img.save(mask_path)
        print(f"Saved mask to {mask_path}")


def main() -> None:
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    model_path = "black-forest-labs/FLUX.2-klein-9B"
    torch_dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lora_paths: List[str] = [
        "loras/harry_klein_9b.safetensors",
        "loras/shalnark_klein_9b.safetensors",
    ]
    lora_names: List[str] = ["harry_klein", "shalnark_klein"]
    lora_weights: List[float] = [1.0, 1.0]

    concept_map: Dict[str, str] = {
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

    t5_prompt = (
        "A picture of two characters, a starry night scene with northern lights: "
        + " and ".join(concept_map[lora_name] for lora_name in lora_names)
    )
    # FLUX.2 klein uses a single Qwen prompt path; use t5_prompt as actual prompt.
    prompt = t5_prompt
    height = 1024
    width = 1024
    num_inference_steps = 4
    guidance_scale = 1.0
    seed = 0

    sim_map_extraction_step = 2
    sim_map_extraction_block: Optional[str] = "single_transformer_blocks.3"  # e.g. "transformer_blocks.4"
    top_k_ratio = 0.1
    exclude_background = True
    use_balanced_argmax = True
    balanced_argmax_debug = False
    # Attention bias settings (aligned with flux.dev style)
    attention_bias_scale = 1.0
    attention_bias_positive = True
    attention_bias_positive_scale = 1.0
    attention_bias_bidirectional = True

    compare = True
    output_path = "freefuse_flux2_klein_9b_output.png"
    save_sim_maps = False
    debug_save_path: Optional[str] = None  # e.g. "debug_output_klein_9b"
    debug_dir = "debug_output_klein_9b"

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    if not lora_paths:
        local_lora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loras")
        default_paths, default_names = resolve_default_klein_loras(model_path, local_lora_dir)
        if default_paths:
            lora_paths = default_paths
            if not lora_names:
                lora_names = default_names
            print(f"Using {len(lora_paths)} default klein LoRAs for {model_path}:")
            for path in lora_paths:
                print(f"  {path}")
        else:
            print(f"No default klein LoRAs found for {model_path} in {local_lora_dir}")

    if lora_paths and not lora_names:
        lora_names = [os.path.splitext(os.path.basename(path))[0] for path in lora_paths]

    if lora_paths and not lora_weights:
        lora_weights = [1.0] * len(lora_paths)

    if len(lora_paths) != len(lora_names):
        raise ValueError("`lora_paths` and `lora_names` must have the same length.")
    if len(lora_paths) != len(lora_weights):
        raise ValueError("`lora_paths` and `lora_weights` must have the same length.")
    missing_concept_loras = [name for name in lora_names if name not in concept_map]
    if missing_concept_loras:
        raise ValueError(
            "`concept_map` must contain all `lora_names`. Missing: "
            + ", ".join(missing_concept_loras)
        )
    missing_loras = [path for path in lora_paths if not os.path.isfile(path)]
    if missing_loras:
        raise FileNotFoundError(
            "Missing LoRA files:\n" + "\n".join(f"  - {path}" for path in missing_loras)
        )

    print(f"Loading FLUX.2 klein pipeline from {model_path} ...")
    pipe = FreeFuseFlux2KleinPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    pipe.transformer.__class__ = FreeFuseFlux2Transformer2DModel
    # pipe.to(device)
    pipe.enable_model_cpu_offload()

    if lora_paths:
        print(f"Loading {len(lora_paths)} LoRA adapters ...")
        for lora_path, lora_name in zip(lora_paths, lora_names):
            print(f"  {lora_name} <- {lora_path}")
            pipe.load_lora_weights(lora_path, adapter_name=lora_name)
        pipe.set_adapters(lora_names, adapter_weights=lora_weights)
        pipe.convert_lora_layers(lora_names)

    print(f"Qwen prompt: {prompt}")
    pipe.setup_freefuse_attention_processors()
    if debug_save_path is not None:
        print(f"Debug visualizations will be saved to {debug_save_path}")

    freefuse_token_pos_maps: Optional[Dict[str, List[List[int]]]] = None
    eos_idx: Optional[int] = None
    if concept_map:
        freefuse_token_pos_maps = find_concept_positions(pipe, prompt, concept_map)
        for lora_name in lora_names:
            positions = freefuse_token_pos_maps.get(lora_name, [[]])[0]
            print(f"{lora_name} token positions: {positions}")
            if not positions:
                print(
                    f"Warning: no token positions found for concept `{lora_name}` "
                    f"in prompt."
                )

        eos_idx = pipe.find_eos_token_index(prompt)
        print(f"EOS token index: {eos_idx}")

        pipe.transformer.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
        pipe.transformer.set_freefuse_background_info(eos_token_index=eos_idx)
        pipe.transformer.set_freefuse_top_k_ratio(top_k_ratio)
        pipe.transformer.enable_concept_sim_map_extraction(sim_map_extraction_block)

    # -------------------------------------------------------------------------
    # Baseline (optional)
    # -------------------------------------------------------------------------
    image_no_freefuse: Optional[Image.Image] = None
    if compare:
        print("\nGenerating baseline without FreeFuse ...")
        generator = torch.Generator(device=device).manual_seed(seed)
        pipe.transformer.clear_freefuse_state()
        pipe.transformer.enable_concept_sim_map_extraction(None)

        output_no_freefuse = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image_no_freefuse = output_no_freefuse.images[0]
        baseline_path = output_path.replace(".png", "_no_freefuse.png")
        image_no_freefuse.save(baseline_path)
        print(f"Baseline image saved to {baseline_path}")

        # Restore FreeFuse settings for phase-1 extraction.
        if freefuse_token_pos_maps:
            pipe.transformer.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
            pipe.transformer.set_freefuse_background_info(eos_token_index=eos_idx)
            pipe.transformer.set_freefuse_top_k_ratio(top_k_ratio)
            pipe.transformer.enable_concept_sim_map_extraction(sim_map_extraction_block)

    # -------------------------------------------------------------------------
    # FreeFuse generation
    # -------------------------------------------------------------------------
    generator = torch.Generator(device=device).manual_seed(seed)
    output = None

    if freefuse_token_pos_maps:
        print("\n[Phase 1] Extracting concept sim maps ...")
        captured_sim_maps: Dict[str, torch.Tensor] = {}

        def extract_sim_maps_callback(pipeline, step_index, timestep, callback_kwargs):
            if step_index == sim_map_extraction_step:
                sim_maps = pipeline.transformer.get_concept_sim_maps()
                if sim_maps:
                    captured_sim_maps.clear()
                    captured_sim_maps.update(sim_maps)
                    print(
                        f"Captured sim maps at step {step_index}: {list(sim_maps.keys())}"
                    )
            return callback_kwargs

        output_phase1 = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=extract_sim_maps_callback,
            attention_kwargs={"top_k_ratio": top_k_ratio},
        )

        if captured_sim_maps and lora_names:
            print("[Phase 1.5] Building masks and attention bias ...")
            lat_height = height // (pipe.vae_scale_factor * 2)
            lat_width = width // (pipe.vae_scale_factor * 2)

            lora_masks = pipe.sim_maps_to_masks(
                captured_sim_maps,
                height=lat_height,
                width=lat_width,
                exclude_background=exclude_background,
                debug_save_path=debug_save_path,
                debug_step_idx=sim_map_extraction_step,
                use_balanced_argmax=use_balanced_argmax,
                balanced_argmax_debug=balanced_argmax_debug,
            )

            if save_sim_maps:
                save_masks_for_debug(lora_masks, debug_save_path or debug_dir)

            txt_len = pipe.tokenizer_max_length
            img_len = lat_height * lat_width
            attention_bias = pipe.build_attention_bias(
                lora_masks,
                freefuse_token_pos_maps,
                txt_len=txt_len,
                img_len=img_len,
                suppress_strength=-attention_bias_scale,
                debug_save_path=debug_save_path,
                debug_step_idx=sim_map_extraction_step,
                positive_bias_scale=attention_bias_positive_scale,
                bidirectional=attention_bias_bidirectional,
                use_positive_bias=attention_bias_positive,
            )

            pipe.transformer.set_freefuse_masks(lora_masks)
            pipe.transformer.set_freefuse_attention_bias(attention_bias)
            pipe.transformer.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
            pipe.transformer.set_freefuse_background_info(eos_token_index=eos_idx)
            pipe.transformer.set_freefuse_top_k_ratio(top_k_ratio)
            pipe.transformer.enable_concept_sim_map_extraction(None)

            print("[Phase 2] Regenerating with FreeFuse masks ...")
            generator = torch.Generator(device=device).manual_seed(seed)
            output = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                attention_kwargs={"top_k_ratio": top_k_ratio},
            )
        else:
            print("No sim maps captured. Falling back to phase-1 output.")
            output = output_phase1
    else:
        print("\nNo concept_map configured. Running standard generation.")
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    image = output.images[0]
    image.save(output_path)
    print(f"Output saved to {output_path}")

    if compare and image_no_freefuse is not None:
        compare_image = create_comparison_image(image_no_freefuse, image)
        compare_path = output_path.replace(".png", "_compare.png")
        compare_image.save(compare_path)
        print(f"Comparison image saved to {compare_path}")

    pipe.transformer.clear_freefuse_state()
    print("Done.")


if __name__ == "__main__":
    main()
