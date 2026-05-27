#!/usr/bin/env python3
"""
Common utilities for FLUX.2-klein FreeFuse block comparison.
"""

import os
import re
import string
import traceback
from pathlib import Path
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
    """Find token positions for each concept text by character-span overlap."""
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
                pos for pos, token_text in positions_with_text if not is_meaningless_token(token_text)
            ]

            if not filtered_positions:
                non_punct_positions = [
                    pos for pos, token_text in positions_with_text if not is_pure_punctuation(token_text)
                ]
                if non_punct_positions:
                    filtered_positions = [non_punct_positions[0]]
                else:
                    filtered_positions = [positions_with_text[0][0]]

            positions = filtered_positions

        concept_pos_map[concept_name] = [positions]

    return concept_pos_map


def _discover_klein_blocks(transformer) -> List[str]:
    """Discover sim-map extraction blocks from transformer module names."""
    block_names = set()
    pattern = re.compile(r"(transformer_blocks|single_transformer_blocks)\.(\d+)")

    for name, module in transformer.named_modules():
        if not hasattr(module, "processor"):
            continue
        match = pattern.search(name)
        if match is None:
            continue
        block_names.add(f"{match.group(1)}.{int(match.group(2))}")

    def sort_key(block_name: str):
        m = pattern.search(block_name)
        stream = m.group(1) if m is not None else ""
        idx = int(m.group(2)) if m is not None else 0
        stream_order = 0 if stream == "transformer_blocks" else 1
        return (stream_order, idx)

    return sorted(block_names, key=sort_key)


def _open_first_existing(path_candidates: List[str]) -> Optional[Image.Image]:
    for path in path_candidates:
        if os.path.exists(path):
            return Image.open(path)
    return None


def _build_grids(
    results: List[Tuple[str, str, str]],
    output_base_dir: str,
    concept_names: List[str],
    sim_map_extraction_step: int,
) -> None:
    comparison_data = []
    for block_name, status, output_dir in results:
        if status != "success":
            continue

        result_path = os.path.join(output_dir, "result.png")
        if not os.path.exists(result_path):
            continue

        result_img = Image.open(result_path)
        mask_images = {}
        sim_images = {}
        for concept_name in concept_names:
            mask_img = _open_first_existing(
                [
                    os.path.join(output_dir, f"lora_mask_{concept_name}_step{sim_map_extraction_step}.png"),
                    os.path.join(output_dir, f"mask_{concept_name}.png"),
                ]
            )
            if mask_img is not None:
                mask_images[concept_name] = mask_img

            sim_img = _open_first_existing(
                [
                    os.path.join(output_dir, f"concept_sim_map_{concept_name}_step{sim_map_extraction_step}.png"),
                ]
            )
            if sim_img is not None:
                sim_images[concept_name] = sim_img

        comparison_data.append(
            {
                "block_name": block_name,
                "result_img": result_img,
                "mask_images": mask_images,
                "sim_images": sim_images,
            }
        )

    if not comparison_data:
        print("No successful results to build comparison grids.")
        return

    num_blocks = len(comparison_data)
    thumb_size = 256
    label_width = 240
    padding = 5
    header_height = 50
    row_height = thumb_size + padding

    col_labels = []
    for cn in concept_names:
        col_labels.append(f"SimMap: {cn}")
    for cn in concept_names:
        col_labels.append(f"Mask: {cn}")
    col_labels.append("Result")

    num_img_cols = len(col_labels)
    total_width = label_width + num_img_cols * (thumb_size + padding) + padding
    total_height = header_height + num_blocks * row_height + padding

    grid_img = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(grid_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        header_font = font

    draw.text((padding, 14), "Block", fill="black", font=header_font)
    x_off = label_width + padding
    for col_label in col_labels:
        draw.text((x_off + 10, 14), col_label, fill="black", font=header_font)
        x_off += thumb_size + padding
    draw.line([(0, header_height - 3), (total_width, header_height - 3)], fill="gray", width=2)

    for row_idx, data in enumerate(comparison_data):
        y = header_height + row_idx * row_height
        draw.text((padding, y + thumb_size // 2 - 8), data["block_name"], fill="black", font=font)
        draw.line([(label_width, y), (label_width, y + thumb_size)], fill="lightgray", width=1)

        x = label_width + padding

        for cn in concept_names:
            if cn in data["sim_images"]:
                img = data["sim_images"][cn].resize((thumb_size, thumb_size), Image.LANCZOS).convert("RGB")
                grid_img.paste(img, (x, y))
            else:
                draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], outline="lightgray", fill="#f0f0f0")
                draw.text((x + 30, y + thumb_size // 2), "N/A", fill="gray", font=font)
            x += thumb_size + padding

        for cn in concept_names:
            if cn in data["mask_images"]:
                img = data["mask_images"][cn].resize((thumb_size, thumb_size), Image.LANCZOS).convert("RGB")
                grid_img.paste(img, (x, y))
            else:
                draw.rectangle([(x, y), (x + thumb_size, y + thumb_size)], outline="lightgray", fill="#f0f0f0")
                draw.text((x + 30, y + thumb_size // 2), "N/A", fill="gray", font=font)
            x += thumb_size + padding

        res = data["result_img"].resize((thumb_size, thumb_size), Image.LANCZOS).convert("RGB")
        grid_img.paste(res, (x, y))

        if row_idx < num_blocks - 1:
            line_y = y + thumb_size + padding // 2
            draw.line([(0, line_y), (total_width, line_y)], fill="#e0e0e0", width=1)

    comprehensive_path = os.path.join(output_base_dir, "comprehensive_comparison.png")
    grid_img.save(comprehensive_path, quality=95)
    print(f"Saved comprehensive comparison grid -> {comprehensive_path}")

    grid_cols = min(6, num_blocks)
    grid_rows = (num_blocks + grid_cols - 1) // grid_cols
    label_h = 30
    gw = grid_cols * thumb_size + (grid_cols + 1) * padding
    gh = grid_rows * (thumb_size + label_h) + (grid_rows + 1) * padding
    simple_grid = Image.new("RGB", (gw, gh), "white")
    sdraw = ImageDraw.Draw(simple_grid)

    for idx, data in enumerate(comparison_data):
        row = idx // grid_cols
        col = idx % grid_cols
        x = padding + col * (thumb_size + padding)
        y = padding + row * (thumb_size + label_h + padding)
        img = data["result_img"].resize((thumb_size, thumb_size), Image.LANCZOS)
        simple_grid.paste(img, (x, y))
        sdraw.text((x + 5, y + thumb_size + 5), data["block_name"], fill="black", font=font)

    simple_path = os.path.join(output_base_dir, "comparison_grid.png")
    simple_grid.save(simple_path, quality=95)
    print(f"Saved result-only comparison grid -> {simple_path}")


def run_flux2_klein_block_comparison(
    model_path: str,
    lora_paths: List[str],
    lora_names: List[str],
    lora_weights: List[float],
    concept_map: Dict[str, str],
    prompt: str,
    output_base_dir: str,
    sim_map_extraction_step: int,
    block_names: Optional[List[str]] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    top_k_ratio: float = 0.1,
    exclude_background: bool = True,
    attention_bias_scale: float = 4.0,
    attention_bias_positive: bool = True,
    attention_bias_positive_scale: float = 2.0,
    attention_bias_bidirectional: bool = True,
    seed: int = 42,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
) -> None:
    os.makedirs(output_base_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not lora_paths:
        local_lora_dir = os.path.join(Path(__file__).resolve().parent.parent, "loras")
        default_paths, default_names = resolve_default_klein_loras(model_path, local_lora_dir)
        if default_paths:
            lora_paths = default_paths
            if not lora_names:
                lora_names = default_names
            print(f"Using {len(lora_paths)} default klein LoRAs from {local_lora_dir}")
        else:
            raise FileNotFoundError(f"No default klein LoRAs found for {model_path} in {local_lora_dir}")

    if lora_paths and not lora_names:
        lora_names = [os.path.splitext(os.path.basename(path))[0] for path in lora_paths]
    if lora_paths and not lora_weights:
        lora_weights = [1.0] * len(lora_paths)

    missing_loras = [path for path in lora_paths if not os.path.isfile(path)]
    if missing_loras:
        raise FileNotFoundError("Missing LoRA files:\n" + "\n".join(f"  - {path}" for path in missing_loras))

    print(f"Loading pipeline from {model_path} ...")
    pipe = FreeFuseFlux2KleinPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    pipe.transformer.__class__ = FreeFuseFlux2Transformer2DModel
    # pipe.to(device)
    pipe.enable_model_cpu_offload()
    pipe.setup_freefuse_attention_processors()

    print(f"Loading {len(lora_paths)} LoRA adapters ...")
    for lora_path, lora_name in zip(lora_paths, lora_names):
        pipe.load_lora_weights(lora_path, adapter_name=lora_name)
        print(f"  {lora_name} <- {lora_path}")
    pipe.set_adapters(lora_names, adapter_weights=lora_weights)
    pipe.convert_lora_layers(lora_names)

    freefuse_token_pos_maps = find_concept_positions(pipe, prompt, concept_map)
    eos_idx = pipe.find_eos_token_index(prompt)
    print(f"EOS token index: {eos_idx}")

    discovered = _discover_klein_blocks(pipe.transformer)
    if block_names is None:
        block_names = discovered
    print(f"Discovered {len(discovered)} sim-map blocks; testing {len(block_names)} blocks.")

    results: List[Tuple[str, str, str]] = []

    for i, block_name in enumerate(block_names):
        output_dir = os.path.join(output_base_dir, block_name.replace(".", "_"))
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[{i + 1}/{len(block_names)}] sim_map_extraction_block = {block_name}")

        pipe.transformer.clear_freefuse_state()
        pipe.transformer.set_freefuse_token_pos_maps(freefuse_token_pos_maps)
        pipe.transformer.set_freefuse_background_info(eos_token_index=eos_idx)
        pipe.transformer.set_freefuse_top_k_ratio(top_k_ratio)
        pipe.transformer.enable_concept_sim_map_extraction(block_name)

        captured_sim_maps: Dict[str, torch.Tensor] = {}

        def extract_sim_maps_callback(pipeline, step_index, timestep, callback_kwargs):
            if step_index == sim_map_extraction_step:
                sim_maps = pipeline.transformer.get_concept_sim_maps()
                if sim_maps:
                    captured_sim_maps.clear()
                    captured_sim_maps.update(sim_maps)
                    print(f"  captured sim maps at step {step_index}: {list(sim_maps.keys())}")
            return callback_kwargs

        generator = torch.Generator(device=device).manual_seed(seed)

        try:
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

            if not captured_sim_maps:
                sim_maps = pipe.transformer.get_concept_sim_maps()
                if sim_maps:
                    captured_sim_maps.update(sim_maps)

            output = output_phase1
            if captured_sim_maps:
                lat_height = height // (pipe.vae_scale_factor * 2)
                lat_width = width // (pipe.vae_scale_factor * 2)
                lora_masks = pipe.sim_maps_to_masks(
                    captured_sim_maps,
                    height=lat_height,
                    width=lat_width,
                    exclude_background=exclude_background,
                    debug_save_path=output_dir,
                    debug_step_idx=sim_map_extraction_step,
                )

                attention_bias = pipe.build_attention_bias(
                    lora_masks,
                    freefuse_token_pos_maps,
                    txt_len=pipe.tokenizer_max_length,
                    img_len=lat_height * lat_width,
                    suppress_strength=-attention_bias_scale,
                    debug_save_path=output_dir,
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
                print("  no sim maps captured; using phase-1 output.")

            image = output.images[0]
            result_path = os.path.join(output_dir, "result.png")
            image.save(result_path)
            print(f"  ✓ saved -> {result_path}")
            results.append((block_name, "success", output_dir))

        except Exception as exc:
            traceback.print_exc()
            print(f"  ✗ error: {exc}")
            results.append((block_name, f"error: {exc}", output_dir))

    _build_grids(
        results=results,
        output_base_dir=output_base_dir,
        concept_names=list(concept_map.keys()),
        sim_map_extraction_step=sim_map_extraction_step,
    )

    print("\nSummary:")
    success_count = sum(1 for _, status, _ in results if status == "success")
    for block_name, status, _ in results:
        icon = "✓" if status == "success" else "✗"
        print(f"  {icon} {block_name}: {status}")
    print(f"{success_count}/{len(results)} blocks succeeded.")
    print(f"All outputs saved to: {output_base_dir}")
