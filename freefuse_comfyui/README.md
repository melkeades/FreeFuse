# ComfyUI-FreeFuse

FreeFuse for ComfyUI: multi-concept LoRA composition with spatial awareness.

## Workflows

- [workflows/flux_freefuse_complete.json](workflows/flux_freefuse_complete.json)
- [workflows/flux2_klein_4b_freefuse_complete.json](workflows/flux2_klein_4b_freefuse_complete.json)
- [workflows/flux2_klein_9b_freefuse_complete.json](workflows/flux2_klein_9b_freefuse_complete.json)
- [workflows/qwen_image_2512_freefuse_complete.json](workflows/qwen_image_2512_freefuse_complete.json)
- [workflows/qwen_image_2512_freefuse_with_editor.json](workflows/qwen_image_2512_freefuse_with_editor.json)
- [workflows/sdxl_freefuse_complete.json](workflows/sdxl_freefuse_complete.json)
- [workflows/zimage_freefuse_complete.json](workflows/zimage_freefuse_complete.json)

## Installation

```bash
git clone <this-repo>
ln -s /path/to/FreeFuse/comfyui ComfyUI/custom_nodes
```

## Example LoRAs and Prompt (from test_parameters.py)

**LoRA download links**

- Daiyu: https://huggingface.co/lsmpp/freefuse_community_loras/resolve/main/daiyu_lin.safetensors?download=true
- Harry: https://huggingface.co/lsmpp/freefuse_community_loras/resolve/main/harry_potter.safetensors?download=true
- Jinx (Z-Image-Turbo): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/Jinx_Arcane_zit.safetensors?download=true
- Skeletor (Z-Image-Turbo): https://huggingface.co/lsmpp/freefuse_example_loras/resolve/main/skeletor_zit.safetensors?download=true

> The workflows expect these filenames by default:
> - Flux: harry_potter_flux.safetensors, daiyu_lin_flux.safetensors
> - Flux2.Klein 4B: flux-2-klein-4b.safetensors + qwen_3_4b.safetensors + flux2-vae.safetensors
> - Flux2.Klein 9B: flux-2-klein-9b-fp8.safetensors + qwen_3_8b_fp8mixed.safetensors + flux2-vae.safetensors
> - Qwen-Image-2512: qwen_image_2512_fp8_e4m3fn.safetensors + qwen_2.5_vl_7b_fp8_scaled.safetensors + qwen_image_vae.safetensors
> - SDXL: harry_potter_xl.safetensors, daiyu_lin_xl.safetensors
> - Z-Image-Turbo: Jinx_Arcane_zit.safetensors, skeletor_zit.safetensors
> If you use the downloads above, rename the files or update the workflow nodes.

**Qwen-Image-2512 model files**

Place the native ComfyUI split files in:

- `models/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors`
- `models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `models/vae/qwen_image_vae.safetensors`
- Optional/recommended for the fast workflow: `models/loras/Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors`

The Qwen workflows use placeholder subject LoRA names (`qwen_subject_a.safetensors`, `qwen_subject_b.safetensors`). Replace them with your Qwen-Image subject LoRAs and keep each `adapter_name` matched to its concept map entry.

**Prompt**

Realistic photography, harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes hugging daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression, autumn leaves blurred in the background, high quality, detailed

**Negative Prompt (SDXL only)**

low quality, blurry, deformed, ugly, bad anatomy

**Concept Map**

- harry: harry potter, an European photorealistic style teenage wizard boy with messy black hair, round wire-frame glasses, and bright green eyes, wearing a white shirt, burgundy and gold striped tie, and dark robes
- daiyu: daiyu_lin, a young East Asian photorealistic style woman in traditional Chinese hanfu dress, elaborate black updo hairstyle adorned with delicate white floral hairpins and ornaments, dangling red tassel earrings, soft pink and red color palette, gentle smile with knowing expression
- background_text: autumn leaves blurred in the background

## Important Prompt Rule

- Every **subject** `concept_text` (adapter trigger phrase) must appear verbatim in the **main prompt**.
- If any subject concept is missing from the main prompt, `FreeFuseTokenPositions` / `FreeFuseConceptMapSimple` now raises an error in ComfyUI.
- `background_text` is optional for runtime safety: if provided but not found in the main prompt, FreeFuse only prints a warning and continues.
- Qwen-Image uses explicit `background_text` only; automatic background-token fallback is intentionally disabled because the native Qwen prompt template is trimmed before conditioning.

Example:
- `concept_text = "harry potter"` means your main prompt must contain `"harry potter"`.

## Hyperparameters

### Phase 1 (FreeFuse Phase1 Sampler)

- `steps`: Total steps for the denoise schedule. Match this to Phase 2.
- `collect_step`: Which step to collect attention and early-stop
- `collect_block`: Transformer block/layer to extract attention (Flux/Qwen-Image: `transformer_blocks.<idx>`, Flux2: `single_transformer_blocks.<idx>`, Z-Image: `layers.<idx>`, SDXL ignored)
- `collect_block_end`: Optional inclusive end index for range-mode collection (Flux/Flux2/Z-Image/Qwen-Image). Set `collect_block_end > collect_block` to enable majority-vote aggregation across blocks.
- `temperature`: Softmax temperature for similarity; 0 = auto (Flux/Flux2/Qwen-Image/Z-Image=4000, SDXL=300)
- `top_k_ratio`: Ratio of top-k tokens used for similarity
- `disable_lora_phase1`: Disable LoRA in Phase 1 (recommended for cleaner attention)
- `bg_scale`: Background similarity scale (higher = more background)
- `use_morphological_cleaning`: Apply morphological cleanup
- `balance_iterations`: Iterations for balanced argmax (higher = more stable, slower)

### Phase 2 (FreeFuse Mask Applicator)

- `enable_token_masking`: Token-level masking (zero out other concept tokens)
- `enable_attention_bias`: Enable attention bias
- `bias_scale`: Negative bias strength (suppresses wrong concepts)
- `positive_bias_scale`: Positive bias strength (enhances correct concepts)
- `bidirectional`: Flux/Flux2 bidirectional bias (text↔image)
- `use_positive_bias`: Enable positive bias
- `bias_blocks`: Which blocks to apply bias (recommended all or double_stream_only)

### Sampling (KSampler / FluxGuidance)

- Flux uses FluxGuidance for CFG; set KSampler CFG to 1.0
- Flux2.Klein uses CLIPTextEncode + CLIPLoader(type=`flux2`); keep KSampler CFG at 1.0 as a safe default
- Qwen-Image-2512 uses CLIPLoader(type=`qwen_image`) and ModelSamplingAuraFlow shift 3.1. Match Phase 1 to Phase 2. For the Lightning 4-step LoRA workflow use steps 4, CFG 1.0, collect_step 2, collect_block 30, temperature 4000, top_k_ratio 0.3. For non-Lightning/base Qwen use the matching base Phase 2 schedule instead.
- SDXL uses KSampler CFG directly (recommended 7.0)

## Preview Image

The workflows include a preview image:
freefuse_flux_square_1024_output.png. It shows up in the Preview when the workflow loads.

## License

Apache 2.0
