# Qwen Image FreeFuse Postmortem

Date: 2026-05-29

## Image Degradation

- Symptom: 4-step outputs became soft/painterly and no longer matched the sharp reference images.
- Cause: the saved workflow no longer had the 4-step Lightning model LoRA before the two FreeFuse LoRAs.
- Fix: restored `LoraLoaderModelOnly` with `Qwen\Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors` at strength `1`.
- Validation: generated a 4-step output and visually checked that image quality and LoRA styling were restored.

## VRAM And Shared GPU Memory

- Symptom: dedicated VRAM hit the 32 GB limit, Windows shared GPU memory rose, and repeat generations stayed slow.
- Cause: Qwen attention bias cached the same large `(txt_len, img_len)` bias tensor once per transformer block.
- Impact: one bias tensor was roughly `119 MiB`; duplicated across all Qwen blocks this wasted about `6-8 GiB`.
- Fix: shared one Qwen attention-bias cache across all patched Qwen transformer blocks.
- Validation: hot 4-step run completed in `11.6s`, peaked at `29572 MiB` dedicated VRAM, and kept shared GPU memory flat at `88 MiB` with `Non Local=0`.

## Guardrails

- Do not change the LoRAs or the 4-step Lightning setup.
- Test with a real 4-step generation before calling the work done.
- Inspect the final image and monitor dedicated VRAM plus Windows Shared/Non Local GPU memory.
