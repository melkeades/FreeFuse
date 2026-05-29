Run ComfyUI bat "H:\cg-git\comfy-port\run_nvidia_gpu.bat" in Windows Terminal app for visibility.
Do not change LoRAs.
Do not revert to qwen 50 steps workflow, keep it 4 steps.
If needed Use Comfyui python H:\cg-git\comfy-port\python_embeded (not global).
Monitor the shared gpu memory, if vram is full and shared memory starts rising (above 1gb) try to fix but after the quality of the image always prioritize speed (tokens per second) not just vram stats.
Always check the final image quality, masks and adherence to the prompt.
Use the \freefuse_comfyui\workflows\qwen_image_2512_freefuse_with_editor.json as the primary test workflow.
Unload the models and clear cache when restarting the comfyui process.
