import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


SERVER = "http://127.0.0.1:8188"
WORKSPACE = Path(__file__).resolve().parents[1]
COMFY_ROOT = WORKSPACE.parents[1]
OUTPUT_DIR = COMFY_ROOT / "output"
INPUT_DIR = COMFY_ROOT / "input"
WORKFLOW_PATH = WORKSPACE / "freefuse_comfyui" / "workflows" / "qwen_image_2512_freefuse_complete.json"
RESULTS_DIR = WORKSPACE / "analysis" / "qwen_mask_refiner_results"

PHASE1_SEED = 42
PHASE2_SEED = 327228017462953
QWEN_STEPS = 4
QWEN_CFG = 1.0


def http_json(path, data=None, timeout=30):
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(SERVER + path, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload.strip() else {}
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {payload}") from exc


def wait_queue_empty():
    while True:
        queue = http_json("/queue", timeout=10)
        running = queue.get("queue_running") or []
        pending = queue.get("queue_pending") or []
        if not running and not pending:
            return
        time.sleep(1)


def find_comfy_pid():
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*ComfyUI*main.py*' } | "
        "Select-Object -First 1 -ExpandProperty ProcessId"
    )
    out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True)
    return int(out.strip().splitlines()[0])


class MemoryMonitor:
    def __init__(self, pid, label):
        self.pid = pid
        self.label = label
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self):
        last_counter = 0.0
        while not self.stop_event.is_set():
            sample = {"time": time.time()}
            sample.update(self._sample_nvidia())
            now = time.time()
            if now - last_counter >= 5.0:
                sample.update(self._sample_process_counters())
                last_counter = now
            self.samples.append(sample)
            self.stop_event.wait(1.0)

    @staticmethod
    def _sample_nvidia():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return {}
        data = {}
        for line in out.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            idx, used, free, total = parts
            data[f"gpu{idx}_used_mb"] = float(used)
            data[f"gpu{idx}_free_mb"] = float(free)
            data[f"gpu{idx}_total_mb"] = float(total)
        return data

    def _sample_process_counters(self):
        pattern = f"\\GPU Process Memory(pid_{self.pid}_*)\\* Usage"
        cmd = (
            f"$samples=(Get-Counter '{pattern}' -ErrorAction SilentlyContinue).CounterSamples; "
            "$samples | ForEach-Object { \"{0}|{1}\" -f $_.Path,$_.CookedValue }"
        )
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except Exception:
            return {}
        totals = {
            "process_shared_mb": 0.0,
            "process_nonlocal_mb": 0.0,
            "process_dedicated_mb": 0.0,
            "process_local_mb": 0.0,
        }
        for line in out.splitlines():
            if "|" not in line:
                continue
            path, value = line.rsplit("|", 1)
            try:
                mb = float(value.replace(",", ".")) / (1024 * 1024)
            except ValueError:
                continue
            lower = path.lower()
            if "shared usage" in lower:
                totals["process_shared_mb"] += mb
            elif "non local usage" in lower:
                totals["process_nonlocal_mb"] += mb
            elif "dedicated usage" in lower:
                totals["process_dedicated_mb"] += mb
            elif "local usage" in lower:
                totals["process_local_mb"] += mb
        return totals

    def summary(self):
        def max_key(key):
            values = [sample[key] for sample in self.samples if key in sample]
            return max(values) if values else None

        def min_key(key):
            values = [sample[key] for sample in self.samples if key in sample]
            return min(values) if values else None

        return {
            "samples": len(self.samples),
            "gpu0_used_peak_mb": max_key("gpu0_used_mb"),
            "gpu0_free_min_mb": min_key("gpu0_free_mb"),
            "process_shared_peak_mb": max_key("process_shared_mb"),
            "process_nonlocal_peak_mb": max_key("process_nonlocal_mb"),
            "process_dedicated_peak_mb": max_key("process_dedicated_mb"),
            "process_local_peak_mb": max_key("process_local_mb"),
        }


def save_monitor_csv(run_dir, label, monitor):
    path = run_dir / f"{label}_memory_samples.json"
    path.write_text(json.dumps(monitor.samples, indent=2), encoding="utf-8")
    return str(path)


def load_api_workflow():
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def patch_qwen_editor_values(prompt):
    prompt = json.loads(json.dumps(prompt))

    for node_id in list(prompt.keys()):
        if prompt[node_id]["class_type"] == "PreviewImage":
            del prompt[node_id]

    prompt["1"]["inputs"]["unet_name"] = r"qwen\qwen_image_2512_fp8_e4m3fn.safetensors"
    prompt["5"]["inputs"].update(
        {
            "lora_name": r"Qwen\ghibli_style_qwen_v3.safetensors",
            "adapter_name": "arden",
            "strength_model": 1.0,
            "strength_clip": 0.0,
            "model": ["28", 0],
        }
    )
    prompt["6"]["inputs"].update(
        {
            "lora_name": r"Qwen\80sFantasyV2Qwen8bit.safetensors",
            "adapter_name": "mila",
            "strength_model": 1.0,
            "strength_clip": 0.0,
        }
    )
    prompt["7"]["inputs"].update(
        {
            "adapter_name_1": "arden",
            "concept_text_1": (
                "arden_qwen, a tall explorer wearing a cobalt field jacket with silver pins, "
                "with Ghibli inspired animation style"
            ),
            "adapter_name_2": "mila",
            "concept_text_2": (
                "mila_qwen, a cheerful engineer in a saffron raincoat with braided hair, "
                "80s Fantasy Movie Still"
            ),
            "enable_background": True,
            "background_text": "rain-washed neon market street at night",
        }
    )
    prompt["12"]["inputs"].update(
        {
            "seed": PHASE1_SEED,
            "steps": QWEN_STEPS,
            "collect_step": 2,
            "cfg": QWEN_CFG,
            "sampler_name": "euler",
            "scheduler": "simple",
            "collect_block": 30,
            "collect_block_end": 30,
            "temperature": 4000.0,
            "top_k_ratio": 0.3,
        }
    )
    prompt["14"]["inputs"].update(
        {
            "enable_token_masking": True,
            "enable_attention_bias": True,
            "bias_scale": 5.0,
            "positive_bias_scale": 1.0,
            "bidirectional": True,
            "use_positive_bias": True,
            "bias_blocks": "all",
        }
    )
    prompt["16"]["inputs"].update(
        {
            "seed": PHASE2_SEED,
            "steps": QWEN_STEPS,
            "cfg": QWEN_CFG,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        }
    )
    prompt["28"] = {
        "inputs": {
            "lora_name": r"Qwen\Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
            "strength_model": 1.0,
            "model": ["4", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "Qwen Lightning 4-step LoRA"},
    }
    return prompt


def add_save_image(prompt, node_id, images_ref, prefix, title):
    prompt[str(node_id)] = {
        "inputs": {
            "images": images_ref,
            "filename_prefix": prefix,
        },
        "class_type": "SaveImage",
        "_meta": {"title": title},
    }


def make_baseline_prompt(prefix):
    prompt = patch_qwen_editor_values(load_api_workflow())
    prompt["18"]["inputs"]["filename_prefix"] = f"{prefix}/final"
    add_save_image(prompt, 90, ["12", 2], f"{prefix}/phase1_mask_preview", "Save Phase 1 Mask Preview")
    add_save_image(prompt, 91, ["15", 0], f"{prefix}/phase1_detailed_masks", "Save Detailed Phase 1 Masks")
    return prompt


def make_direct_prompt(prefix):
    prompt = patch_qwen_editor_values(load_api_workflow())
    prompt["18"]["inputs"]["filename_prefix"] = f"{prefix}/final"
    prompt["29"] = {
        "inputs": {"samples": ["12", 3], "vae": ["3", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "Decode Phase 1 x0 Preview"},
    }
    prompt["30"] = {
        "inputs": {
            "mask_bank": ["12", 1],
            "image": ["29", 0],
            "freefuse_data": ["8", 0],
            "use_ultralytics": True,
            "ultralytics_model": r"segm\person_yolov8m-seg.pt",
            "use_sam": True,
            "sam_checkpoint": "sam_vit_b_01ec64.pth",
            "detector_confidence": 0.25,
            "assignment_threshold": 0.02,
            "phase_threshold": 0.25,
            "gate_dilation_px": 96,
            "min_refined_coverage": 0.003,
            "max_refined_coverage": 0.75,
            "detector_device": "auto",
            "sam_device": "auto",
            "keep_models_loaded": False,
            "write_debug_stats": True,
            "stats_prefix": f"{prefix}/refiner_stats",
        },
        "class_type": "FreeFuseSAMMaskRefiner",
        "_meta": {"title": "Refine Masks from Phase 1 Preview"},
    }
    prompt["14"]["inputs"]["masks"] = ["30", 0]
    prompt["15"]["inputs"]["masks"] = ["30", 0]
    add_save_image(prompt, 90, ["12", 2], f"{prefix}/phase1_mask_preview", "Save Phase 1 Mask Preview")
    add_save_image(prompt, 91, ["29", 0], f"{prefix}/phase1_preview_decode", "Save Decoded Phase 1 Preview")
    add_save_image(prompt, 92, ["30", 1], f"{prefix}/refined_overlay", "Save Refined Overlay")
    add_save_image(prompt, 93, ["30", 2], f"{prefix}/refined_individual_masks", "Save Refined Individual Masks")
    add_save_image(prompt, 94, ["15", 0], f"{prefix}/refined_detailed_masks", "Save Refined Detailed Masks")
    return prompt


def make_cheap_refined_prompt(prefix, preview_input_name):
    prompt = patch_qwen_editor_values(load_api_workflow())
    prompt["18"]["inputs"]["filename_prefix"] = f"{prefix}/final"
    prompt["29"] = {
        "inputs": {"image": preview_input_name},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Cheap Preview Pass Image"},
    }
    prompt["30"] = {
        "inputs": {
            "mask_bank": ["12", 1],
            "image": ["29", 0],
            "freefuse_data": ["8", 0],
            "use_ultralytics": True,
            "ultralytics_model": r"segm\person_yolov8m-seg.pt",
            "use_sam": True,
            "sam_checkpoint": "sam_vit_b_01ec64.pth",
            "detector_confidence": 0.25,
            "assignment_threshold": 0.02,
            "phase_threshold": 0.25,
            "gate_dilation_px": 96,
            "min_refined_coverage": 0.003,
            "max_refined_coverage": 0.75,
            "detector_device": "auto",
            "sam_device": "auto",
            "keep_models_loaded": False,
            "write_debug_stats": True,
            "stats_prefix": f"{prefix}/refiner_stats",
        },
        "class_type": "FreeFuseSAMMaskRefiner",
        "_meta": {"title": "Refine Masks from Cheap Preview Pass"},
    }
    prompt["14"]["inputs"]["masks"] = ["30", 0]
    prompt["15"]["inputs"]["masks"] = ["30", 0]
    add_save_image(prompt, 90, ["12", 2], f"{prefix}/phase1_mask_preview", "Save Phase 1 Mask Preview")
    add_save_image(prompt, 91, ["29", 0], f"{prefix}/cheap_preview_loaded", "Save Loaded Cheap Preview")
    add_save_image(prompt, 92, ["30", 1], f"{prefix}/refined_overlay", "Save Refined Overlay")
    add_save_image(prompt, 93, ["30", 2], f"{prefix}/refined_individual_masks", "Save Refined Individual Masks")
    add_save_image(prompt, 94, ["15", 0], f"{prefix}/refined_detailed_masks", "Save Refined Detailed Masks")
    return prompt


def submit_and_wait(prompt, label, run_dir):
    wait_queue_empty()
    prompt_id = str(uuid.uuid4())
    prompt_path = run_dir / f"{label}_prompt.json"
    prompt_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")
    monitor = MemoryMonitor(find_comfy_pid(), label)
    monitor.start()
    start = time.perf_counter()
    response = http_json("/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4()), "prompt_id": prompt_id}, timeout=60)
    if response.get("prompt_id") != prompt_id:
        prompt_id = response["prompt_id"]
    while True:
        history = http_json(f"/history/{prompt_id}", timeout=20)
        if prompt_id in history:
            elapsed = time.perf_counter() - start
            monitor.stop()
            return {
                "prompt_id": prompt_id,
                "elapsed_seconds": elapsed,
                "history": history[prompt_id],
                "memory": monitor.summary(),
                "memory_samples_path": save_monitor_csv(run_dir, label, monitor),
                "prompt_path": str(prompt_path),
            }
        time.sleep(1)


def image_path_from_record(record):
    image_type = record.get("type", "output")
    root = OUTPUT_DIR if image_type == "output" else INPUT_DIR if image_type == "input" else COMFY_ROOT / "temp"
    subfolder = record.get("subfolder") or ""
    return root / subfolder / record["filename"]


def collect_output_images(history):
    outputs = {}
    for node_id, output in history.get("outputs", {}).items():
        paths = []
        for image_record in output.get("images", []):
            paths.append(str(image_path_from_record(image_record)))
        if paths:
            outputs[node_id] = paths
    return outputs


def copy_for_load_image(path, run_dir):
    src = Path(path)
    target_dir = INPUT_DIR / "freefuse_experiment"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"{run_dir.name}_{src.name}"
    target = target_dir / target_name
    shutil.copy2(src, target)
    return f"freefuse_experiment/{target_name}"


def find_by_prefix(outputs, prefix_tail):
    matches = []
    for paths in outputs.values():
        for path in paths:
            if prefix_tail in Path(path).as_posix():
                matches.append(path)
    return matches


def summarize_run(result):
    outputs = collect_output_images(result["history"])
    status = result["history"].get("status", {})
    return {
        "prompt_id": result["prompt_id"],
        "elapsed_seconds": result["elapsed_seconds"],
        "status": status,
        "memory": result["memory"],
        "outputs": outputs,
        "memory_samples_path": result["memory_samples_path"],
        "prompt_path": result["prompt_path"],
    }


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    http_json("/free", {"unload_models": False, "free_memory": True}, timeout=20)
    http_json("/history", {"clear": True}, timeout=20)

    direct_prefix = f"FreeFuse/experiments/{timestamp}/direct_phase1"
    baseline_prefix = f"FreeFuse/experiments/{timestamp}/cheap_preview_source"
    cheap_prefix = f"FreeFuse/experiments/{timestamp}/cheap_preview_refined"

    runs = {}

    direct = submit_and_wait(make_direct_prompt(direct_prefix), "direct_phase1", run_dir)
    runs["direct_phase1"] = summarize_run(direct)

    baseline = submit_and_wait(make_baseline_prompt(baseline_prefix), "cheap_preview_source", run_dir)
    runs["cheap_preview_source"] = summarize_run(baseline)
    baseline_outputs = runs["cheap_preview_source"]["outputs"]
    final_candidates = find_by_prefix(baseline_outputs, f"{timestamp}/cheap_preview_source/final")
    if not final_candidates:
        raise RuntimeError("Could not locate cheap preview source final image in outputs")
    preview_input_name = copy_for_load_image(final_candidates[0], run_dir)

    cheap = submit_and_wait(make_cheap_refined_prompt(cheap_prefix, preview_input_name), "cheap_preview_refined", run_dir)
    runs["cheap_preview_refined"] = summarize_run(cheap)

    stats = {
        "timestamp": timestamp,
        "workflow": str(WORKFLOW_PATH),
        "qwen_steps": QWEN_STEPS,
        "phase1_seed": PHASE1_SEED,
        "phase2_seed": PHASE2_SEED,
        "preview_input_name": preview_input_name,
        "runs": runs,
    }
    out_path = run_dir / "experiment_summary.json"
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(out_path), "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
