"""
FreeFuse mask refinement from a decoded preview image.

This node uses a detector (Ultralytics) to find candidate subjects, assigns
those candidates back to Phase 1 masks by overlap, then optionally uses SAM to
cut tighter silhouettes. The output masks are resized back to the original
FreeFuse mask resolution so Phase 2 keeps the same attention-bias geometry.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import folder_paths
except Exception:  # pragma: no cover - ComfyUI provides this at runtime
    folder_paths = None


def _comfy_root() -> Path:
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        return Path(folder_paths.models_dir).parent
    return Path(__file__).resolve().parents[4]


def _models_dir() -> Path:
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        return Path(folder_paths.models_dir)
    return _comfy_root() / "models"


def _find_model_files(*parts: str, suffixes=(".pt", ".pth")) -> List[str]:
    base = _models_dir().joinpath(*parts)
    if not base.exists():
        return []
    relative_root = _models_dir() / parts[0] if parts else _models_dir()
    found = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            found.append(str(path.relative_to(relative_root)))
    return found


def _default_ultralytics_models() -> List[str]:
    models = _find_model_files("ultralytics", "segm", suffixes=(".pt",))
    models += _find_model_files("ultralytics", "bbox", suffixes=(".pt",))
    preferred = "segm/person_yolov8m-seg.pt"
    if preferred in models:
        models.remove(preferred)
        models.insert(0, preferred)
    if not models:
        models = [preferred]
    return models


def _default_sam_models() -> List[str]:
    models = _find_model_files("sams", suffixes=(".pt", ".pth"))
    preferred = "sam_vit_b_01ec64.pth"
    if preferred in models:
        models.remove(preferred)
        models.insert(0, preferred)
    if not models:
        models = [preferred]
    return models


class FreeFuseSAMMaskRefiner:
    """Refine FreeFuse concept masks from a preview image using Ultralytics + SAM."""

    _YOLO_CACHE = {}
    _SAM_CACHE = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_bank": ("FREEFUSE_MASKS",),
                "image": ("IMAGE",),
            },
            "optional": {
                "freefuse_data": ("FREEFUSE_DATA",),
                "use_ultralytics": ("BOOLEAN", {"default": True}),
                "ultralytics_model": (_default_ultralytics_models(),),
                "use_sam": ("BOOLEAN", {"default": True}),
                "sam_checkpoint": (_default_sam_models(),),
                "detector_confidence": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.01, "max": 0.95, "step": 0.01},
                ),
                "assignment_threshold": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "phase_threshold": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "gate_dilation_px": (
                    "INT",
                    {"default": 96, "min": 0, "max": 512, "step": 1},
                ),
                "min_refined_coverage": (
                    "FLOAT",
                    {"default": 0.003, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "max_refined_coverage": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "detector_device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "sam_device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "keep_models_loaded": ("BOOLEAN", {"default": False}),
                "write_debug_stats": ("BOOLEAN", {"default": True}),
                "stats_prefix": (
                    "STRING",
                    {"default": "FreeFuse/refiner_stats", "multiline": False},
                ),
            },
        }

    RETURN_TYPES = ("FREEFUSE_MASKS", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("masks", "combined_preview", "individual_masks", "stats")
    FUNCTION = "refine_masks"
    CATEGORY = "FreeFuse"

    def refine_masks(
        self,
        mask_bank,
        image,
        freefuse_data=None,
        use_ultralytics=True,
        ultralytics_model="segm/person_yolov8m-seg.pt",
        use_sam=True,
        sam_checkpoint="sam_vit_b_01ec64.pth",
        detector_confidence=0.25,
        assignment_threshold=0.02,
        phase_threshold=0.25,
        gate_dilation_px=96,
        min_refined_coverage=0.003,
        max_refined_coverage=0.75,
        detector_device="auto",
        sam_device="auto",
        keep_models_loaded=False,
        write_debug_stats=True,
        stats_prefix="FreeFuse/refiner_stats",
    ):
        total_start = time.perf_counter()
        image_np = self._image_tensor_to_np(image)
        img_h, img_w = image_np.shape[:2]
        masks_in = mask_bank.get("masks", {}) if isinstance(mask_bank, dict) else {}
        if not masks_in:
            empty = torch.zeros(1, img_h, img_w, 3)
            stats = {"error": "No input masks", "total_seconds": 0.0}
            return (mask_bank, empty, empty, json.dumps(stats, indent=2))

        concept_names = self._concept_names(masks_in, freefuse_data)
        background_names = [name for name in masks_in.keys() if self._is_background_name(name)]
        phase_full = {}
        original_shapes = {}
        for name in concept_names:
            mask = self._as_2d_float(masks_in[name])
            original_shapes[name] = tuple(mask.shape[-2:])
            phase_full[name] = self._resize_mask_np(mask, img_h, img_w, threshold=None)

        timings = {}
        candidates = []
        if use_ultralytics:
            t0 = time.perf_counter()
            candidates = self._detect_ultralytics(
                image_np,
                ultralytics_model,
                detector_confidence,
                detector_device,
                keep_models_loaded,
            )
            timings["ultralytics_seconds"] = time.perf_counter() - t0
        else:
            timings["ultralytics_seconds"] = 0.0

        assignments = self._assign_candidates(phase_full, candidates, assignment_threshold)

        predictor = None
        sam_model = None
        selected_sam_device = None
        if use_sam:
            t0 = time.perf_counter()
            predictor, sam_model, selected_sam_device = self._load_sam_predictor(
                image_np,
                sam_checkpoint,
                sam_device,
                keep_models_loaded,
            )
            timings["sam_load_seconds"] = time.perf_counter() - t0
        else:
            timings["sam_load_seconds"] = 0.0

        refined_full = {}
        refined_down = {}
        per_concept = {}
        sam_predict_seconds = 0.0
        for name in concept_names:
            phase = phase_full[name]
            phase_bin = phase >= phase_threshold
            assignment = assignments.get(name)
            fallback_reason = None
            box = None
            if assignment is not None:
                box = assignment["box"]
                base_mask = assignment["mask"] > 0.5
            else:
                box = self._mask_bbox(phase_bin, padding=12, width=img_w, height=img_h)
                base_mask = phase_bin

            if box is None:
                fallback_reason = "empty_phase_mask"
                refined = phase_bin
                box = [0, 0, img_w - 1, img_h - 1]
            elif predictor is not None:
                t0 = time.perf_counter()
                refined = self._sam_refine(
                    predictor,
                    image_np,
                    phase,
                    base_mask,
                    box,
                    phase_threshold,
                )
                sam_predict_seconds += time.perf_counter() - t0
                if refined is None:
                    fallback_reason = "sam_no_valid_mask"
                    refined = base_mask
            else:
                refined = base_mask

            refined = self._gate_mask(refined, phase, phase_threshold, gate_dilation_px)
            coverage = float(refined.mean()) if refined.size else 0.0
            if coverage < min_refined_coverage or coverage > max_refined_coverage:
                fallback_reason = fallback_reason or f"coverage_out_of_range:{coverage:.4f}"
                refined = phase_bin
                coverage = float(refined.mean()) if refined.size else 0.0

            refined_full[name] = refined.astype(np.float32)
            target_h, target_w = original_shapes[name]
            down = self._downsample_refined_mask(refined, target_h, target_w, masks_in[name])
            refined_down[name] = down

            phase_binary = phase_bin.astype(bool)
            iou = self._binary_iou(phase_binary, refined.astype(bool))
            per_concept[name] = {
                "phase_coverage": float(phase_binary.mean()),
                "refined_coverage": coverage,
                "phase_refined_iou": iou,
                "boundary_edge_strength": self._boundary_edge_strength(image_np, refined),
                "components": self._component_count(refined),
                "assignment_score": None if assignment is None else float(assignment["score"]),
                "detector_confidence": None if assignment is None else float(assignment["confidence"]),
                "box": [int(v) for v in box],
                "fallback_reason": fallback_reason,
            }

        timings["sam_predict_seconds"] = sam_predict_seconds
        if predictor is not None and not keep_models_loaded:
            del predictor
            del sam_model
            if selected_sam_device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_masks = dict(masks_in)
        output_masks.update(refined_down)
        if background_names:
            max_foreground = None
            for mask in refined_down.values():
                mask_float = self._as_2d_float(mask).to(dtype=torch.float32)
                max_foreground = mask_float if max_foreground is None else torch.maximum(max_foreground, mask_float)
            for bg_name in background_names:
                if max_foreground is not None:
                    bg = (1.0 - max_foreground).clamp(0.0, 1.0)
                    output_masks[bg_name] = bg.to(device=masks_in[bg_name].device, dtype=masks_in[bg_name].dtype)

        stats = {
            "image_size": [int(img_w), int(img_h)],
            "concepts": per_concept,
            "detector_candidates": len(candidates),
            "ultralytics_model": str(self._resolve_ultralytics_path(ultralytics_model)),
            "sam_checkpoint": str(self._resolve_sam_path(sam_checkpoint)),
            "detector_device": self._select_device(detector_device),
            "sam_device": selected_sam_device,
            "timings": timings,
            "total_seconds": time.perf_counter() - total_start,
        }
        stats_path = self._write_stats(stats, stats_prefix) if write_debug_stats else None
        if stats_path:
            stats["stats_path"] = stats_path

        combined_preview = self._combined_preview(image_np, refined_full)
        individual_masks = self._individual_mask_previews(refined_full, img_h, img_w)
        out_bank = dict(mask_bank) if isinstance(mask_bank, dict) else {}
        metadata = dict(out_bank.get("metadata", {}))
        metadata["refiner"] = stats
        out_bank["masks"] = output_masks
        out_bank["metadata"] = metadata

        print(
            "[FreeFuse] SAM mask refiner: "
            f"{len(concept_names)} concepts, {len(candidates)} detector candidates, "
            f"total={stats['total_seconds']:.2f}s"
        )
        return (out_bank, combined_preview, individual_masks, json.dumps(stats, indent=2))

    @staticmethod
    def _image_tensor_to_np(image: torch.Tensor) -> np.ndarray:
        img = image.detach().cpu()
        if img.dim() == 4:
            img = img[0]
        img = img.clamp(0.0, 1.0).numpy()
        return (img * 255.0 + 0.5).astype(np.uint8)

    @staticmethod
    def _as_2d_float(mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 3:
            mask = mask[0]
        return mask.detach().float()

    @staticmethod
    def _is_background_name(name: str) -> bool:
        lower = name.lower()
        return name.startswith("_") and ("background" in lower or lower in {"__bg__", "_bg_"})

    def _concept_names(self, masks: Dict[str, torch.Tensor], freefuse_data) -> List[str]:
        names = []
        if isinstance(freefuse_data, dict):
            for name in freefuse_data.get("concepts", {}).keys():
                if name in masks and not self._is_background_name(name):
                    names.append(name)
        for name in masks.keys():
            if name not in names and not self._is_background_name(name) and not name.startswith("__"):
                names.append(name)
        return names

    @staticmethod
    def _resize_mask_np(mask: torch.Tensor, height: int, width: int, threshold: Optional[float]) -> np.ndarray:
        resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].clamp(0.0, 1.0).cpu().numpy()
        if threshold is not None:
            return (resized >= threshold).astype(np.float32)
        return resized.astype(np.float32)

    @staticmethod
    def _downsample_refined_mask(refined: np.ndarray, height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        tensor = torch.from_numpy(refined.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        down = F.interpolate(tensor, size=(height, width), mode="area")[0, 0].clamp(0.0, 1.0)
        return down.to(device=reference.device, dtype=reference.dtype)

    def _detect_ultralytics(
        self,
        image_np: np.ndarray,
        model_name: str,
        confidence: float,
        device_request: str,
        keep_models_loaded: bool,
    ) -> List[Dict]:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            print(f"[FreeFuse] Ultralytics unavailable: {exc}")
            return []

        model_path = self._resolve_ultralytics_path(model_name)
        if not model_path.exists():
            print(f"[FreeFuse] Ultralytics model not found: {model_path}")
            return []

        cache_key = str(model_path)
        model = self._YOLO_CACHE.get(cache_key)
        if model is None:
            model = YOLO(str(model_path))
            if keep_models_loaded:
                self._YOLO_CACHE[cache_key] = model

        selected_device = self._select_device(device_request)
        try:
            results = model.predict(
                image_np,
                conf=float(confidence),
                device=0 if selected_device == "cuda" else "cpu",
                verbose=False,
            )
        except Exception as exc:
            print(f"[FreeFuse] Ultralytics prediction failed on {selected_device}: {exc}")
            if selected_device == "cuda":
                results = model.predict(image_np, conf=float(confidence), device="cpu", verbose=False)
            else:
                return []

        height, width = image_np.shape[:2]
        candidates = []
        if not results:
            return candidates
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return candidates

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        masks = getattr(result, "masks", None)
        mask_data = None
        if masks is not None and masks.data is not None:
            mask_data = masks.data.detach().cpu().float()
            mask_data = F.interpolate(
                mask_data.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[:, 0].numpy()

        for idx, box in enumerate(xyxy):
            if mask_data is not None and idx < mask_data.shape[0]:
                mask = mask_data[idx] >= 0.5
            else:
                mask = np.zeros((height, width), dtype=bool)
                x1, y1, x2, y2 = self._clamp_box(box, width, height)
                mask[y1:y2 + 1, x1:x2 + 1] = True
            candidates.append(
                {
                    "box": [int(v) for v in self._clamp_box(box, width, height)],
                    "mask": mask.astype(np.float32),
                    "confidence": float(confs[idx]),
                }
            )

        if not keep_models_loaded and cache_key not in self._YOLO_CACHE:
            del model
            if selected_device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return candidates

    def _assign_candidates(self, phase_full, candidates, threshold):
        scored = []
        for name, phase in phase_full.items():
            phase_bin = phase > 0.25
            phase_area = float(phase_bin.sum())
            for idx, cand in enumerate(candidates):
                cand_bin = cand["mask"] > 0.5
                cand_area = float(cand_bin.sum())
                if phase_area <= 0 or cand_area <= 0:
                    continue
                inter = float(np.logical_and(phase_bin, cand_bin).sum())
                union = float(np.logical_or(phase_bin, cand_bin).sum())
                iou = inter / union if union > 0 else 0.0
                candidate_inside = inter / cand_area
                phase_covered = inter / phase_area
                score = 0.5 * iou + 0.35 * candidate_inside + 0.15 * phase_covered
                if score >= threshold:
                    scored.append((score, name, idx, cand))
        scored.sort(key=lambda item: item[0], reverse=True)
        assignments = {}
        used = set()
        for score, name, idx, cand in scored:
            if name in assignments or idx in used:
                continue
            assigned = dict(cand)
            assigned["score"] = float(score)
            assignments[name] = assigned
            used.add(idx)
        return assignments

    def _load_sam_predictor(self, image_np, checkpoint, device_request, keep_models_loaded):
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            print(f"[FreeFuse] segment_anything unavailable: {exc}")
            return None, None, None

        path = self._resolve_sam_path(checkpoint)
        if not path.exists():
            print(f"[FreeFuse] SAM checkpoint not found: {path}")
            return None, None, None

        model_type = self._sam_model_type(path)
        selected_device = self._select_device(device_request)
        cache_key = (str(path), selected_device)
        sam = self._SAM_CACHE.get(cache_key)
        if sam is None:
            sam = sam_model_registry[model_type](checkpoint=str(path))
            sam.to(device=selected_device)
            sam.eval()
            if keep_models_loaded:
                self._SAM_CACHE[cache_key] = sam
        predictor = SamPredictor(sam)
        with torch.inference_mode():
            predictor.set_image(image_np)
        return predictor, sam, selected_device

    def _sam_refine(self, predictor, image_np, phase, base_mask, box, phase_threshold):
        phase_bin = phase >= phase_threshold
        pos = self._positive_point(phase, box)
        point_coords = np.array([pos], dtype=np.float32) if pos is not None else None
        point_labels = np.array([1], dtype=np.int32) if pos is not None else None
        box_np = np.array(box, dtype=np.float32)
        try:
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_np,
                    multimask_output=True,
                )
        except Exception as exc:
            print(f"[FreeFuse] SAM prediction failed: {exc}")
            return None

        best_score = -1.0
        best_mask = None
        for mask, sam_score in zip(masks, scores):
            mask_bool = mask.astype(bool)
            iou_phase = self._binary_iou(mask_bool, phase_bin)
            iou_base = self._binary_iou(mask_bool, base_mask.astype(bool))
            score = 0.55 * float(sam_score) + 0.30 * iou_phase + 0.15 * iou_base
            if score > best_score:
                best_score = score
                best_mask = mask_bool
        return best_mask

    @staticmethod
    def _positive_point(phase, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = phase[max(0, y1):y2 + 1, max(0, x1):x2 + 1]
        if crop.size == 0:
            return None
        y, x = np.unravel_index(int(np.argmax(crop)), crop.shape)
        return [max(0, x1) + x, max(0, y1) + y]

    def _gate_mask(self, refined, phase, phase_threshold, dilation_px):
        if refined is None:
            return phase >= phase_threshold
        if dilation_px <= 0:
            return refined.astype(bool)
        gate = phase >= max(phase_threshold * 0.5, 0.05)
        gate = self._dilate(gate, int(dilation_px))
        gated = np.logical_and(refined.astype(bool), gate)
        if gated.sum() == 0:
            return refined.astype(bool)
        return gated

    @staticmethod
    def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
        if pixels <= 0:
            return mask
        try:
            import cv2

            kernel_size = max(3, pixels | 1)
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        except Exception:
            tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            kernel_size = max(3, pixels | 1)
            padded = F.max_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            return padded[0, 0].numpy() > 0.5

    @staticmethod
    def _mask_bbox(mask: np.ndarray, padding: int, width: int, height: int):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [
            max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(width - 1, int(xs.max()) + padding),
            min(height - 1, int(ys.max()) + padding),
        ]

    @staticmethod
    def _clamp_box(box, width, height):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        return [
            max(0, min(width - 1, x1)),
            max(0, min(height - 1, y1)),
            max(0, min(width - 1, x2)),
            max(0, min(height - 1, y2)),
        ]

    @staticmethod
    def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _component_count(mask: np.ndarray) -> int:
        try:
            import cv2

            return max(0, int(cv2.connectedComponents(mask.astype(np.uint8))[0]) - 1)
        except Exception:
            return 0

    @staticmethod
    def _boundary_edge_strength(image_np: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        gray = image_np.astype(np.float32).mean(axis=2) / 255.0
        gy, gx = np.gradient(gray)
        grad = np.sqrt(gx * gx + gy * gy)
        try:
            import cv2

            eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
            boundary = np.logical_xor(mask.astype(bool), eroded.astype(bool))
        except Exception:
            boundary = mask.astype(bool)
        if boundary.sum() == 0:
            return 0.0
        mean_grad = float(grad.mean()) + 1e-8
        return float(grad[boundary].mean() / mean_grad)

    @staticmethod
    def _combined_preview(image_np: np.ndarray, masks: Dict[str, np.ndarray]) -> torch.Tensor:
        base = image_np.astype(np.float32) / 255.0
        overlay = base.copy()
        colors = np.array(
            [
                [1.0, 0.15, 0.10],
                [0.10, 0.70, 1.0],
                [0.25, 1.0, 0.30],
                [1.0, 0.85, 0.10],
            ],
            dtype=np.float32,
        )
        for idx, mask in enumerate(masks.values()):
            color = colors[idx % len(colors)]
            alpha = (mask > 0.5).astype(np.float32)[..., None] * 0.45
            overlay = overlay * (1.0 - alpha) + color * alpha
        return torch.from_numpy(overlay.clip(0.0, 1.0)).unsqueeze(0)

    @staticmethod
    def _individual_mask_previews(masks: Dict[str, np.ndarray], height: int, width: int) -> torch.Tensor:
        if not masks:
            return torch.zeros(1, height, width, 3)
        images = []
        colors = [
            np.array([1.0, 0.2, 0.1], dtype=np.float32),
            np.array([0.1, 0.6, 1.0], dtype=np.float32),
            np.array([0.2, 1.0, 0.3], dtype=np.float32),
            np.array([1.0, 0.8, 0.1], dtype=np.float32),
        ]
        for idx, mask in enumerate(masks.values()):
            m = (mask > 0.5).astype(np.float32)[..., None]
            images.append(m * colors[idx % len(colors)])
        return torch.from_numpy(np.stack(images, axis=0).clip(0.0, 1.0))

    @staticmethod
    def _select_device(requested: str) -> str:
        if requested == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            if requested == "cuda":
                return "cuda"
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes > 2 * 1024**3:
                    return "cuda"
            except Exception:
                return "cuda"
        return "cpu"

    @staticmethod
    def _sam_model_type(path: Path) -> str:
        lower = path.name.lower()
        if "vit_h" in lower:
            return "vit_h"
        if "vit_l" in lower:
            return "vit_l"
        return "vit_b"

    @staticmethod
    def _resolve_ultralytics_path(model_name: str) -> Path:
        path = Path(model_name)
        if path.is_absolute():
            return path
        root = _models_dir() / "ultralytics"
        direct = root / model_name
        if direct.exists():
            return direct
        for subdir in ("segm", "bbox"):
            candidate = root / subdir / model_name
            if candidate.exists():
                return candidate
        return direct

    @staticmethod
    def _resolve_sam_path(checkpoint: str) -> Path:
        path = Path(checkpoint)
        if path.is_absolute():
            return path
        direct = _models_dir() / "sams" / checkpoint
        if direct.exists():
            return direct
        return direct

    @staticmethod
    def _write_stats(stats: Dict, prefix: str) -> Optional[str]:
        if folder_paths is None:
            return None
        out_dir = Path(folder_paths.get_output_directory())
        safe_prefix = prefix.replace("\\", "/").strip("/")
        subdir = out_dir / Path(safe_prefix).parent
        subdir.mkdir(parents=True, exist_ok=True)
        stem = Path(safe_prefix).name or "refiner_stats"
        path = subdir / f"{stem}_{int(time.time() * 1000)}.json"
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return str(path)


NODE_CLASS_MAPPINGS = {
    "FreeFuseSAMMaskRefiner": FreeFuseSAMMaskRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeFuseSAMMaskRefiner": "FreeFuse SAM Mask Refiner",
}
