#!/usr/bin/env python
"""
Lightweight tests for FreeFuse Qwen-Image-2512 support.
"""

import os
import sys
import importlib.util

import torch
import torch.nn as nn


def _find_comfyui_dir(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "comfy")) and os.path.isfile(os.path.join(cur, "main.py")):
            return cur
        sibling = os.path.join(cur, "ComfyUI")
        if os.path.isdir(sibling) and os.path.isdir(os.path.join(sibling, "comfy")):
            return sibling
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError("Could not locate ComfyUI directory")


def _load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FREEFUSE_COMFYUI_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(FREEFUSE_COMFYUI_DIR)
COMFYUI_DIR = _find_comfyui_dir(REPO_ROOT)
if COMFYUI_DIR not in sys.path:
    sys.path.insert(0, COMFYUI_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(COMFYUI_DIR)

TOKEN_UTILS = _load_module(
    "token_utils_qwen_image_test",
    os.path.join(FREEFUSE_COMFYUI_DIR, "freefuse_core", "token_utils.py"),
)

from freefuse_comfyui.freefuse_core.attention_replace import (  # noqa: E402
    FreeFuseQwenImageBlockReplace,
    FreeFuseState,
)
from freefuse_comfyui.freefuse_core.attention_bias import AttentionBiasConfig  # noqa: E402
from freefuse_comfyui.freefuse_core.attention_bias_patch import (  # noqa: E402
    FreeFuseQwenImageBiasBlockReplace,
    _freefuse_sdpa_attention,
    apply_attention_bias_patches,
)
from freefuse_comfyui.freefuse_core.bypass_lora_loader import (  # noqa: E402
    MultiAdapterBypassForwardHook,
)


IM_START = 151644
IM_END = 151645
USER = 872
NEWLINE = 198


class FakeQwenTokenizer:
    def __init__(self):
        self.id_to_text = {
            IM_START: "<|im_start|>",
            IM_END: "<|im_end|>",
            USER: "user",
            NEWLINE: "\n",
            100: "system",
            101: "sys",
            10: "harry",
            11: " potter",
            12: " and",
            13: " forest",
            14: " assistant",
        }
        self.text_to_id = {v.strip(): k for k, v in self.id_to_text.items()}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.id_to_text.get(int(i), f"<{int(i)}>") for i in ids)

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in text.replace("\n", " \n ").split():
            ids.append(self.text_to_id.get(piece.strip(), 999))
        return ids


class _SubTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class FakeClipTokenizerWrapper:
    def __init__(self, tokenizer):
        self.qwen25_7b = _SubTokenizer(tokenizer)


class FakeClip:
    def __init__(self):
        self.fake_tokenizer = FakeQwenTokenizer()
        self.tokenizer = FakeClipTokenizerWrapper(self.fake_tokenizer)

    def tokenize(self, text, **kwargs):
        token_ids = [
            IM_START, 100, NEWLINE, 101, IM_END, NEWLINE,
            IM_START, USER, NEWLINE,
            10, 11, 12, 13,
            IM_END, NEWLINE, IM_START, 14, NEWLINE,
        ]
        return {"qwen25_7b": [[(tid, 1.0) for tid in token_ids]]}


def test_model_detection_and_aliases():
    class QwenImageTransformer2DModel:
        pass

    class Inner:
        diffusion_model = QwenImageTransformer2DModel()

    class Patcher:
        model = Inner()

    assert TOKEN_UTILS.detect_model_type(model=Patcher()) == "qwen_image"
    assert TOKEN_UTILS.detect_model_type(model_type_hint="qwen-image-2512") == "qwen_image"

    class ModelConfig:
        unet_config = {"image_model": "qwen_image"}

    class ConfigInner:
        model_config = ModelConfig()

    class ConfigPatcher:
        model = ConfigInner()

    assert TOKEN_UTILS.detect_model_type(model=ConfigPatcher()) == "qwen_image"


def test_qwen25_tokenizer_resolution_and_template_trim():
    clip = FakeClip()
    tokenizer = TOKEN_UTILS.get_tokenizer_for_model(clip, "qwen_image")
    assert tokenizer is clip.fake_tokenizer

    pos = TOKEN_UTILS.find_concept_positions(
        clip=clip,
        prompts="harry potter and forest",
        concepts={"hero": "harry potter", "place": "forest"},
        filter_meaningless=True,
        filter_single_char=True,
        model_type="qwen_image",
    )
    assert pos["hero"][0] == [0, 1]
    assert pos["place"][0] == [3]


def test_qwen_image_background_positions():
    clip = FakeClip()
    explicit = TOKEN_UTILS.find_background_positions(
        clip=clip,
        prompt="harry potter and forest",
        background_text="forest",
        model_type="qwen_image",
    )
    automatic = TOKEN_UTILS.find_background_positions(
        clip=clip,
        prompt="harry potter and forest",
        background_text=None,
        model_type="qwen_image",
    )
    assert explicit == [3]
    assert automatic == []


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))

    def forward(self, x):
        return x


def test_qwen_image_lora_layer_routing():
    cases = [
        ("diffusion_model.transformer_blocks.0.attn.to_q", True, "img_only"),
        ("diffusion_model.transformer_blocks.0.attn.to_k", True, "img_only"),
        ("diffusion_model.transformer_blocks.0.attn.to_v", True, "img_only"),
        ("diffusion_model.transformer_blocks.0.attn.to_out.0", True, "img_only"),
        ("diffusion_model.transformer_blocks.0.img_mlp.net.0.proj", True, "img_only"),
        ("diffusion_model.transformer_blocks.0.attn.add_q_proj", False, None),
        ("diffusion_model.transformer_blocks.0.attn.add_k_proj", False, None),
        ("diffusion_model.transformer_blocks.0.attn.add_v_proj", False, None),
        ("diffusion_model.transformer_blocks.0.attn.to_add_out", False, None),
        ("diffusion_model.transformer_blocks.0.txt_mlp.net.0.proj", False, None),
        ("diffusion_model.transformer_blocks.0.img_mod.1", False, None),
        ("diffusion_model.transformer_blocks.0.txt_mod.1", False, None),
    ]
    for key, expected_apply, expected_type in cases:
        hook = MultiAdapterBypassForwardHook(DummyModule(), module_key=key)
        assert hook._should_apply_spatial_mask() == expected_apply, key
        assert hook._get_mask_type() == expected_type, key


class FakeQwenAttention(nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.add_q_proj = nn.Linear(dim, dim, bias=False)
        self.add_k_proj = nn.Linear(dim, dim, bias=False)
        self.add_v_proj = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.norm_added_q = nn.Identity()
        self.norm_added_k = nn.Identity()
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Dropout(0.0)])
        self.to_add_out = nn.Linear(dim, dim, bias=False)


class FakeQwenBlock(nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        self.num_attention_heads = heads
        self.img_mod = nn.Linear(dim, 6 * dim)
        self.txt_mod = nn.Linear(dim, 6 * dim)
        self.img_norm1 = nn.Identity()
        self.txt_norm1 = nn.Identity()
        self.img_norm2 = nn.Identity()
        self.txt_norm2 = nn.Identity()
        self.attn = FakeQwenAttention(dim=dim, heads=heads)
        self.img_mlp = nn.Linear(dim, dim)
        self.txt_mlp = nn.Linear(dim, dim)

    def _modulate(self, x, mod_params, timestep_zero_index=None):
        shift, scale, gate = torch.chunk(mod_params, 3, dim=-1)
        return shift.unsqueeze(1) + x * (1 + scale.unsqueeze(1)), gate.unsqueeze(1)

    def _apply_gate(self, x, y, gate, timestep_zero_index=None):
        return y + gate * x


def test_qwen_image_phase1_collection_shape():
    torch.manual_seed(1)
    state = FreeFuseState()
    state.phase = "collect"
    state.collect_step = 0
    state.collect_block = 1
    state.collect_block_end = 1
    state.token_pos_maps = {"hero": [[0, 2]]}
    state.top_k_ratio = 0.5
    state.temperature = 4000.0

    block = FakeQwenBlock()
    replacer = FreeFuseQwenImageBlockReplace(state, block=block, block_index=1)
    fn = replacer.create_block_replace()

    img = torch.randn(1, 4, 16)
    txt = torch.randn(1, 5, 16)
    vec = torch.randn(1, 16)

    def original_block(args):
        return {"img": args["img"], "txt": args["txt"]}

    out = fn(
        {
            "img": img,
            "txt": txt,
            "vec": vec,
            "pe": None,
            "transformer_options": {"sigmas_index": 0},
        },
        {"original_block": original_block},
    )

    assert out["img"].shape == img.shape
    assert "hero" in state.similarity_maps
    assert state.similarity_maps["hero"].shape == (1, 4, 1)
    assert 1 in state.collected_outputs["block_similarity_maps"]


class FakeModelPatcher:
    def __init__(self, num_blocks=4):
        self.calls = []

        class Diffusion:
            pass

        class Inner:
            pass

        self.model = Inner()
        self.model.diffusion_model = Diffusion()
        self.model.diffusion_model.transformer_blocks = [object() for _ in range(num_blocks)]

    def set_model_patch_replace(self, fn, model_part, block_kind, block_index):
        self.calls.append((model_part, block_kind, int(block_index), fn))


def test_qwen_image_attention_bias_routing_and_dimensions():
    patcher = FakeModelPatcher(num_blocks=4)
    config = AttentionBiasConfig(
        enabled=True,
        bias_scale=5.0,
        positive_bias_scale=1.0,
        bidirectional=True,
        use_positive_bias=True,
        apply_to_blocks="last_half_double",
    )
    lora_masks = {"hero": torch.ones(1, 4)}
    token_pos_maps = {"hero": [[0, 1]]}

    apply_attention_bias_patches(
        model_patcher=patcher,
        attention_bias=None,
        config=config,
        txt_seq_len=5,
        model_type="qwen_image",
        lora_masks=lora_masks,
        token_pos_maps=token_pos_maps,
    )
    assert [call[2] for call in patcher.calls] == [2, 3]
    assert all(call[1] == "double_block" for call in patcher.calls)

    replacer = FreeFuseQwenImageBiasBlockReplace(
        lora_masks=lora_masks,
        token_pos_maps=token_pos_maps,
        config=config,
        block_index=2,
        block=object(),
    )
    bias = replacer._get_or_build_bias(
        txt_len=5,
        img_len=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert bias.shape == (1, 9, 9)
    assert bias[0, 5:, :5].abs().sum().item() > 0


def test_dense_bias_attention_uses_sdpa_shape():
    torch.manual_seed(4)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)
    dense_bias = torch.zeros(1, 1, 6, 6)
    dense_bias[:, :, 2:, :2] = -3.0

    out = _freefuse_sdpa_attention(
        q,
        k,
        v,
        heads=2,
        mask=dense_bias,
        skip_reshape=True,
    )

    assert out.shape == (1, 6, 16)
    assert torch.isfinite(out).all()


def test_optional_qwen_model_files_smoke_skip():
    model_files = [
        os.path.join(COMFYUI_DIR, "models", "diffusion_models", "qwen_image_2512_fp8_e4m3fn.safetensors"),
        os.path.join(COMFYUI_DIR, "models", "text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
        os.path.join(COMFYUI_DIR, "models", "vae", "qwen_image_vae.safetensors"),
    ]
    if not all(os.path.exists(path) for path in model_files):
        print("Skipping Qwen-Image-2512 integration smoke test; model files are not installed.")
        return
    print("Qwen-Image-2512 model files are present; integration smoke is left to ComfyUI workflow execution.")


def run_all_tests():
    test_model_detection_and_aliases()
    test_qwen25_tokenizer_resolution_and_template_trim()
    test_qwen_image_background_positions()
    test_qwen_image_lora_layer_routing()
    test_qwen_image_phase1_collection_shape()
    test_qwen_image_attention_bias_routing_and_dimensions()
    test_dense_bias_attention_uses_sdpa_shape()
    test_optional_qwen_model_files_smoke_skip()
    print("All Qwen-Image support tests passed.")


if __name__ == "__main__":
    run_all_tests()
