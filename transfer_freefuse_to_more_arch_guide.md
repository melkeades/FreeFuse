# FreeFuse Transfer Guide: Bringing FreeFuse to More Architectures

> Version: repository-internal practical guide (Diffusers + ComfyUI)
> 
> Audience: contributors who want to add FreeFuse support for a new base model architecture

---

## 中文版

### 0. 写在前面：为什么写这份指南

如果你正在看这份文档，说明你也遇到了同样的问题：

- FreeFuse 已经支持了 `SDXL`、`Flux.dev`、`FLUX.2-klein`、`Z-Image-Turbo`
- 但用户对“更多架构”的需求会持续出现
- 每次从零接入都很耗时、很容易在细节上踩坑

这份文档的目标是把迁移工作流程标准化，让你可以把注意力放在“模型差异”上，而不是反复重造基础工程。

核心思路是三步：

1. 先在 **Diffusers** 里接入 FreeFuse（attn processor + model wrapper + pipeline）
2. 用 `analysis/` 脚本找出最适合抽取 sim map 的 block，并固化 router rule
3. 再迁移到 **ComfyUI**（节点接线 + patch + bypass + 测试）

---

### 1. FreeFuse 的最小工作闭环

无论什么架构，FreeFuse 的最小闭环都一样：

- Phase 1: 从 attention 内部抽 concept similarity map
- sim_map -> mask: 把每个概念映射到空间区域
- 构造 attention bias，抑制 cross-LoRA 干扰
- Phase 2: 回到初始噪声，带 mask/bias 重新生成

在本仓库里，这个闭环对应 3 类文件：

- `src/attn_processor/*`
- `src/models/*`
- `src/pipeline/*`

一个实用原则：

- 先选最像的现有架构做模板，再小步改动

模板建议：

- 双流/混合流 Transformer（Flux2 类）
  - `src/attn_processor/freefuse_flux2_attn_processor.py`
  - `src/models/freefuse_transformer_flux2.py`
  - `src/pipeline/freefuse_flux2_klein_pipeline.py`
- UNet cross-attn（SDXL 类）
  - `src/attn_processor/freefuse_sdxl_attn_processor.py`
  - `src/models/freefuse_unet_sdxl.py`
  - `src/pipeline/freefuse_sdxl_pipeline.py`
- 单流 unified sequence（Z-Image / NextDiT 类）
  - `src/attn_processor/freefuse_z_image_attn_processor.py`
  - `src/models/freefuse_transformer_z_image.py`
  - `src/pipeline/freefuse_z_image_pipeline.py`

---

### 2. 第一步：在 Diffusers 中完成 FreeFuse 支持

#### 2.1 先做架构分类（非常关键）

建议先确认三个信息：

1. attention 是 `cross-attn`（img->txt）还是 `joint/unified`（img+txt 同序列）？
2. 文本与图像 token 的顺序是什么？
3. LoRA 作用层里，哪些层处理 image sequence，哪些处理 text/context？

这三点会直接影响后续实现细节：

- sim_map 抽取公式怎么写
- mask 如何 reshape/pad
- attention bias 的矩阵形状与方向
- FreeFuseLinear 该在哪些层启用 mask

#### 2.2 Attn Processor 迁移清单

目标：在“不破坏原 attention 输出”的前提下，增加 FreeFuse 的附加能力。

建议你在新 processor 里至少有这些状态字段：

- `cal_concept_sim_map`
- `concept_sim_maps` 或 `_last_concept_sim_maps`
- `_freefuse_token_pos_maps`
- `_top_k_ratio`
- `_eos_token_index`
- `_background_token_positions`
- `_attention_bias`（标准字段）

实现点：

1. 保留原始 attention forward 语义
2. 在 pre-softmax logits 阶段支持 additive bias
3. 在指定 block + 指定 step 触发 sim map 抽取
4. 支持 background/eos 分支（建议保留）

建议复用的算法骨架：

1. 用 cross-attn score 做 top-k image token 选择
2. 用 hidden state 内积（concept attention）做最终 sim map
3. `softmax(sim / temperature_or_scale)` 得到概率图

你可以直接对照：

- Flux/Flux2 风格：`src/attn_processor/freefuse_attn_processor.py`、`src/attn_processor/freefuse_flux2_attn_processor.py`
- SDXL 风格（attn1 cache + attn2）：`src/attn_processor/freefuse_sdxl_attn_processor.py`
- Z-Image 单流风格：`src/attn_processor/freefuse_z_image_attn_processor.py`

#### 2.3 Model Wrapper 迁移清单

目标：把 FreeFuse 的运行时控制项统一挂到 model 上，供 pipeline 调度。

你至少需要这些接口：

- `set_freefuse_token_pos_maps(...)`
- `set_freefuse_masks(...)`
- `set_freefuse_attention_bias(...)`
- `set_freefuse_top_k_ratio(...)`
- `set_freefuse_background_info(...)`
- `clear_freefuse_state()` / `reset_freefuse()`
- `enable_concept_sim_map_extraction(block_name)`
- `get_concept_sim_maps()`

此外，建议把 mask 下发逻辑处理清楚：

- image-only 层：下发 `img_len` 对齐的 mask
- unified [txt,img] 层：需要 prepend text 1-mask 再下发
- text/context 专属层：通常不下发 spatial mask，仅可下发 token-pos map

可参考实现：

- Flux2：`src/models/freefuse_transformer_flux2.py`
- Flux1：`src/models/freefuse_transformer_flux.py`
- SDXL：`src/models/freefuse_unet_sdxl.py`
- Z-Image：`src/models/freefuse_transformer_z_image.py`

#### 2.4 Pipeline 迁移清单

目标：在 pipeline 中完成两阶段调度与中间信息衔接。

建议包含：

1. processor 安装函数
2. LoRA 转换（PEFT LoRA -> FreeFuseLinear）
3. tokenizer 对齐的 token position finder
4. Phase1 收集 sim map
5. sim_map -> mask
6. 构建 attention bias
7. Phase2 从同一初始噪声重跑

典型路径：

- FLUX.2-klein：`src/pipeline/freefuse_flux2_klein_pipeline.py`
- SDXL：`src/pipeline/freefuse_sdxl_pipeline.py`
- Z-Image：`src/pipeline/freefuse_z_image_pipeline.py`

建议你在 pipeline 里额外保留 debug 导出：

- concept sim map 热力图
- per-lora mask 图
- attention bias 可视化
- phase1 tensors dump（pt）

这会极大提升你调参效率。

#### 2.5 Diffusers 常见坑

1. 文本模板不一致导致 token 位置全错

- Qwen/Lumina 系列建议保证 concept position 查找与 encode_prompt 使用同一 chat template

2. mask 与 sequence 长度不一致

- 尤其是 packed latent / padding 后的 sequence，需要补齐或插值

3. CFG batch 翻倍未处理

- bias 与 sim_map 在 batch 维要能 repeat 或对齐

4. phase1/phase2 没有复用同一初始噪声

- 会导致“看起来无提升”的假阴性

5. 错把 text-only 层当 image 层下发 spatial mask

- 会出现语义崩坏或生成不稳定

---

### 3. 第二步：用 analysis 脚本找最佳 block 并抽 router rule

这一步的目标不是“出图好看”，而是找到 **最能分离概念** 的采样块。

#### 3.1 现有脚本结构（可直接复用）

- FLUX.2-klein：
  - `analysis/flux2_klein_4b_block_comparison.py`
  - `analysis/flux2_klein_9b_block_comparison.py`
  - `analysis/flux2_klein_block_comparison_common.py`
- SDXL：`analysis/sdxl_block_comparison.py`
- Z-Image：`analysis/zit_block_comparison.py`

建议新架构也做成同样模式：

- 一个 `*_block_comparison_common.py` 封装通用流程
- 若干模型规模脚本只填配置

#### 3.2 推荐实验流程

1. 自动发现可收集 block
2. 对每个 block 运行完整两阶段（固定 seed）
3. 保存：sim_map、mask、final result
4. 生成对比大图（comprehensive grid）
5. 人工 + 指标联合选择最佳 block

实践参数建议：

- 固定 seed
- 小步数快速筛选（如 4/8/12 steps）
- 至少比较：早期、中期、后期 block

#### 3.3 Router Rule 要固化什么

最后要落地到“可配置规则”，至少包含：

- `collect_block`
- `collect_step`
- `top_k_ratio`
- `temperature`（或固定 scale）
- `bg_scale`
- `use_morphological_cleaning`
- `bias_scale / positive_bias_scale / bidirectional`
- `attention_bias_blocks`

建议把这组参数写入：

- 文档示例
- 主脚本默认值
- ComfyUI 的 MODEL_DEFAULTS
- 测试配置

---

### 4. 第三步：迁移到 ComfyUI（架构扩展）

ComfyUI 侧的核心是三层：

- `nodes/`：工作流节点输入输出
- `freefuse_core/`：token、attention patch、bias、bypass hook
- `tests/`：参数回归与对比回归

#### 4.1 新架构接入总清单

1. 先完成模型类型识别

- `freefuse_comfyui/freefuse_core/token_utils.py`
- 需要补：
  - model_type alias
  - model-side detection（优先）
  - tokenizer resolver

2. 再让 token position 查找可用

- 同文件中 `find_concept_positions*` 分支
- 保证与该架构实际 prompt template 对齐

3. 接入 Phase1 attention replace patch

- `freefuse_comfyui/freefuse_core/attention_replace.py`
- 在 `apply_freefuse_replace_patches(...)` 增加新 `model_type` 分支
- 实现该架构的 sim_map 抽取函数

4. 接入 Phase2 attention bias patch

- `freefuse_comfyui/freefuse_core/attention_bias.py`
- `freefuse_comfyui/freefuse_core/attention_bias_patch.py`
- 增加该架构的 bias 构建与 block 路由

5. 对齐 bypass LoRA mask 作用层

- `freefuse_comfyui/freefuse_core/bypass_lora_loader.py`
- 关键是 `_check_*_layer` 与 `mask_type`
- 定义清楚该架构 sequence 排列（例如 `[txt,img]` 还是 `[img,txt]`）

6. 节点层参数与默认值对齐

- `freefuse_comfyui/nodes/sampler.py`
- `freefuse_comfyui/nodes/mask_applicator.py`
- `freefuse_comfyui/nodes/concept_map.py`
- `freefuse_comfyui/nodes/lora_loader.py`

#### 4.2 一个建议先明确的点：序列顺序

同样叫 unified sequence，不同实现可能顺序相反。

- Diffusers Z-Image 侧逻辑是按其内部顺序处理
- ComfyUI Lumina2/NextDiT 侧有自己的拼接顺序约定

这一步直接复用往往不太稳妥，建议在代码中显式验证：

- token_pos_map 是否落在 text 区间
- mask 是否落在 image 区间
- bias 的象限是否对应正确

#### 4.3 ComfyUI 常见坑

1. model_type 识别走了 clip-only 分支，识别成错误模型
2. tokenizer 找对了，但 prompt template 没对齐
3. bias blocks 预设沿用 Flux 命名，实际架构 block 名不同
4. bypass hook 对错误层施加 spatial mask
5. latent/packed 空间尺寸假设错误，导致 mask reshape 崩坏

---

### 5. 测试扩展（重点：你指定的两个文件）

你点名的两个测试文件是迁移完成度的关键验收点。

## 5.1 `freefuse_comfyui/tests/test_parameters.py`

新增架构时建议补齐：

1. 新模型加载函数

- 参考 `load_flux_models` / `load_sdxl_models` / `load_zimage_models`
- 新增 `load_<new_model>_models()`

2. LoRA 查找策略

- 在 `load_loras(...)` 里为新 model_type 补 LoRA 文件匹配与 adapter 名

3. prompt 与 concept map 模板

- 在 `setup_concepts_and_conditioning(...)` 新增分支
- 特别注意 text encoder 编码路径（是否需要系统 prompt / chat template）

4. 测试参数集

- 按现有模式新增：
  - aspect ratio tests
  - block tests
  - simmap tests
  - bias tests
  - （必要时）mask/lora tests

5. 主入口接线

- 在 `run_test_suite(...)` 与 `main()` 加入新 model_type 分支

推荐最小验收：

- 至少跑通 `--quick`（aspect）
- 至少一组 block sweep
- 至少一组 bias sweep

## 5.2 `freefuse_comfyui/tests/test_freefuse_compare.py`

新增架构时建议补齐：

1. 全局默认表

- `MODEL_ORDER`
- `MODEL_DEFAULTS`

2. 模型加载函数

- 新增 `load_<new_model>_models()`，并在 `run_model_compare(...)` 分支调用

3. LoRA 规则

- `_find_loras_for_model(...)` 增加新 model_type 的 LoRA 策略

4. prompt/config 规则

- `_default_prompt_config(...)`
- `setup_concepts_and_conditioning(...)`

5. 生成路径验证

- baseline 和 freefuse 两条路径都要能跑
- 输出 `baseline.png`、`freefuse.png`、`compare.png`

推荐最小验收：

- `--models <new_model>` 单模型比较通过
- 与现有模型一起 `--models flux,<new_model>` 不互相干扰

---

### 6. 一份可直接执行的迁移 Checklist

#### Diffusers

- [ ] 复制并重命名 attn processor 文件
- [ ] 实现 sim_map 抽取 + attention bias 注入
- [ ] 复制并改造 model wrapper，补齐 `set_freefuse_*` 接口
- [ ] 复制并改造 pipeline，打通两阶段流程
- [ ] 完成 tokenizer 对齐的 concept position 查找
- [ ] 主脚本跑通（至少 2 个 LoRA）

#### Block analysis

- [ ] 写 `analysis/<new_model>_block_comparison*.py`
- [ ] 跑 block sweep 并输出综合对比图
- [ ] 固化 router rule（collect_block 等）

#### ComfyUI

- [ ] `token_utils.py` 新 model_type 识别 + tokenizer + token position
- [ ] `attention_replace.py` Phase1 patch 分支
- [ ] `attention_bias.py` / `attention_bias_patch.py` Phase2 bias 分支
- [ ] `bypass_lora_loader.py` 层级 mask 作用域与 mask_type
- [ ] `nodes/sampler.py`、`nodes/mask_applicator.py` 参数路由
- [ ] `nodes/concept_map.py`（若需要）模板对齐

#### Tests

- [ ] `freefuse_comfyui/tests/test_parameters.py` 新架构全链路
- [ ] `freefuse_comfyui/tests/test_freefuse_compare.py` baseline/freefuse 对比
- [ ] 至少一次 quick + compare + block test

---

### 7. 建议的提交策略（减少返工）

推荐按 4 个 PR（或 4 个 commit 组）推进：

1. Diffusers 最小闭环（能跑通两阶段）
2. Analysis block sweep + 默认 router rule
3. ComfyUI 接入（节点 + core）
4. 测试与文档

这样每一步都可验证，定位问题也更快。

---

### 8. 结束语

你不需要一次把所有参数调到完美。

更推荐的做法是：

- 先保证迁移链路正确
- 再用 block comparison 找到可靠起点
- 最后在 ComfyUI 里把默认参数调到“开箱能用”

当这三件事做完，新架构基本就不是“试验性支持”，而是可维护、可复用、可扩展的正式支持了。

如果你按这份指南迁移了一个新架构，欢迎把你最终的 router rule 和默认参数回填到文档里，后面的贡献者会非常感谢你。

---

## English Version

### 0. Why this guide exists

If you are reading this, you probably face the same situation:

- FreeFuse already supports `SDXL`, `Flux.dev`, `FLUX.2-klein`, and `Z-Image-Turbo`
- Users keep asking for more architectures
- Re-implementing integration from scratch is slow and error-prone

This guide standardizes the transfer workflow so you can focus on model-specific differences instead of rebuilding infrastructure every time.

The migration flow has three stages:

1. Implement FreeFuse in **Diffusers** (attn processor + model wrapper + pipeline)
2. Use `analysis/` scripts to find the best sim-map block and extract router rules
3. Port to **ComfyUI** (node wiring + patching + bypass + tests)

---

### 1. Minimal FreeFuse loop

For any architecture, the core loop is the same:

- Phase 1: collect concept similarity maps from internal attention
- sim_map -> mask: assign concept regions spatially
- construct attention bias to suppress cross-LoRA interference
- Phase 2: regenerate from the same initial noise with masks/bias enabled

In this repo, that loop maps to three file groups:

- `src/attn_processor/*`
- `src/models/*`
- `src/pipeline/*`

Practical rule:

- Start from the closest existing architecture template, then modify incrementally.

Template suggestions:

- Dual/mixed stream Transformer (Flux2-like)
  - `src/attn_processor/freefuse_flux2_attn_processor.py`
  - `src/models/freefuse_transformer_flux2.py`
  - `src/pipeline/freefuse_flux2_klein_pipeline.py`
- UNet cross-attention (SDXL-like)
  - `src/attn_processor/freefuse_sdxl_attn_processor.py`
  - `src/models/freefuse_unet_sdxl.py`
  - `src/pipeline/freefuse_sdxl_pipeline.py`
- Single-stream unified sequence (Z-Image / NextDiT-like)
  - `src/attn_processor/freefuse_z_image_attn_processor.py`
  - `src/models/freefuse_transformer_z_image.py`
  - `src/pipeline/freefuse_z_image_pipeline.py`

---

### 2. Step 1: Diffusers integration

#### 2.1 Classify architecture first

Before coding, it helps to clarify:

1. Is attention `cross-attn` (img->txt) or `joint/unified` (img+txt in one sequence)?
2. What is the exact text/image token order?
3. Which LoRA-affected layers process image sequence vs text/context sequence?

These points influence:

- sim-map extraction logic
- mask reshape/padding strategy
- attention-bias matrix shape and direction
- which FreeFuseLinear layers should receive masks

#### 2.2 Attention processor checklist

Goal: add FreeFuse capabilities without breaking base attention behavior.

Recommended state fields:

- `cal_concept_sim_map`
- `concept_sim_maps` or `_last_concept_sim_maps`
- `_freefuse_token_pos_maps`
- `_top_k_ratio`
- `_eos_token_index`
- `_background_token_positions`
- `_attention_bias` (standard field)

Implementation checklist:

1. Preserve original attention outputs and signatures
2. Support additive bias at pre-softmax logits
3. Trigger sim-map extraction only on selected block + selected step
4. Keep background/eos branch support

Recommended algorithm skeleton:

1. Use cross-attn scores to select top-k image tokens
2. Use hidden-state inner product (concept attention) to produce final sim map
3. Normalize with softmax

Reference implementations:

- Flux/Flux2 style: `src/attn_processor/freefuse_attn_processor.py`, `src/attn_processor/freefuse_flux2_attn_processor.py`
- SDXL style (attn1 cache + attn2): `src/attn_processor/freefuse_sdxl_attn_processor.py`
- Z-Image single-stream style: `src/attn_processor/freefuse_z_image_attn_processor.py`

#### 2.3 Model wrapper checklist

Goal: expose a unified FreeFuse control API on the model class.

Minimum APIs:

- `set_freefuse_token_pos_maps(...)`
- `set_freefuse_masks(...)`
- `set_freefuse_attention_bias(...)`
- `set_freefuse_top_k_ratio(...)`
- `set_freefuse_background_info(...)`
- `clear_freefuse_state()` / `reset_freefuse()`
- `enable_concept_sim_map_extraction(block_name)`
- `get_concept_sim_maps()`

Mask dispatch logic is easier to reason about when explicit:

- image-only layers: image-length masks
- unified [txt,img] layers: prepend text ones, then image mask
- text/context-only layers: usually no spatial mask; token-pos map may still apply

Reference wrappers:

- Flux2: `src/models/freefuse_transformer_flux2.py`
- Flux1: `src/models/freefuse_transformer_flux.py`
- SDXL: `src/models/freefuse_unet_sdxl.py`
- Z-Image: `src/models/freefuse_transformer_z_image.py`

#### 2.4 Pipeline checklist

Goal: orchestrate the full two-phase workflow.

Core pieces:

1. processor setup function
2. LoRA conversion (PEFT LoRA -> FreeFuseLinear)
3. tokenizer-aligned concept token position finder
4. Phase 1 sim-map collection
5. sim_map -> mask conversion
6. attention bias construction
7. Phase 2 regeneration from identical initial noise

Reference pipelines:

- `src/pipeline/freefuse_flux2_klein_pipeline.py`
- `src/pipeline/freefuse_sdxl_pipeline.py`
- `src/pipeline/freefuse_z_image_pipeline.py`

Strongly recommended debug outputs:

- concept sim-map heatmaps
- per-LoRA masks
- attention-bias visualizations
- phase1 tensor dumps (`.pt`)

#### 2.5 Common Diffusers pitfalls

1. Token positions are wrong because template differs from actual prompt encoding
2. Mask length mismatches packed/padded sequence length
3. CFG batch doubling not handled for bias/sim_map
4. Phase1/Phase2 not sharing the same initial noise
5. Applying spatial masks to text-only layers

---

### 3. Step 2: Block exploration and router rule extraction

The goal here is not “best looking image”; it is finding the block with the strongest concept separability.

#### 3.1 Reusable script structure

- FLUX.2-klein:
  - `analysis/flux2_klein_4b_block_comparison.py`
  - `analysis/flux2_klein_9b_block_comparison.py`
  - `analysis/flux2_klein_block_comparison_common.py`
- SDXL: `analysis/sdxl_block_comparison.py`
- Z-Image: `analysis/zit_block_comparison.py`

Recommended pattern for new architecture:

- one `*_block_comparison_common.py` for shared logic
- small model-specific launchers with config only

#### 3.2 Experiment protocol

1. auto-discover collectable blocks
2. run full two-phase generation per block with fixed seed
3. save sim maps, masks, and final result
4. build comprehensive comparison grids
5. pick best block using visual + metric evidence

#### 3.3 Router rules to freeze

At minimum, freeze these as defaults:

- `collect_block`
- `collect_step`
- `top_k_ratio`
- `temperature` (or fixed scale)
- `bg_scale`
- `use_morphological_cleaning`
- `bias_scale / positive_bias_scale / bidirectional`
- `attention_bias_blocks`

Write these defaults into:

- docs
- main example scripts
- ComfyUI `MODEL_DEFAULTS`
- test configs

---

### 4. Step 3: ComfyUI port (architecture extension)

ComfyUI integration has three layers:

- `nodes/`: workflow-facing inputs/outputs
- `freefuse_core/`: token handling, attention patching, bias, bypass hooks
- `tests/`: parameter and compare regressions

#### 4.1 New architecture integration checklist

1. Model-type detection

- `freefuse_comfyui/freefuse_core/token_utils.py`
- add aliases, model-side detection, tokenizer resolver

2. Concept token position support

- same file, `find_concept_positions*` branches
- should match the architecture’s real prompt/template path

3. Phase1 attention replace patch

- `freefuse_comfyui/freefuse_core/attention_replace.py`
- add new `model_type` branch in `apply_freefuse_replace_patches(...)`
- implement similarity-map extraction function for that architecture

4. Phase2 attention bias patch

- `freefuse_comfyui/freefuse_core/attention_bias.py`
- `freefuse_comfyui/freefuse_core/attention_bias_patch.py`
- add bias construction + block routing for new architecture

5. Bypass LoRA mask scope

- `freefuse_comfyui/freefuse_core/bypass_lora_loader.py`
- update layer checks and `mask_type`
- explicitly define sequence order (for example `[txt,img]` vs `[img,txt]`)

6. Node-level defaults and parameter routing

- `freefuse_comfyui/nodes/sampler.py`
- `freefuse_comfyui/nodes/mask_applicator.py`
- `freefuse_comfyui/nodes/concept_map.py`
- `freefuse_comfyui/nodes/lora_loader.py`

#### 4.2 Sequence order is worth making explicit

It is safer not to assume unified-sequence order is the same across implementations.

- Diffusers and ComfyUI may use different concatenation conventions
- verify with runtime checks:
  - token positions should land in text range
  - masks should land in image range
  - attention-bias quadrants should match intended direction

#### 4.3 Common ComfyUI pitfalls

1. model_type falls back to wrong branch
2. tokenizer is correct but template is not
3. bias-block presets copied from another architecture without remapping
4. spatial masks applied to incorrect layer types
5. wrong latent/packed spatial assumptions in reshape logic

---

### 5. Test extension focus (the two files you requested)

### 5.1 `freefuse_comfyui/tests/test_parameters.py`

For a new architecture, it helps to cover:

1. model loader function

- like `load_flux_models` / `load_sdxl_models` / `load_zimage_models`

2. LoRA finder branch

- update `load_loras(...)` with model-specific LoRA naming and adapter names

3. prompt/concept conditioning branch

- update `setup_concepts_and_conditioning(...)`
- ensure text encoding path matches architecture expectations

4. parameter test suites

- add architecture-specific sets for:
  - aspect ratio
  - block sweep
  - sim-map params
  - bias params
  - optionally mask/lora-phase behavior

5. main test routing

- update `run_test_suite(...)` and `main()` for the new model type

Minimum acceptance:

- `--quick` passes
- at least one block sweep passes
- at least one bias sweep passes

### 5.2 `freefuse_comfyui/tests/test_freefuse_compare.py`

For a new architecture, it helps to cover:

1. global defaults

- `MODEL_ORDER`
- `MODEL_DEFAULTS`

2. model loading path

- add `load_<new_model>_models()` and branch in `run_model_compare(...)`

3. LoRA lookup branch

- update `_find_loras_for_model(...)`

4. prompt/config branch

- update `_default_prompt_config(...)`
- update `setup_concepts_and_conditioning(...)`

5. baseline vs FreeFuse run path

- both paths should run and save:
  - `baseline.png`
  - `freefuse.png`
  - `compare.png`

Minimum acceptance:

- `--models <new_model>` works
- mixed run `--models flux,<new_model>` works without cross-model side effects

---

### 6. Practical migration checklist

#### Diffusers

- [ ] clone and adapt attention processor
- [ ] implement sim-map extraction + attention-bias injection
- [ ] adapt model wrapper with full `set_freefuse_*` API
- [ ] adapt pipeline and wire two-phase generation
- [ ] verify tokenizer-aligned concept positions
- [ ] run at least one 2-LoRA example end-to-end

#### Block analysis

- [ ] create `analysis/<new_model>_block_comparison*.py`
- [ ] run block sweep and produce comparison grids
- [ ] freeze router defaults

#### ComfyUI

- [ ] `token_utils.py` model detection + tokenizer + positions
- [ ] `attention_replace.py` Phase1 patch branch
- [ ] `attention_bias.py` / `attention_bias_patch.py` Phase2 bias branch
- [ ] `bypass_lora_loader.py` mask scope and `mask_type`
- [ ] node-level parameter routing in sampler/mask applicator

#### Tests

- [ ] extend `test_parameters.py`
- [ ] extend `test_freefuse_compare.py`
- [ ] run quick + compare + block checks

---

### 7. Suggested commit strategy

Recommended 4-step delivery:

1. Diffusers minimal two-phase support
2. Analysis sweeps + router defaults
3. ComfyUI support (nodes + core)
4. Tests + docs

This keeps each stage verifiable and reduces debugging overhead.

---

### 8. Final note

You do not need perfect tuning at day one.

A robust order is:

- first make the transfer pipeline correct
- then find a reliable block/router baseline
- then tune defaults for out-of-box usability in ComfyUI

Once these three are done, your new architecture support is maintainable and production-ready, not just experimental.

If you add a new architecture, please append your finalized router defaults to this guide so future contributors can move faster.
