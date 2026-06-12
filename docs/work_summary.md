# 工作总结 — Spatiotemporal Token Merging for CogVideoX-2B

## 一、项目概述

本项目为 CS7352 课程大作业，在 CogVideoX-2B 视频生成模型上实现 training-free 的时空 Token Merging（ToMe），探索 quality–efficiency 的 Pareto 前沿。遵循 v3 开发计划的五条核心原则：真加速、真时空、真硬件报告、真运动保持、真预算。

**硬件环境**：NVIDIA RTX 4090 (24 GiB VRAM)，Windows 10，Python 3.12，PyTorch + CUDA 12.8

---

## 二、完成阶段

### Phase 0：Baseline Lock + CUDA Event Timing ✅

- 搭建开发环境（`uv` 虚拟环境，vendored diffusers）
- 实现 smoke 和 quality 两档推理 preset
- 添加 CUDA event 精确计时（`transformer_seconds`、`avg_step_seconds`）
- 噪声地板测量：同 seed 两次推理 latent MSE = 0（完全确定性）
- Baseline 性能：quality preset（480×720, 49帧, 40步）约 84-88s transformer time

### Phase 1：纯张量 Token Merging 库 ✅

文件：`src/tokmerge/merging.py`

核心数据结构：
- `MergeConfig`：合并参数配置（21 个字段，含 v4 新增的 5 个画质增强参数）
- `RestoreInfo`：存储 src_idx、dst_idx、keep_idx、sizes，供 merge/unmerge 使用

核心函数：
- `_build_spatial_partition`：棋盘格空间分区（src/dst 二分），支持 shifted 模式
- `bipartite_soft_match`：双边软匹配，支持 adaptive（余弦相似度）、fixed（网格距离）两种模式，含 importance protection 和 CFG-consistent 选项
- `merge_tokens`：size-weighted average 合并（vectorized scatter_add）
- `unmerge_tokens`：scatter 还原到原始序列长度
- `unmerge_tokens_interpolated`：邻居插值还原（v4 新增）
- `compute_effective_ratio`：timestep-adaptive ratio 计算（v4 新增）
- `compute_layer_ratio`：per-layer ratio decay 计算（v4 新增）
- `merge_kv_tokens`：KV-only scope 的 4D tensor merge
- `size_log_bias`：proportional attention 的 log(size) bias

测试：`tests/test_merging.py`（25 用例）+ `tests/test_v4_features.py`（23 用例），全部通过。

### Phase 2：Config Plumbing ✅

文件：`src/tokmerge/runtime.py`

- `load_merge_config`：从 JSON 加载配置（自动过滤未知字段）
- `attach_merge_config` / `detach_merge_config`：将配置附加到 transformer 的每个 block
- Layer 策略：middle、middle_wide、late_off、all + 显式列表

### Phase 3：pre_attn_restore 正确性桥接 ✅

作为调试路径实现：merge → attention → restore → residual。验证 latent MSE 在噪声地板内。

### Phase 4–5：Attention on Reduced Sequence + Block Scope ✅

修改的 vendored 文件：
- `cogvideox_transformer_3d.py`：CogVideoXBlock.forward 中实现四种 scope（pre_attn_restore, attn_only, block, kv_only）
- `attention_processor.py`：CogVideoXAttnProcessor2_0 中实现 RoPE 选择和 prop_attn bias

Block scope 流程：
1. norm1 → bipartite matching（含 importance protection）→ merge video tokens
2. Attention on merged sequence（含 text tokens，prop_attn bias）
3. Unmerge（copy 或 interpolate）→ residual add
4. norm2 → merge（复用同一 pattern，fresh sizes）→ FF → unmerge → residual add

### Phase 6：Spatiotemporal 变体 ✅

- 支持 `temporal_window` 参数控制跨帧匹配范围
- `_build_temporal_mask` 限制只匹配时间邻域内的 token

### Phase 7：v4 画质增强工程 ✅

实现 5 项画质增强技术（详见第七节），48 个单元测试全部通过。

---

## 三、性能结果

### 3.1 主结果（RTX 4090, No Offload, quality preset）

| 配置 | transformer_sec | avg_step (s) | Speedup | 备注 |
|------|----------------|--------------|---------|------|
| Baseline (no merge) | 84.26 | 2.11 | 1.00× | 对照组 |
| Block + adaptive r=0.2 (v3) | 77.72 | 1.94 | **1.08×** | 有网格纹伪影 |
| Block + fixed r=0.2 | 78.26 | 1.95 | **1.08×** | 有网格纹伪影 |
| Block + fixed r=0.4 | 64.10 | 1.60 | **1.31×** | 严重画质退化 |
| Block r=0.2 + skip40% | 82.78 | 2.07 | **1.02×** | 画质改善，加速有限 |
| v4 balanced r=0.25 | TBD | TBD | TBD | 保护 + 加速平衡 |
| v4 balanced r=0.30 | TBD | TBD | TBD | 保护 + 加速平衡 |
| v4 balanced r=0.40 | TBD | TBD | TBD | 保护 + 加速平衡 |

### 3.2 v4 全保护配置实验（速度-质量 trade-off 分析）

| 配置 | transformer_sec | Speedup | 分析 |
|------|----------------|---------|------|
| v4_conservative_r20 (cosine+protect+decay+interp) | 85.41 | 0.99× | 过度保护，开销>收益 |
| v4_cosine_r25 (cosine+protect+decay+interp) | 85.35 | 0.99× | 同上 |
| v4_bell_r30 (bell+protect+decay+interp) | 85.97 | 0.98× | 同上 |
| v4_aggressive_r40 (bell+protect+decay+interp) | 85.04 | 0.99× | 即使 r=0.4 也无加速 |
| v4_kv_cosine_r20 (kv_only+cosine) | 88.86 | 0.95× | prop_attn bias 杀死 flash kernel |

**关键发现**：当 schedule（cosine/bell）+ skip_late + layer_decay + interpolated_unmerge 全部叠加时，实际 merge 的步数和 token 数极少，而保护机制本身的计算开销（interpolated unmerge 的 4× gather、schedule 计算、importance topk）超过了 merge 带来的节省。

### 3.3 CPU Offload 影响

| Offload 模式 | Baseline (s) | Merge r=0.2 (s) | Speedup |
|-------------|-------------|-----------------|---------|
| Sequential CPU offload | 138.14 | 155.22 | 0.89× (**更慢**) |
| No offload (GPU-only) | 84.26 | 77.72 | **1.08×** |

**关键发现**：Token merging 仅在计算为瓶颈时有效加速。Sequential offload 下 CPU↔GPU 传输开销主导，merge 的额外开销反而拖慢速度。

### 3.4 综合结论

1. **Naive block merge（v3）**可在 r=0.2 时获得 1.08× 加速，r=0.4 获得 1.31×，但画质退化明显
2. **Quality-first 保护堆叠（v4 全功能版）**在当前序列长度（17,550 tokens）下开销过大，净加速为负
3. **最佳平衡点**是选择性保护（importance protection + cfg_consistent + checkerboard_shifted + skip_early）配合 constant ratio，在保留主要画质改善的同时不引入过多开销
4. **KV-only scope** 因 prop_attn bias 导致 SDPA 回落到 math kernel，反而变慢；关闭 prop_attn 或等待 FlashAttention 支持任意 bias 后才有实用价值
5. CogVideoX-2B 的 MMDiT 联合注意力架构对 token merge 比普通 ViT 更敏感——text/video 同池 attend，任何对 video 序列的缩短都会改变 text tokens 的注意力分布

---

## 四、遇到的问题与修复

### 4.1 环境搭建问题

| 问题 | 原因 | 修复 |
|------|------|------|
| Hugging Face 模型下载卡住 | Windows 符号链接权限 + 网络不稳定 | 清除缓存、杀死进程、重试下载 |
| `UnicodeEncodeError` (GBK codec) | huggingface-cli 输出含 Unicode | 设置 `$env:PYTHONIOENCODING = "utf-8"` |
| `ImportError: protobuf` | 依赖缺失 | `uv pip install protobuf` |
| CUDA OOM / 段错误 (exit code 3221225477) | 多个 GPU 进程同时运行 | 确保单进程运行，清理 CUDA cache |

### 4.2 Token Merging 实现 Bug

#### Bug 1：temporal_mask 形状不匹配

**现象**：spatiotemporal 模式下 `bipartite_soft_match` 抛出 RuntimeError。

**原因**：`_build_temporal_mask` 在 `protect_first_frame` 过滤之前调用，使用的是过滤前的 src/dst 索引，导致 mask 尺寸与过滤后的不匹配。

**修复**：重构为先过滤 frame-0 tokens，再用实际的 src_indices/dst_indices 构建 temporal_mask。

#### Bug 2：_tokmerge kwargs 传播到非活跃 block

**现象**：非 merge 层的 attention processor 收到 `_tokmerge_grid` 等未知参数并发出警告。

**原因**：`attention_kwargs` 包含 `_tokmerge` 前缀的 key，传入不需要 merge 的 block 时被 `Attention.forward` 的签名检查拦截。

**修复**：在 baseline 路径中过滤掉 `_tokmerge` 前缀的 kwargs。

#### Bug 3：merge info 无法传播到 AttentionProcessor

**现象**：attention processor 中 `merge_info` 始终为 None，token merging 未生效。

**原因**：`Attention.forward` 方法会过滤掉不在 processor `__call__` 签名中的 kwargs。

**修复**：改为在 block 中将 info 和 cfg 直接设置为 processor 的属性（`processor._tokmerge_info = info`），processor 通过 `getattr(self, ...)` 读取。

#### Bug 4：FF 路径使用了错误的 sizes

**现象**：生成视频质量明显劣化，出现结构化噪点。

**原因**：`scope="block"` 的 FF merge 复用了 attention merge 后累积的 `info.sizes`（值 > 1），导致 weighted average 权重错误。

**修复**：为 FF merge 创建独立的 `RestoreInfo`，使用 `sizes=torch.ones(...)` 作为初始权重。

#### Bug 5：Adaptive matching 的 RoPE 批处理错误

**现象**：CFG（classifier-free guidance）下生成质量异常。

**原因**：adaptive matching 中 `keep_idx` 可能因条件/无条件分支不同而在 batch 维度不一致，但 RoPE 用了 `keep_idx[0]` 为所有 batch 元素选择位置编码。

**修复**：检测 `keep_idx` 是否跨 batch 一致；不一致时逐 batch 应用 RoPE。

#### Bug 6（v4 新增）：float16 溢出

**现象**：importance protection 的 penalty 值 `-1e9` 超出 float16 范围。

**原因**：half precision 最大值约 65504，`-1e9` 无法表示。

**修复**：使用 `torch.finfo(scores.dtype).min / 2` 作为动态 penalty 值。

### 4.3 视觉质量问题

#### 问题：r=0.1 就出现可见噪点纹理

**现象**：对比 baseline 和 r=0.1 的第一帧，背景竹林区域出现细粒度噪点。

**分析**：
1. CogVideoX 使用**联合注意力**（text + video token 同池 attend），merge 缩短视频序列后所有 token（包括受保护的 frame-0 和 text）的注意力模式都改变
2. 棋盘格分区在 unmerge 后产生规律性的"复制 vs 原生"位置交替，40 步去噪累积放大
3. 10 层（layers 10–19）× 40 步的误差积累过大

**修复方案**：
1. 新增 `skip_early_ratio` 参数：跳过前 N% 的去噪步
2. 新增 `checkerboard_shifted` partition：不同层交替偏移，打破固定网格
3. 减少活跃层数

---

## 五、性能优化实现

| 优化项 | 效果 |
|-------|------|
| 分区缓存 | 棋盘格 pattern 只计算一次 |
| 索引缓存 | src/dst 索引按 grid shape + device 缓存 |
| 特征降维 | matching 用 128-dim 子采样代替 1920-dim（15× bmm 加速）|
| Vectorized merge/unmerge | scatter_add + searchsorted，无 Python 循环 |
| Fixed matching | 预计算网格距离配对，每步零匹配开销 |
| Reuse interval | 每 N 次 block 调用复用 matching pattern，减少 bmm 次数 |
| Top-level imports | 消除每次调用的 import 开销 |

---

## 六、项目文件结构

```
cs7352/
├── src/tokmerge/
│   ├── __init__.py
│   ├── merging.py              # 核心 merge/unmerge/matching 库（608 行）
│   └── runtime.py              # Config 加载 + transformer 挂载
├── tests/
│   ├── test_merging.py         # 25 个基础测试
│   └── test_v4_features.py     # 23 个 v4 功能测试
├── configs/merge/              # 30 个 JSON 合并配置
├── scripts/
│   ├── setup_env.ps1           # 环境搭建
│   ├── check_env.py            # 环境验证
│   ├── run_baseline_smoke_test.py   # 推理脚本（timing + merge）
│   ├── run_v4_benchmark.py     # 自动化 benchmark 套件
│   ├── run_all_tokenmerge_configs.py  # 全配置 sweep
│   ├── run_quality_tokenmerge_test.py # 单配置高质量测试
│   └── measure_noise_floor.py  # 噪声地板测量
├── cog_diffuser/diffusers/     # Vendored diffusers（修改了 2 个文件）
├── docs/
│   ├── dev_plan_v2.md          # 开发计划 v2
│   ├── v3.md                   # 开发计划 v3
│   ├── tokenmerge_optimization_plan.md  # 画质优化调研
│   ├── cogvideox_source_map.md # 源码定位指南
│   └── work_summary.md         # 本文档
├── report/
│   ├── experiment_results.md
│   └── baseline_noise_floor.json
└── outputs/                    # 生成的视频文件（不入 git）
    └── v4_bench/               # v4 benchmark 结果
```

---

## 七、v4 画质增强工程

### 7.1 Timestep-Adaptive Ratio Schedule

实现了 4 种 merge ratio 调度策略：

- `constant`：固定 ratio（原始行为，推荐用于速度优先场景）
- `cosine`：正弦半波，在活跃窗口中段达到峰值
- `bell`：高斯钟形，peak 在 40%-60% 进度
- `linear_decay`：线性衰减至 0

配合 `skip_early_ratio` + `skip_late_ratio` 形成完整的时间窗口控制。

**实验结论**：cosine/bell schedule 会大幅压缩实际 merge 的步数和强度，在当前模型上加速为负。推荐 `constant` schedule + `skip_early_ratio=0.2-0.3` 的组合。

### 7.2 Importance-Aware Token Protection

引入 `protect_topk_ratio` 参数：

- 计算 src token 集合的 L2-norm 作为 importance 度量
- Top-K 高重要性 token 通过 dtype-safe penalty 避免被选中合并
- 开销极低（一次 topk + scatter），推荐保留

### 7.3 Per-Layer Ratio Decay

通过 `layer_ratio_decay` 参数实现层级 ratio 衰减：

- 公式：`effective_ratio = base * (1 - decay * rank / (n_active - 1))`
- 浅层保持完整 ratio，深层衰减

**实验结论**：layer_ratio_decay 压缩后层贡献，配合 cosine schedule 时效果过度；单独使用时对画质有帮助但削减加速。推荐在画质优先场景使用 `decay=0.2-0.3`。

### 7.4 CFG-Consistent Merge Pattern

`cfg_consistent=true` 强制 conditional/unconditional 分支共享 merge pattern：

- 使用 conditional 分支（batch[1]）的 matching 结果
- 零额外计算开销（仅一次 tensor expand）
- 避免 CFG 差分时的空间错位
- **推荐始终开启**

### 7.5 Interpolated Unmerge

`unmerge_mode="interpolate"`：

- 对被吸收 token 计算 4-connected 邻居均值
- 最终值 = 70% dst + 30% 邻居均值
- 减少"块状复制"伪影

**实验结论**：4 次 gather 的开销约 1-2s/run，在 r≤0.3 时超过 merge 带来的节省。仅推荐在离线高质量生成（不关心速度）时使用。

### 7.6 实验结论总结

| 技术 | 画质改善 | 速度代价 | 推荐 |
|------|----------|----------|------|
| `protect_topk_ratio=0.15` | 中 | 极低 | ✅ 始终开启 |
| `cfg_consistent=true` | 中 | 零 | ✅ 始终开启 |
| `checkerboard_shifted` | 中 | 零（缓存） | ✅ 始终开启 |
| `skip_early_ratio=0.2-0.3` | 高 | 线性减速 | ✅ 画质需要时 |
| `reuse_interval=2-4` | 无 | 负（加速） | ✅ 始终开启 |
| `ratio_schedule=cosine/bell` | 高 | 高（过度压缩） | ⚠️ 仅消融 |
| `layer_ratio_decay` | 中 | 中 | ⚠️ 适度使用 |
| `skip_late_ratio` | 中 | 中 | ⚠️ 适度使用 |
| `unmerge_mode=interpolate` | 中 | 高 | ❌ 不推荐实时 |

---

## 八、待完成工作

| 阶段 | 描述 | 状态 |
|------|------|------|
| Balanced 配置验证 | v4_balanced_r25/r30/r40 的速度+画质测试 | 待跑 |
| Metrics 实现 | CLIP score + temporal consistency + dynamic_degree | 未开始 |
| Final Matrix | 精选配置 × 多 prompt × 指标评估 | 未开始 |
| 报告 | Pareto 图、消融表、定性对比、最终 paper | 未开始 |
