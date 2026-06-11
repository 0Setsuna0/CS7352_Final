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
- Baseline 性能：quality preset（480×720, 49帧, 40步）约 88s transformer time

### Phase 1：纯张量 Token Merging 库 ✅

文件：`src/tokmerge/merging.py`

核心数据结构：
- `MergeConfig`：合并参数配置（ratio, mode, scope, rope_mode, prop_attn 等）
- `RestoreInfo`：存储 src_idx、dst_idx、keep_idx、sizes，供 merge/unmerge 使用

核心函数：
- `_build_spatial_partition`：棋盘格空间分区（src/dst 二分）
- `_build_spatiotemporal_partition`：时空联合分区
- `bipartite_soft_match`：双边软匹配，支持 adaptive（余弦相似度）和 fixed（网格距离）两种模式
- `merge_tokens`：size-weighted average 合并（vectorized scatter_add）
- `unmerge_tokens`：scatter 还原到原始序列长度
- `size_log_bias`：proportional attention 的 log(size) bias

测试：`tests/test_merging.py`，25 个测试用例全部通过。

### Phase 2：Config Plumbing ✅

文件：`src/tokmerge/runtime.py`

- `load_merge_config`：从 JSON 加载配置
- `attach_merge_config` / `detach_merge_config`：将配置附加到 transformer 的每个 block
- Layer 策略：middle、middle_wide、late_off、all + 显式列表

### Phase 3：pre_attn_restore 正确性桥接 ✅

作为调试路径实现：merge → attention → restore → residual。验证 latent MSE 在噪声地板内。

### Phase 4–5：Attention on Reduced Sequence + Block Scope ✅

修改的 vendored 文件：
- `cogvideox_transformer_3d.py`：CogVideoXBlock.forward 中实现三种 scope
- `attention_processor.py`：CogVideoXAttnProcessor2_0 中实现 RoPE 和 prop_attn

Block scope 流程：
1. norm1 → bipartite matching → merge video tokens
2. Attention on merged sequence（含 text tokens）
3. Unmerge → residual add
4. norm2 → merge（复用同一 pattern，fresh sizes）→ FF → unmerge → residual add

### Phase 6：Spatiotemporal 变体 ✅

- 支持 `temporal_window` 参数控制跨帧匹配范围
- `_build_temporal_mask` 限制只匹配时间邻域内的 token

---

## 三、性能结果

| 配置 | transformer_sec | avg_step (s) | Speedup |
|------|----------------|--------------|---------|
| Baseline (no merge) | 88.42 | 2.21 | 1.00× |
| Block + adaptive r=0.2 | 77.87 | 1.94 | **1.14×** |
| Block + fixed r=0.2 | 78.26 | 1.95 | **1.13×** |
| Block + fixed r=0.4 | 64.10 | 1.60 | **1.38×** |

### CPU Offload 影响

| Offload 模式 | Baseline (s) | Merge r=0.2 (s) | Speedup |
|-------------|-------------|-----------------|---------|
| Sequential CPU offload | 138.14 | 155.22 | 0.89× (**更慢**) |
| No offload (GPU-only) | 88.42 | 78.26 | **1.13×** |

**关键发现**：Token merging 仅在计算为瓶颈时有效加速。Sequential offload 下 CPU↔GPU 传输开销主导，merge 的额外开销反而拖慢速度。

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

```python
# 修复后
ff_info = _RestoreInfo(
    src_idx=info.src_idx, dst_idx=info.dst_idx,
    keep_idx=info.keep_idx,
    sizes=torch.ones(B_cur, info.keep_idx.shape[1], ...),
    num_video_tokens=info.num_video_tokens, grid=info.grid,
)
```

#### Bug 5：Adaptive matching 的 RoPE 批处理错误

**现象**：CFG（classifier-free guidance）下生成质量异常。

**原因**：adaptive matching 中 `keep_idx` 可能因条件/无条件分支不同而在 batch 维度不一致，但 RoPE 用了 `keep_idx[0]` 为所有 batch 元素选择位置编码。

**修复**：检测 `keep_idx` 是否跨 batch 一致；不一致时逐 batch 应用 RoPE。

```python
same_idx = (keep_idx[0] == keep_idx).all() if keep_idx.shape[0] > 1 else True
if same_idx:
    # 共享索引路径
else:
    # 逐 batch 应用 RoPE
    for b in range(keep_idx.shape[0]):
        rope_idx_b = keep_idx[b]
        ...
```

### 4.3 视觉质量问题

#### 问题：r=0.1 就出现可见噪点纹理

**现象**：对比 baseline 和 r=0.1 的第一帧，背景竹林区域出现细粒度噪点。

**分析**：
1. CogVideoX 使用**联合注意力**（text + video token 同池 attend），merge 缩短视频序列后所有 token（包括受保护的 frame-0 和 text）的注意力模式都改变
2. 棋盘格分区在 unmerge 后产生规律性的"复制 vs 原生"位置交替，40 步去噪累积放大
3. 10 层（layers 10–19）× 40 步的误差积累过大

**修复方案**：
1. 新增 `skip_early_ratio` 参数：跳过前 N% 的去噪步（噪声阶段），只在后期（图像成形后）才启用 merge
2. 减少活跃层数：从 10 层缩减到 6 层（layers 12–17）
3. 创建保守配置 `block_r10_skip40.json`（r=0.1, skip 40%, 6 layers）

---

## 五、性能优化

| 优化项 | 效果 |
|-------|------|
| 分区缓存 | 棋盘格 pattern 只计算一次 |
| 索引缓存 | src/dst 索引按 grid shape + device 缓存 |
| 特征降维 | matching 用 128-dim 子采样代替 1920-dim（15× bmm 加速）|
| Vectorized merge/unmerge | scatter_add + searchsorted，无 Python 循环 |
| Fixed matching | 预计算网格距离配对，每步零匹配开销 |
| Top-level imports | 消除每次调用的 import 开销 |

---

## 六、项目文件结构

```
cs7352/
├── src/tokmerge/
│   ├── __init__.py
│   ├── merging.py          # 核心 merge/unmerge/matching 库
│   └── runtime.py          # Config 加载 + transformer 挂载
├── tests/
│   └── test_merging.py     # 25 个测试用例
├── configs/merge/          # 12 个 JSON 合并配置
├── scripts/
│   ├── setup_env.ps1       # 环境搭建
│   ├── check_env.py        # 环境验证
│   ├── run_baseline_smoke_test.py  # 推理脚本（支持 timing + merge）
│   └── measure_noise_floor.py      # 噪声地板测量
├── cog_diffuser/diffusers/ # Vendored diffusers（修改了 2 个文件）
├── docs/
│   ├── dev_plan_v2.md      # 开发计划 v2
│   ├── v3.md               # 开发计划 v3
│   └── work_summary.md     # 本文档
├── report/
│   ├── experiment_results.md
│   └── baseline_noise_floor.json
└── outputs/                # 生成的视频文件（不入 git）
```

---

## 七、v4 画质增强工程（新增）

### 7.1 Timestep-Adaptive Ratio Schedule

实现了 4 种 merge ratio 调度策略，让 merge 强度随 denoising 进度自适应变化：

- `constant`：固定 ratio（原始行为）
- `cosine`：正弦半波，在活跃窗口中段达到峰值
- `bell`：高斯钟形，peak 在 40%-60% 进度
- `linear_decay`：线性衰减至 0

配合 `skip_early_ratio` + `skip_late_ratio` 形成完整的时间窗口控制：
- 早期（高噪声）：不 merge → 保护结构成形
- 中期：merge 强度最大 → 获取速度
- 晚期（精细细节）：不 merge → 保护纹理和细节

### 7.2 Importance-Aware Token Protection

引入 `protect_topk_ratio` 参数，按 token 激活幅度保护高语义重要性的 token 不被吸收：

- 计算 src token 集合的 L2-norm 作为 importance 度量
- Top-K 高重要性 token 通过 score penalty 避免被选中合并
- 保护主体（熊猫、吉他、手部）等高激活区域

### 7.3 Per-Layer Ratio Decay

通过 `layer_ratio_decay` 参数，让不同深度的 transformer block 使用不同的 merge ratio：

- 浅层 block（负责全局结构）使用完整 ratio → 速度最大化
- 深层 block（负责细节恢复）使用衰减后的 ratio → 保护精细纹理
- 公式：`effective_ratio = base * (1 - decay * rank / (n_active - 1))`

### 7.4 CFG-Consistent Merge Pattern

通过 `cfg_consistent=true`，强制 classifier-free guidance 的 conditional 和 unconditional 分支共享同一 merge pattern：

- 避免两分支独立 matching 导致 CFG 差分时的空间错位
- 统一使用 conditional 分支（batch[1]）的 matching 结果

### 7.5 Interpolated Unmerge

新增 `unmerge_mode="interpolate"` 替代原始的"复制 dst 值"策略：

- 对每个被吸收的 src token，计算其 4-connected 空间邻居的均值
- 最终值 = 70% × dst值 + 30% × 邻居均值
- 显著减少"块状复制"感和网格纹伪影

### 7.6 新增配置文件

| 配置 | 描述 |
|------|------|
| `v4_block_cosine_r25.json` | 主推配置：cosine schedule + 全部 v4 增强 |
| `v4_block_bell_r30.json` | 激进配置：bell schedule, r=0.3 |
| `v4_block_conservative_r20.json` | 保守配置：middle layers, r=0.2 |
| `v4_kv_cosine_r20.json` | KV-only scope + cosine schedule |
| `v4_st_bell_r20.json` | 时空模式 + bell schedule |
| `v4_block_aggressive_r40.json` | 极限测试：r=0.4 + 所有保护 |
| `v4_ablation_no_schedule_r25.json` | 消融：无 schedule（对照组） |
| `v4_ablation_no_protect_r25.json` | 消融：无 importance protection（对照组） |

---

## 八、待完成工作

| 阶段 | 描述 | 状态 |
|------|------|------|
| Phase 7 | 跑 v4 配置 benchmark + 微消融 | 未开始 |
| Phase 8 | Metrics (CLIP, VBench, dynamic_degree) + Final Matrix | 未开始 |
| Phase 9 | 分析、图表、最终报告 | 未开始 |
| 质量验证 | v4 配置 vs 旧 block_r20 对比 | 未开始 |
