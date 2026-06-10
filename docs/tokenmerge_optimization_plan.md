# TokenMerge 画质优化调研与工程方案

日期：2026-06-11  
仓库：`C:\WorkSpace\AI\CS7352_Final`  
目标：在 CogVideoX-2B 上保留 training-free 推理加速，同时显著降低当前 TokenMerge 带来的网格纹、涂抹、主体形变和 temporal ghosting。

## 1. 当前现象与判断

我们已经完成一轮 13 组配置 sweep。速度结果证明当前实现确实能减少 wall-clock time，但视频画质明显退化：

- `block_adaptive_r15.mp4`：背景出现规律性颗粒/网格纹，熊猫和吉他比例改变。
- `st_r20.mp4`：跨帧合并后主体运动区域拖影更重，吉他和手部细节崩坏。
- `block_fixed_r40.mp4`：速度最好，但属于极端配置，画质风险最大。

初步结论：当前代码不是“没生效”，而是 **full-block hidden token merge 太激进**。这种做法把位置 token 物理合并，然后在 block 输出处复制回原始位置；对分类 ViT 或许可以忍受，但对视频扩散模型会在多步 denoising 中放大伪影。

## 2. 相关工作调研

### 2.1 Token Merging / ToMe 系列

ToMe 原始论文提出 bipartite soft matching，在 ViT 中逐层合并相似 token，报告了 ViT 图像/视频任务上的吞吐提升和小精度损失。关键点是：ToMe 用轻量匹配算法合并相似 token，并通过 token size 做 proportional attention，避免 merged token 权重失真。参考：[Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461)，[OpenReview](https://openreview.net/forum?id=JroZRaRw7Eu)。

ToMeSD 将 ToMe 扩展到 Stable Diffusion，说明扩散模型里存在 token 冗余，并可在不训练的情况下获得速度和显存收益。它也提醒我们：扩散模型上的 token merge 需要 diffusion-specific 改造，而不是直接搬 ViT 策略。参考：[Token Merging for Fast Stable Diffusion](https://arxiv.org/abs/2303.17604)，[CVF 页面](https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Bolya_Token_Merging_for_Fast_Stable_Diffusion_CVPRW_2023_paper.html)，[官方代码](https://github.com/dbolya/tomesd)。

### 2.2 视频 token merging

VidToMe 面向视频编辑，在 self-attention 中跨帧合并 token，目标不是单纯速度，而是提升 temporal consistency 并降低显存。它强调“跨帧 token 对齐”而不是简单把相邻帧 token 混合掉。参考：[VidToMe arXiv](https://arxiv.org/abs/2312.10656)，[CVPR 2024 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_VidToMe_Video_Token_Merging_for_Zero-Shot_Video_Editing_CVPR_2024_paper.pdf)，[项目页](https://vidtome-diffusion.github.io/)，[官方代码](https://github.com/lixirui142/VidToMe)。

### 2.3 更适合 DiT 的 token merge 改造

ToMA 是更接近我们问题的工作：它指出普通 ToMe/ToMeSD 的 sorting、scatter write 等操作在现代 attention/FlashAttention 场景下会带来实际开销；并提出将 merge/unmerge 改写为 attention-like linear transformation，利用局部性和跨步模式复用降低开销。参考：[ToMA OpenReview](https://openreview.net/forum?id=51l8tvuIxo)，[ToMA arXiv](https://arxiv.org/abs/2509.10918)，[官方代码](https://github.com/wenboluu/ToMA)。

Importance-based Token Merging 提出用 CFG magnitude 估计 token 重要性，优先保护重要 token，再合并冗余 token。这对我们很关键：当前实现没有显式保护熊猫、吉他、手部等高语义/高运动区域。参考：[Importance-based Token Merging for Diffusion Models](https://arxiv.org/abs/2411.16720)。

### 2.4 非 token-merge 但更稳的 video diffusion 加速

Pyramid Attention Broadcast (PAB) 是 DiT 视频生成里很强的 training-free attention 加速路线。它发现 diffusion 过程中 attention 差异呈 U-shaped pattern，中间大段 timestep 更稳定，因此可以 broadcast/reuse attention outputs。参考：[PAB arXiv](https://arxiv.org/abs/2408.12588)，[项目页](https://oahzxl.github.io/PAB/)。

TeaCache 使用 timestep embedding 估计输出差异并决定是否复用缓存，主打 video diffusion training-free 加速。参考：[TeaCache arXiv](https://arxiv.org/abs/2411.19108)，[官方代码](https://github.com/ali-vilab/TeaCache)。

FasterCache 指出直接复用相邻 step feature 会损失细微变化，提出动态 feature reuse 和 CFG-Cache，强调 high-quality video diffusion acceleration。参考：[FasterCache arXiv](https://arxiv.org/abs/2410.19355)，[项目页](https://vchitect.github.io/FasterCache/)，[官方代码](https://github.com/Vchitect/FasterCache)。

CacheDiT 是 Diffusers 生态里较成熟的 DiT cache acceleration 框架，官方文档称其支持接近所有 Diffusers DiT-based pipelines，可作为工程对照或备选集成路线。参考：[Hugging Face Diffusers CacheDiT docs](https://huggingface.co/docs/diffusers/en/optimization/cache_dit)，[Cache-DiT GitHub](https://github.com/vipshop/cache-dit)。

## 3. 当前代码的主要问题

### 3.1 棋盘格分区导致规律性伪影

当前 `src/tokmerge/merging.py` 里 `_build_spatial_partition` 使用固定棋盘格 `(h + w) % 2` 划分 src/dst。这个设计很容易把误差固定在同一类空间位置上，经过 40 步 denoising 后形成截图里看到的网格/纱窗感。

### 3.2 unmerge 直接复制目标 token

当前 `unmerge_tokens` 把 absorbed token 位置直接复制 destination token 的输出。这样会让局部细节变成块状重复。视频生成模型对这种局部复制非常敏感，尤其背景竹林、毛发、手指、吉他弦这种高频结构会被涂抹。

### 3.3 `pre_rope` 语义不准确

配置里 `rope_mode="pre_rope"`，但当前 block merge 路径实际是：

1. merge hidden states；
2. 进入 attention；
3. 在 shortened sequence 上用 `keep_idx` 选择 RoPE。

这不是严格的“全长 Q/K 先做 RoPE，再 merge rotated Q/K”。对 DiT/CogVideoX 这种强依赖 3D RoPE 位置语义的模型，这会增加空间错位和 temporal ghosting。

### 3.4 full-block scope 伤害太大

`scope="block"` 同时缩短 attention 和 FFN 输入，理论加速好，但也意味着每个被吸收位置在整个 block 中都失去自己的独立表征，只能在 block 末尾复制回来。CogVideoX 联合 attention 中 text/video token 强耦合，这种扰动会影响主体、动作和背景。

### 3.5 CFG batch 独立 matching 可能放大差异

当前 adaptive matching 对 batch 内不同样本独立匹配。对于 classifier-free guidance，conditional/unconditional 分支如果形成不同 merge pattern，最终 CFG 差分可能放大局部错位。这个问题不一定是主因，但值得在优化时控制。

## 4. 推荐主线：KV-only Attention Token Merge

### 4.1 核心思想

不要再合并 full hidden sequence，也不要 unmerge 输出位置。改为：

- Query 保持全长：每个空间/时间位置仍然产生自己的输出。
- Text tokens 不合并。
- 只合并 video Key/Value tokens。
- Attention 输出仍然是全长，因此不需要把 dst 复制回 src。

原始 attention 近似复杂度：

```text
Q: N
K/V: N
cost ~ N * N
```

KV-only merge 后：

```text
Q: N
K/V: N - r
cost ~ N * (N - r)
```

它的加速弱于 full-block merge 的 `(N-r)^2`，也不加速 FFN，但画质风险小很多。对我们现在的情况，这是更合理的 trade-off。

### 4.2 为什么它适合当前项目

- 避免 `unmerge_tokens` 复制造成的网格纹。
- 可以在 attention processor 内完成，改动集中。
- 可以先对 self-attention 做，不动 VAE/text encoder/FFN。
- 和 VidToMe/PAB/ToMA 的方向更一致：减少 attention 冗余，而不是粗暴替换所有 token 表征。
- 仍然符合 proposal 的 training-free token merging 主题。

### 4.3 预期收益

保守估计：

- `ratio=0.1`：attention 部分约 10% K/V 缩短，端到端可能 3%-6%。
- `ratio=0.2`：attention 部分约 20% K/V 缩短，端到端可能 6%-12%。
- 画质应明显优于当前 `block_adaptive_r15/r20/st_r20`。

如果要追求更高速度，可以后续叠加 PAB/TeaCache/CacheDiT，但这应作为第二阶段。

## 5. 具体工程方案

### Phase A：冻结当前结果，作为 naive block merge baseline

保留当前 `outputs/config_sweep` 的结果和截图，写入报告：

- naive block merge 能加速；
- aggressive ratio 会造成明显伪影；
- 这证明“不是所有 token merge 都适合视频 DiT”，为后续优化提供动机。

不建议删除当前结果，它们是很好的反例/消融。

### Phase B：新增 `scope="kv_only"` 路径

改动文件：

- `src/tokmerge/merging.py`
- `src/tokmerge/runtime.py`
- `cog_diffuser/diffusers/src/diffusers/models/attention_processor.py`
- `cog_diffuser/diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py`
- `configs/merge/*.json`

实现要点：

1. 在 `CogVideoXBlock.forward` 中，如果 `scope="kv_only"`，不要 merge `norm_hidden_states`。
2. 把 `merge_cfg`、`grid`、`block_index` 传给 `CogVideoXAttnProcessor2_0`。
3. 在 attention processor 内完成 Q/K/V projection 后，先对全长 Q/K 应用 RoPE。
4. 只对 video K/V 做 merge；text K/V 原样保留。
5. Query 不变，因此 attention 输出长度不变，不需要 unmerge。
6. 对 merged K 加 `log(size)` proportional attention bias。

推荐第一版不做 spatiotemporal，只做 spatial KV-only。先把画质救回来。

### Phase C：替换固定棋盘格，降低规律性伪影

新增 partition 策略：

1. `checkerboard_shifted`：不同 layer/timestep 交替 `(h+w+offset)%2`，避免同一空间位置长期被吸收。
2. `local_window_random`：在 2x2 或 4x4 local tile 内按固定 seed pseudo-random 选 src/dst。
3. `importance_protected`：高重要性 token 不进入 src。

优先级：

```text
checkerboard_shifted -> local_window_random -> importance_protected
```

先做 shifted checkerboard，成本低，能快速验证是否能减少网格纹。

### Phase D：加入 importance protection

参考 Importance-based Token Merging，保护重要 token：

- CFG magnitude 大的 token；
- temporal motion 大的 token；
- attention entropy 低/被 text 强关注的 token；
- frame 0；
- 中心主体区域或高频区域。

第一版可用简单启发式：

```text
importance = ||cond_hidden - uncond_hidden||
保护 top 20%-30% video tokens，不允许它们作为 src 被吸收。
```

如果 CFG batch 组织不好取，可以先用 hidden norm / local variance / frame difference 近似。

### Phase E：保守 temporal merge

当前 `st_r20` 画质很差，不建议继续直接跨帧 full-block merge。

新 temporal 方案：

- 只在 KV-only scope 下启用 temporal；
- 只允许 background/低 motion token 跨帧合并；
- `temporal_window=1`；
- 不合并主体/手/吉他等高 motion token；
- 只在中间 denoising steps 开启，避开最早和最后阶段。

推荐配置：

```json
{
  "enabled": true,
  "ratio": 0.1,
  "mode": "spatiotemporal",
  "scope": "kv_only",
  "rope_mode": "pre_rope",
  "prop_attn": true,
  "match_feature": "attn_k",
  "layers": "middle",
  "temporal_window": 1,
  "protect_first_frame": true,
  "skip_early_ratio": 0.35,
  "skip_late_ratio": 0.15,
  "protect_topk_ratio": 0.25,
  "partition": "checkerboard_shifted"
}
```

### Phase F：引入 cache/broadcast 强基线

如果课程允许不局限于 token merge，建议做一个 training-free cache baseline：

- PAB-style attention output broadcast；
- TeaCache-style timestep-aware residual/output cache；
- 或直接调研集成 CacheDiT。

这可以作为报告中的 “strong acceleration baseline”。即使最终 token merge 质量一般，也能让项目整体更稳。

## 6. 推荐配置矩阵

第一轮只跑少量，避免再花一晚扫所有配置：

| Config | 目的 | 预期 |
|---|---|---|
| `baseline_no_merge` | 对照 | 原始质量 |
| `kv_spatial_r05_safe` | 最保守 KV-only | 画质接近 baseline，轻微加速 |
| `kv_spatial_r10_safe` | 主候选 | 画质可接受，3%-6% 加速 |
| `kv_spatial_r20_mid` | 速度候选 | 画质可能下降，6%-12% 加速 |
| `kv_spatial_r10_shifted` | 检验 shifted partition | 网格纹应减少 |
| `kv_st_r10_safe` | temporal 候选 | 检查 flicker/motion |
| `block_adaptive_r20` | 当前 naive block baseline | 速度强，但伪影重 |
| `block_fixed_r40` | 极限反例 | 展示速度/质量 trade-off |

## 7. 验收指标

必须同时看速度和画质，不再只看 `transformer_seconds`。

### 速度指标

- `inference_seconds`
- `transformer_seconds`
- `avg_step_seconds`
- `peak_gpu_memory_gib`

### 画质指标

最少人工检查：

- 主体是否变形；
- 背景是否出现网格纹；
- 吉他/手部是否拖影；
- 动作是否变僵；
- 是否 temporal flickering。

建议自动指标：

- CLIP score；
- frame-to-frame LPIPS 或 DINO cosine；
- optical-flow warping error；
- VBench 子集：`motion_smoothness`、`temporal_flickering`、`imaging_quality`、`dynamic_degree`。

通过门槛建议：

```text
端到端加速 >= 1.05x
Transformer 加速 >= 1.08x
无明显网格纹
主体/手/吉他无严重形变
dynamic_degree 不低于 baseline 的 90%
```

如果达不到，宁愿报告诚实结论：CogVideoX-2B 对 full-block token merging 更敏感，KV-only/caching 更适合。

## 8. 优先级结论

推荐路线：

1. **停止继续调当前 full-block merge 作为主线**。它适合作为反例，不适合作为最终展示。
2. **实现 `scope="kv_only"`**，保留全长 query/output，只合并 video K/V。
3. **先做 spatial KV-only，再做 temporal KV-only**。不要一上来跨帧合并主体区域。
4. **加入 shifted/local partition 和 importance protection**，解决网格纹和主体崩坏。
5. **把 PAB/TeaCache/CacheDiT 作为强基线或备选增强**，提高项目成功率。

一句话版本：

```text
当前方法证明了 naive block TokenMerge 可以加速，但质量不可接受；下一版应转向 KV-only attention merge + importance protection，并用 cache/broadcast 方法作为强基线。
```

## 9. 当前实现状态：ToMA-like v1

已落地第一版 ToMA-like 工程近似：

- 新增 `scope="kv_only"`：保留全长 Query/输出位置，只合并 video K/V。
- 新增 `partition="checkerboard_shifted"`：不同层交替棋盘格源/目标，降低固定网格纹。
- 新增 `reuse_interval`：每个 transformer block 每隔 N 次调用重新计算一次 matching，中间复用上一次 merge pattern，减少 adaptive matching 开销。
- 新增配置：
  - `configs/merge/toma_kv_spatial_r20_reuse4.json`
  - `configs/merge/toma_kv_spatial_r30_reuse4.json`
  - `configs/merge/toma_kv_st_r10_reuse4.json`

需要明确：这还不是 ToMA 论文的完整 attention-matrix merge/unmerge 复现；它是适配当前 CogVideoX/diffusers 代码结构的第一步，目标是先验证 pattern reuse + KV-only 是否能在不明显伤画质的情况下提高速度。
