你现在接手的是一个视频生成推理加速课程大作业代码库。请你作为全栈研究工程 agent，尽可能一次性把后续工作完成到可提交状态：代码实现、实验脚本、benchmark、README、报告骨架、结果记录模板和必要的 smoke tests 都要做完。不要只给建议，要直接修改工程。

==============================
0. 项目背景与硬约束
==============================

这是一个“视频生成模型推理加速”课程大作业。已经提交过 proposal，题目是：

Research on Practical Inference Acceleration of Video Generation Models Based on Spatial Token Merging

proposal 的核心承诺：
- training-free；
- 基于开源视频生成模型，优先 CogVideoX；
- 做 spatial Token Merging / Token Reduction；
- 使用 bipartite matching 找相似 visual tokens；
- 保持整体模型结构和 block 输入输出形状兼容；
- 比较 baseline 与加速模型的 latency、average denoising step time、peak GPU memory、FVD/CLIP Score 或合理替代指标；
- 分析 merge ratio / layer insertion / quality-efficiency trade-off；
- 不能变成完全无关的 cache 或 distillation 项目。

当前代码库已经有一些前期 naive token merge 改动。请不要删除这些改动。你的任务是：
1. 审计当前代码；
2. 保留已有 naive Token Merge 作为 baseline；
3. 选择最合适的后续路线；
4. 在同一个工程里实现更合理、更能保持质量的 Token Merge / Token Reduction 方案；
5. 给出完整可复现实验脚本和报告材料。

最终路线必须仍然和 Token Merge 强相关。

==============================
1. 最终技术路线
==============================

请采用以下路线作为主线，不要随意换成 cache、TeaCache、DeepCache、LCM、DMD 或纯 sparse attention：

主模型：
- CogVideoX-2B 作为主实验模型。
- 原因：计算量相对可控；已有 proposal 提到 CogVideoX；当前工程可能已经基于 CogVideoX；AsymRnR 论文和开源代码都直接支持 CogVideoX-2B，并且直接比较了 ToMe 与 AsymRnR。

主方法：
- 实现一个 AsymRnR-style / RnR-ToMe / SA-RnR-ToMe 方法。
- 全称可以写成：
  Schedule-aware Asymmetric Token Reduction and Restoration for CogVideoX-2B
- 简称可以用：
  SA-RnR-ToMe 或 RnR-ToMe

核心思想：
- 不再只做 naive hidden-state spatial ToMe。
- 保留 naive ToMe 作为 proposal baseline / failure baseline。
- 新方法在 attention 内部做 asymmetric token reduction/restoration：
  - 先完成 q/k/v projection；
  - 对 visual tokens 的 Q 和/或 V/KV 做 reduction；
  - Q reduction 后必须 restore attention output 回原 visual sequence length；
  - K/V 可以共享 reduction scheme；
  - text tokens 不要 reduce；
  - block 外部输入输出 shape 必须保持不变。
- 加入：
  - timestep / block / feature-aware schedule；
  - Euclidean similarity matching；
  - matching cache；
  - replace/discard-style reduction，保留 mean reduction 作为 ablation；
  - 当前已有 naive ToMe 作为 baseline。

参考资料，必须阅读并在 docs/source_audit.md 记录你实际查到的内容：
- AsymRnR paper:
  https://arxiv.org/abs/2412.11706
  https://arxiv.org/html/2412.11706v2
- AsymRnR code:
  https://github.com/wenhao728/AsymRnR
- AsymRnR CogVideoX script:
  https://github.com/wenhao728/AsymRnR/blob/main/scripts/cogvideox/inference.sh
- Diffusers CogVideoX attention processor docs/source:
  https://huggingface.co/docs/diffusers/en/api/attnprocessor
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/cogvideox_transformer_3d.py
- CogVideoX official repo/model:
  https://github.com/zai-org/CogVideo
  https://huggingface.co/zai-org/CogVideoX-2b

如果当前工程已经 vendor 了某个 diffusers / CogVideoX 版本，以当前工程实际代码为准。不要假设最新版 API 一定一致。先读代码，再改代码。

==============================
2. 绝对禁止事项
==============================

- 不要伪造实验结果。
- 没有实际跑过的 latency、memory、CLIP、FVD、VBench、LPIPS 数字不要写成结果。
- 如果因为没有 GPU、模型权重、网络或依赖导致无法跑完整实验，请：
  1. 跑能跑的 smoke test；
  2. 生成完整实验脚本；
  3. 在 README 和 report 中明确标注哪些结果待运行；
  4. 不要编造数字。
- 不要删除已有 naive token merge 实现。要把它整理成一个可开关 baseline。
- 不要原封不动复制 AsymRnR 仓库。可以参考其思想和接口，但要在 README/report 里注明参考来源，并保留我们自己的简化实现和 ablation。
- 不要大改模型权重或训练模型。本项目必须 training-free。
- 不要把主线改成 TeaCache、DeepCache、TaylorCache、LCM、DMD 或 FastVideo。它们只能作为 related work，不是主实现。

==============================
3. 第一步：代码审计
==============================

开始修改前，请完成代码审计，并生成 docs/code_audit.md。

请执行并记录：
- git status
- 当前目录结构
- requirements / pyproject / setup 文件
- 当前使用的 diffusers、torch、transformers、accelerate 版本
- 当前是否已经有 CogVideoX pipeline
- 当前已有 token merge 相关代码位置
- 当前已有 CLI/script 入口
- 当前 naive token merge 插入点：
  - 是 block-level hidden_states merge？
  - 还是 attention-level merge？
  - 是否 scatter/restore？
  - 是否支持 merge ratio？
  - 是否只 merge visual tokens？
  - 是否破坏 text tokens？
  - 是否支持 fixed seed benchmark？
- 当前代码是否能跑 baseline inference 或 smoke test。

建议使用：
- rg -n "tome|merge|token|CogVideoX|AttnProcessor|attention|processor|hidden_states|rotary|RoPE|vae|benchmark|latency|memory" .
- python -c 打印关键包版本
- tree -L 3 或 find . -maxdepth 3

docs/code_audit.md 需要包括：
- “Existing naive ToMe implementation summary”
- “Current CogVideoX integration summary”
- “Risks in current implementation”
- “Modification plan”

==============================
4. 第二步：统一 CLI 和配置
==============================

请整理一个统一推理/benchmark 入口，名称根据当前工程结构决定，例如：

scripts/run_cogvideox_accel.py
或
run_cogvideox_accel.py

必须支持以下模式：

--accel none
--accel naive_tome
--accel kv_rnr
--accel qv_rnr
--accel rnr_tome

含义：
- none：原始 CogVideoX-2B baseline。
- naive_tome：保留当前已有 naive spatial Token Merge 实现。
- kv_rnr：只 reduce visual K/V，Q 保持 full length，输出天然 full length。
- qv_rnr：reduce visual Q，并 restore Q-output；同时对 V/KV 做较保守 reduction。
- rnr_tome：最终方法，包含 asymmetric reduction/restoration + schedule + matching cache。

必须支持参数：
- --model_path，默认可用 THUDM/CogVideoX-2b 或 zai-org/CogVideoX-2b，根据当前工程兼容性选择。
- --prompt 或 --prompt_file
- --output_dir
- --seed
- --num_inference_steps，默认 50，smoke 可用更小。
- --num_frames，默认 49。
- --height，默认 480。
- --width，默认 720。
- --guidance_scale，默认 6.0。
- --dtype fp16/bf16/fp32。
- --enable_cpu_offload。
- --log_latency。
- --log_memory。
- --save_video。
- --accel none/naive_tome/kv_rnr/qv_rnr/rnr_tome。
- --merge_ratio，用于 naive_tome。
- --q_reduce_ratio。
- --kv_reduce_ratio 或 --v_reduce_ratio。
- --similarity_type cosine/euclidean/dot/random。
- --reduce_mode replace/mean。
- --dst_stride，例如 2 2 2。
- --matching_cache_steps，默认 5。
- --disable_schedule。
- --schedule_config configs/rnr_cogvideox2b_default.yaml。
- --benchmark_csv results/benchmark.csv。

如果现有脚本命名不同，请不要强行重构到不可用；在最小破坏基础上增加这些参数。

==============================
5. 第三步：保留并整理 naive ToMe baseline
==============================

请把当前已有前期 token merge 改动整理为可运行 baseline：

--accel naive_tome

要求：
- 不能破坏原始 baseline。
- 能通过 merge_ratio 控制 0.1 / 0.2 / 0.3。
- 记录其插入层、插入 step、merge target。
- 如果当前实现是 hidden-state merge + scatter，要在 README/report 里明确写成 naive frame-wise hidden-state ToMe。
- 如果已有实现质量不好，不要试图掩盖；保留作为 failure baseline。
- 记录 artifacts：
  - blur
  - pixelation
  - flicker
  - structure distortion
  - motion instability
- naive_tome 的结果要和新方法对比。

==============================
6. 第四步：实现 AsymRnR-style 核心模块
==============================

新增一个清晰的模块目录，按当前工程风格决定，例如：

src/accel/rnr/
或
video_accel/rnr/
或
tome/rnr/

推荐文件：
- rnr_config.py
- matching.py
- partition.py
- reduce_restore.py
- scheduler.py
- cogvideox_processor.py
- apply.py
- metrics.py

6.1 matching.py

实现：
- compute_similarity(dst, src, similarity_type)
  - cosine
  - dot
  - euclidean
  - random
- match_features(dst, src, similarity_type)
  - 对 dot/cosine：越大越相似；
  - 对 euclidean：越小越相似；
  - 返回 matched destination index、source ranking、similarity value。
- 优先用 torch 实现，保证可运行。
- 如果项目环境允许，可以可选支持 optimized square_dist extension，但不能强依赖；extension 不可用时 fallback 到 torch.cdist 或手写 squared distance。
- 不要让 extension 编译失败导致整个项目不可运行。

6.2 partition.py

实现 spatiotemporal source/destination partition：
- 输入 visual token layout：
  - latent_frames
  - latent_height
  - latent_width
- 对 CogVideoX-2B 默认：
  - num_frames=49 时 latent_frames 通常是 (num_frames - 1) // 4 + 1；
  - latent_height = height // 16；
  - latent_width = width // 16；
  - 但请以当前 CogVideoX/Diffusers 实际代码为准，必须在 code_audit 里验证。
- 支持 dst_stride，例如 [2,2,2]。
- 只处理 visual tokens，不处理 text tokens。
- 如果当前 attention sequence 中 text tokens 在前，visual tokens 在后，需要正确 split/concat。
- 如果当前工程中顺序相反，也要自动适配或显式配置 encoder_first。

6.3 reduce_restore.py

实现：
- reduce_sequence(...)
- restore_sequence(...)

需要支持：
- replace/discard-style reduction：推荐主方法，速度好，质量往往比 mean 更稳。
- mean reduction：作为 ablation，对应 naive ToMe 思路。
- restore Q-output 回原 visual sequence length。
- K/V reduction 不需要 restore，但 K 和 V 必须共享同一 reduction scheme。
- 所有 tensor shape 都要有 assert 和注释。
- 不要破坏 batch/head 维度。
- 必须支持 classifier-free guidance 下 batch size=2 的情况。

6.4 scheduler.py

实现 schedule：
- feature-aware：Q 和 V/KV 可以不同 reduce ratio。
- block-aware：不同 transformer block 可以不同。
- timestep-aware：不同 denoising step 可以不同。
- 支持从 YAML 加载。
- 支持 conservative fallback schedule。

默认 configs/rnr_cogvideox2b_default.yaml 可以这样设计：

method: rnr_tome
model: cogvideox-2b
encoder_first: true
similarity_type: euclidean
reduce_mode: replace
dst_stride: [2, 2, 2]
matching_cache_steps: 5
schedule:
  q:
    enabled: true
    thresholds:
      - similarity: 0.60
        ratio: 0.40
      - similarity: 0.70
        ratio: 0.80
  v:
    enabled: true
    thresholds:
      - similarity: 0.80
        ratio: 0.30
  k:
    share_with_v: true
block_skip:
  first_n: 0
  last_n: 0
step_skip:
  first_frac: 0.0
  last_frac: 0.0

如果没有预先估计 similarity distribution，可以先实现手动 schedule：
- 中间 steps 开启 reduction；
- 早期/末期可选关闭；
- 先用固定 q_reduce_ratio / v_reduce_ratio；
- 后续再加 similarity-threshold schedule。
但 README/report 里要解释这是 AsymRnR-lite schedule。

6.5 matching cache

实现：
- 每个 feature、block、timestep 或 block/feature 组合缓存 matching pairs。
- --matching_cache_steps 控制复用多少步。
- 默认 5。
- cache 只缓存 matching indices，不缓存 hidden features 或 block output。
- 记录 cache hit rate。
- 每个 prompt 开始时必须 reset cache。
- 每次 denoising step 更新 timestep。
- 不要跨 prompt 污染 cache。

6.6 cogvideox_processor.py

核心：实现 CogVideoX attention processor patch。

如果当前工程使用 diffusers：
- 参考 CogVideoXAttnProcessor2_0。
- 写一个 CogVideoXRnRAttnProcessor2_0 或项目内等价类。
- 在 apply.py 中提供 apply_rnr_to_cogvideox(transformer, config)。
- 不要直接 monkey patch 得难以维护；优先替换 attention processor 或 wrapper attention module。

attention 逻辑：
1. 保持原 CogVideoX attention 的输入输出 signature。
2. 先完成原始 q/k/v projection。
3. 对 q/k 应用 CogVideoX 原有 rotary embedding / image_rotary_emb。
4. split text tokens 和 visual tokens。
5. 对 visual Q 做 reduction，记录 restore function。
6. 对 visual K/V 做共享 reduction，通常用 V 的 matching 决定 K/V reduction。
7. concat text + reduced visual tokens。
8. 调用 torch.nn.functional.scaled_dot_product_attention 或原工程 attention 实现。
9. 对 Q-output restore 到原 visual sequence length。
10. 接回原 output projection、dropout、residual 后续路径。
11. 输出 shape 必须和原 processor 完全一致。

重要：
- Q reduction 后如果不 restore，后续 block 会坏。
- K/V reduction 不 restore，因为它们只参与当前 attention。
- text tokens 永远不 reduce。
- CFG batch=2 时 shape 要正确。
- 所有新增逻辑应该能通过 --accel none 完全关闭。

==============================
7. 第五步：benchmark 和实验脚本
==============================

新增实验脚本：

scripts/benchmark_cogvideox_token_merge.sh
scripts/smoke_test.sh
scripts/run_ablation.sh

7.1 smoke_test.sh

目标：快速验证代码不崩。
- 1 个 prompt。
- 固定 seed。
- 如果默认 49 frames / 480x720 太慢或 OOM，可提供小配置，但要在 README 中说明小配置只是 smoke，不作为正式结果。
- 运行：
  - --accel none
  - --accel naive_tome
  - --accel kv_rnr
  - --accel rnr_tome
- 确认输出视频或 latent。
- 确认 shape assertions 通过。
- 确认 benchmark csv 写入。

7.2 benchmark_cogvideox_token_merge.sh

正式 benchmark：
- 使用 CogVideoX-2B。
- 默认 49 frames，480x720，50 steps。
- prompts 文件：configs/prompts_benchmark.txt。
- 至少包含 8 个 prompts；如果时间允许，扩展到 24 或 40。
- prompt 分类：
  - low-motion background；
  - high-motion subject；
  - camera motion；
  - fine details / human / animal；
  - multi-object interaction。
- 每个方法固定 seed。
- 输出：
  - videos/
  - logs/
  - benchmark.csv
  - summary.md

至少跑以下方法：
- none
- naive_tome ratio=0.1
- naive_tome ratio=0.2
- naive_tome ratio=0.3
- kv_rnr conservative
- qv_rnr conservative
- rnr_tome default
- rnr_tome fast，可选

记录指标：
- end-to-end latency；
- denoising-only latency，如果能分离；
- average denoising step time；
- peak GPU memory allocated；
- peak GPU memory reserved；
- token reduction ratio；
- matching time；
- attention time，如果容易实现；
- cache hit rate；
- output video path；
- seed；
- prompt；
- method config hash。

7.3 run_ablation.sh

至少设计以下 ablation：
- similarity_type: cosine vs euclidean；
- reduce_mode: mean vs replace；
- matching_cache_steps: 1 vs 5；
- Q-only / KV-only / Q+V；
- naive_tome ratios 0.1/0.2/0.3；
- schedule on/off。

如果没有时间全跑，脚本必须存在，README 中说明建议运行顺序。

==============================
8. 第六步：质量评估
==============================

请实现或提供脚本：

scripts/evaluate_quality.py

优先级：
1. 如果能安装并运行 VBench subset，则支持 VBench subset。
2. 如果 VBench 太重，则实现轻量替代：
   - CLIP text-video alignment：可用抽帧 CLIPScore；
   - LPIPS vs baseline generated video：同 prompt/seed 下 baseline 与 accelerated 的 frame-level distance；
   - SSIM/PSNR 可选；
   - temporal consistency proxy：相邻帧 optical-flow-free 差分或 CLIP image feature smoothness；
   - human preference sheet：生成 pairwise 对比 HTML/Markdown。
3. FVD 如果没有真实视频数据集或样本太少，不要强行伪造。可以保留脚本接口，但报告里解释小样本 FVD 不稳定。

输出：
- quality_metrics.csv
- pairwise_review.md
- qualitative_grid.html 或 qualitative_grid.md
- 每个 prompt 的 baseline vs accelerated 视频路径。

==============================
9. 第七步：README
==============================

请重写或补充 README.md，使其达到课程提交标准。

README 必须包含：
- 项目标题：
  Schedule-aware Asymmetric Token Merging for Efficient CogVideoX Inference
- 简短摘要；
- 和 proposal 的关系：
  - 我们保留 naive frame-wise spatial ToMe；
  - 发现其质量/速度 trade-off 不理想；
  - 因此实现 AsymRnR-style attention-level token reduction/restoration；
  - 仍然是 training-free Token Merging / Token Reduction 方向。
- 方法说明：
  - baseline；
  - naive ToMe；
  - KV-only RnR；
  - Q/V asymmetric RnR；
  - schedule；
  - matching cache。
- 安装环境；
- 模型权重下载说明；
- smoke test 命令；
- benchmark 命令；
- evaluation 命令；
- 结果复现说明；
- 代码结构；
- 如何添加新 prompt；
- 如何修改 config；
- 已知问题；
- 引用和参考开源项目；
- 明确声明哪些代码是我们实现的，哪些参考了 AsymRnR/Diffusers/CogVideoX。

README 里不能写没有实际跑出的数字。可以有 “Expected / target” 和 “Observed” 两栏，Observed 只能填真实结果。

==============================
10. 第八步：报告材料
==============================

课程要求英文 NeurIPS 风格报告。请生成：

report/
  main.tex
  references.bib
  figures/
  tables/

报告标题：
Schedule-aware Asymmetric Token Merging for Efficient Video Diffusion Transformers

或：
Revisiting Token Merging for CogVideoX: Asymmetric Reduction and Restoration for Training-free Video Generation Acceleration

报告结构：
1. Abstract
2. Introduction
3. Related Work
   - Video diffusion transformers
   - Token Merging / ToMe
   - Diffusion acceleration
   - AsymRnR and token reduction
4. Method
   - Naive frame-wise spatial ToMe baseline
   - Problems with hidden-state ToMe in video DiT
   - Asymmetric Q/V or Q/K/V reduction-restoration
   - Spatiotemporal bipartite matching
   - Euclidean matching and replace reduction
   - Timestep/block/feature schedule
   - Matching cache
5. Experiments
   - Model and setting: CogVideoX-2B
   - Prompts
   - Metrics
   - Baselines
   - Efficiency results
   - Quality results
   - Ablations
6. Qualitative Analysis
   - blur / pixelation / flicker / motion artifacts
7. Limitations
8. Conclusion
9. Contribution Statement

报告里必须有以下表格模板：
- Table 1: Main latency-quality comparison
- Table 2: Ablation on similarity/reduction/cache
- Table 3: Prompt-category breakdown
- Table 4: Memory and token statistics

如果真实实验已经跑完，把结果自动填入表格。
如果未跑完，表格保留 TODO 或 “not run yet”，不要编造。

Contribution Statement 必须写清楚：
- existing naive ToMe code was inherited from preliminary project work；
- this implementation adds attention-level asymmetric RnR, scheduling, matching cache, benchmark scripts, evaluation scripts, and report structure；
- AsymRnR/Diffusers/CogVideoX were used as references, not copied wholesale。

==============================
11. 第九步：结果整理
==============================

请生成：
results/
  README.md
  benchmark.csv
  quality_metrics.csv
  summary.md
  videos/
  logs/

summary.md 至少包括：
- Hardware；
- Software versions；
- Model；
- Prompt count；
- Methods；
- Main table；
- Observed artifacts；
- Best recommended config；
- Failed configs；
- Notes on reproducibility。

如果无法实际运行：
- results/summary.md 说明未运行原因；
- 仍然保证 scripts 可复现；
- README 中清楚写明如何运行。

==============================
12. 验收标准
==============================

你完成任务前必须检查：

代码层面：
- --accel none 可运行，不受新代码影响。
- --accel naive_tome 可运行，保留已有工作。
- --accel kv_rnr 至少 shape 正确。
- --accel rnr_tome 至少 smoke test 正确。
- 所有新模块有清楚注释。
- 所有 tensor shape 有必要 assert。
- 没有无用的大规模复制代码。
- 没有硬编码只适用于某一个 prompt 的逻辑。

实验层面：
- benchmark 脚本能一键跑。
- 结果 csv 字段完整。
- latency 使用 torch.cuda.Event 或可靠计时。
- memory 使用 torch.cuda.max_memory_allocated / reserved。
- fixed seed。
- 第一个样本 warmup 可选，但必须记录是否 skip warmup。

报告层面：
- NeurIPS-style 英文报告骨架存在。
- README 完整。
- source_audit/code_audit 存在。
- 引用完整。
- 不伪造结果。
- 明确和 proposal 的关系。
- 明确与 AsymRnR、ToMe、CogVideoX、Diffusers 的区别和联系。

==============================
13. 推荐默认实验配置
==============================

优先实现和测试：

Baseline:
python scripts/run_cogvideox_accel.py \
  --accel none \
  --model_path THUDM/CogVideoX-2b \
  --prompt_file configs/prompts_benchmark.txt \
  --output_dir results/baseline \
  --seed 42 \
  --num_inference_steps 50 \
  --num_frames 49 \
  --height 480 \
  --width 720 \
  --log_latency \
  --log_memory \
  --save_video

Naive ToMe:
python scripts/run_cogvideox_accel.py \
  --accel naive_tome \
  --merge_ratio 0.2 \
  --model_path THUDM/CogVideoX-2b \
  --prompt_file configs/prompts_benchmark.txt \
  --output_dir results/naive_tome_r02 \
  --seed 42 \
  --num_inference_steps 50 \
  --num_frames 49 \
  --height 480 \
  --width 720 \
  --log_latency \
  --log_memory \
  --save_video

Final RnR-ToMe:
python scripts/run_cogvideox_accel.py \
  --accel rnr_tome \
  --model_path THUDM/CogVideoX-2b \
  --prompt_file configs/prompts_benchmark.txt \
  --output_dir results/rnr_tome_default \
  --seed 42 \
  --num_inference_steps 50 \
  --num_frames 49 \
  --height 480 \
  --width 720 \
  --similarity_type euclidean \
  --reduce_mode replace \
  --dst_stride 2 2 2 \
  --matching_cache_steps 5 \
  --schedule_config configs/rnr_cogvideox2b_default.yaml \
  --log_latency \
  --log_memory \
  --save_video

如果当前工程使用 zai-org/CogVideoX-2b 而不是 THUDM/CogVideoX-2b，请自动适配并记录。

==============================
14. 最终输出
==============================

完成后，请在终端最后输出一份简短总结：

- 修改了哪些文件；
- 新增了哪些文件；
- 如何运行 smoke test；
- 如何运行 benchmark；
- 当前是否已实际生成结果；
- 如果有真实结果，给出 main result table；
- 如果没有真实结果，说明卡在哪里；
- 下一步人工只需要做什么。

请现在开始执行：先做代码审计，再实现，不要只写计划。