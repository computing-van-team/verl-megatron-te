# 排查报告:GRPO `rollout_probs_diff` 偏大 — 定位与现状

> 现象:`use_remove_padding=True`(thd 打包)时 GRPO 的 `rollout_probs_diff_mean≈0.55`、`pearson_corr≈0.21`(probs 为 0~1 概率),即 vllm rollout 与 megatron actor 重算的 token 概率严重不一致,RL 训练不健康。

## 结论

**bug 在 thd(packed sequence)前向路径**;`use_remove_padding=False`(bshd)路径**完全正确**。我们的整套适配(megatron_adaptor + TE-NPU + verl backend)在 bshd 下 RL 精度无误。

**决定性对照(同模型同数据,2 步 GRPO):**

| 路径 | pearson_corr | rollout_probs_diff | actor ppl vs rollout ppl |
|---|---|---|---|
| **thd**(use_remove_padding=True) | 0.21 ❌ | 0.55 | 319 vs 1.5 |
| **bshd**(use_remove_padding=False) | **0.9993** ✅ | 0.0054 | 1.56 vs 1.56 |

## 排查路径(控制变量,逐个排除)

| 假设 | 实验(脚本) | 结果 | 判定 |
|---|---|---|---|
| vllm A5 ND cache patch 数值偏差 | vllm-patched vs HF fp32(`verify_ndpatch_logprob.py`) | corr 0.9989 | ❌ 排除 |
| vllm chunked_prefill/prefix_caching | 同上 + 开启 | corr 0.9989 | ❌ 排除 |
| 采样 temperature | 检查配置 | temp=1.0 raw | ❌ 排除 |
| 权重同步(megatron→vllm) | mbridge round-trip 逐层(`round_trip_weights.py`) | 310/311 全对 | ❌ 排除 |
| tie 模型 lm_head 未同步 | 补 lm_head 重跑 | corr 不变 | ❌ 证伪(已回退) |
| megatron 非packed forward 数值 | 简单 forward vs HF fp32(`actor_logprob_check.py`) | ppl 1.23, corr 0.99 | ✅ **正确** |
| **thd packed 前向路径** | thd vs bshd GRPO 对照 | thd 0.21 / bshd 0.999 | ✅ **定位** |
| thd 内 attention sparse_mode | TE-NPU backends.py 2→3 | corr 不变 | ❌ 证伪(已回退) |

诊断脚本见 `scripts/`。`actor_logprob_check.py` 证明 megatron 非 packed forward 正确(ppl 1.23 ≈ HF);bshd GRPO 证明 bshd 路径 RL 精度完美(corr 0.999)。

## thd 确切修复:待续(诚实记录)

bug 确在 thd packed 前向路径,但**确切修复点尚未找到**。已排除:权重、lm_head、attention sparse_mode。剩余嫌疑(待验证):thd packed 下 logits/label 的对齐、position/RoPE 在 packed 布局的应用、或 TE-NPU thd attention 的其他数值环节。`thd_repro.py` 单序列 thd 也偏离简单 forward(corr 0.43),但不完全等于 GRPO 的 319,复现仍需精化。

## 当前可用方案

- **bshd(`use_remove_padding=False`)**:RL 精度完全正确(corr 0.999),**立即可用于训练与精度验证**。代价:放弃 thd 的去 padding 吞吐优化。
- **mxfp8 在 bshd 下验证通过**:RL corr 0.9993(精度无损);训练侧 SFT 饱和负载 MFU +12.6%(见上层 README)。
- thd 满吞吐根治为后续工作项(不阻塞 dense 模型精度+性能收益的验证)。

## 复现脚本(scripts/)
- `verify_ndpatch_logprob.py` — vllm vs HF fp32(排除 patch/优化)
- `round_trip_weights.py` — mbridge 权重 round-trip(排除权重同步)
- `actor_logprob_check.py` — megatron 非packed forward vs HF(证明 forward 正确)
- `thd_repro.py` — thd vs 简单 forward 对照 + 对齐探针
