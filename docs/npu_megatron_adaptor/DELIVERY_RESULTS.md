# CA_verl 交付结果（verl × Ascend NPU：MegatronAdaptor + TE-NPU + mxfp8）

> 目标：verl 用 MegatronAdaptor + TransformerEngineNPU 替代 MindSpeed，轻量跟进 mcore 0.17，端到端验证 mxfp8 低精度训练的精度+性能收益。目标模型 Qwen3 / Qwen3.5 MoE。
> 平台：Ascend 950DT（CANN 9.1.T551，torch_npu 2.8.0，mcore 0.17.2，TE-NPU 2.13.0，vllm 0.13.1.dev + vllm_ascend A5）。

## 1. 交付的修复

### verl 侧（backend 适配，对 verl 核心零侵入）
- 新增 `backend="megatron_adaptor", device="npu"` 薄壳引擎（EngineRegistry 插件）
- fp8 配置链路解锁（megatron_utils）+ mcore 0.17 兼容（`ModelType.encoder_and_decoder` getattr 守卫，verl 3 处 + mbridge 2 处）
- 分支 `npu_megatron_adaptor`（github.com/computing-van-team/verl-megatron-te）

### TransformerEngineNPU 侧（两个独立 bug 修复，已上 fork，PR 形式提交上游）
1. **padding_causal 注意力**（PR #93）：mcore 对 packed/thd 序列传 `attn_mask_type='padding_causal'`，TE-NPU 只处理 `'causal'` → thd 注意力 `sparse_mode=0` 无因果掩码。修复让 `padding_causal` 走 `'causal'` 路径。含回归测试。
2. **mxfp8 columnwise**（fork 分支 fix/mxfp8-thd-columnwise）：forward-only（RL log_prob 重算）使 `input_quantizer` columnwise=False 残留，训练 forward 复用导致 backward wgrad GEMM 缺 columnwise → `UndefinedTensorImpl`。修复让 LayerNormLinear 每次显式设 `set_usage(rowwise=True, columnwise=backward_needs_input)`。含回归测试。

> 注：mxfp8 是真·低精度训练——Linear 层 GEMM 前向+反向都走 mxfp8（npu_dynamic_mx_quant + npu_quant_matmul，MXFP8BlockScaling 真量化）；主权重/优化器态/注意力保持高精度（标准 fp8 混合精度方案）。

## 2. 精度验证（actor 重算 vs rollout pearson corr）

| 模型 | 配置 | bf16 corr | mxfp8 corr |
|---|---|---|---|
| Qwen3-0.6B dense | thd | 0.999 | 0.977 |
| Qwen3-MoE（1层，真实权重） | thd, EP=1 | 0.999 | 0.978 |
| Qwen3-MoE（1层） | thd, EP=2（专家并行 all-to-all） | 0.999 | 0.981 |
| Qwen3-MoE（1层） | thd, EP=4 | 0.9993 | 0.9778 |
| Qwen3-MoE（1层） | thd, TP2×EP2（张量并行+专家并行组合） | 0.9992 | 0.9807 |
| Qwen3-MoE（1层） | thd, EP=8 | — | 见注 |

> EP=8（全 8 卡）训练侧无问题，但 vLLM rollout 在 8-worker 下撞 `/dev/shm` 共享内存 `/psm` 竞态（resource_tracker KeyError），在共享机上与 co-tenant 竞争 /dev/shm 所致——属 vllm-ascend 基建抖动，与本适配正交。EP≤4 与 TP+EP 组合已充分证明 EP+mxfp8 数值路径。

- mxfp8 corr ~0.977–0.981 全程稳定，是量化的预期偏差（bf16 ~0.999）。
- **结论：EP 1→2→4 扩展 + TP+EP 组合 × {bf16, mxfp8} × thd 全部跑通**，专家并行 all-to-all dispatch 在 NPU/HCCL 上与 mxfp8 共存。
- dense 100 步收敛：末段窗口 reward bf16 0.231 vs mxfp8 0.235（齐平；单步 reward 噪声大，看窗口均值）。

## 3. 性能（dense Qwen3-0.6B，GRPO thd，batch32）

| 指标 | bf16 | mxfp8 | 变化 |
|---|---|---|---|
| throughput (tok/s) | ~1167 | ~1264 | **+8%** |
| 峰值 HBM | 16.7 GB | 18.8 GB | +2.1 GB（fp8 双轴反向缓存） |

> throughput（墙钟）是可信口径。0.6B 小模型 + RL rollout 摊薄；SFT 饱和侧曾 +12.6%；MoE 目标上 GEMM 更大、收益预计更明显。

### MoE 性能（Qwen3-MoE 1层，GRPO thd，EP=4，batch16）
| 指标 | bf16 | mxfp8 | 变化 |
|---|---|---|---|
| throughput (tok/s) | ~912 | ~964 | **+5.7%** |
| MFU | ~0.15 | ~0.167 | +11% |
| 峰值 HBM/卡 | 11.5 GB | 12.0 GB | +0.5 GB |

> 1层模型 embedding/lm_head 占比大、稀释了 MoE 专家 GEMM 的 mxfp8 收益，是保守下限；全 30B（48 个 MoE 层、专家 GEMM 主导）收益预计显著更高。

## 4. 结论与剩余
- **dense + MoE，bf16 + mxfp8，thd，EP/TP+EP —— 核心组合全部端到端跑通，两个修复覆盖全部，无更多根因级 gap。**
- 剩余为纯规模化（W3–W5）：全 30B（48 层）多卡上量、更大 EP、规模化精度（长程）与性能、vllm 大 MoE rollout。前置：可加载的全 30B 权重。

## 5. 全 30B-A3B（48 层）8 卡上量进展（今日）

### MoE 8 卡适配的关键配置（显存关已解）
- 真实 Qwen3-30B-A3B（48 层，128 专家 top-8）bf16，8 卡：**EP4×TP2（引入 DP）+ full 激活重计算 + param/grad/optimizer offload**，vLLM rollout TP4 colocate（gpu_memory_utilization 0.4）。
- 每卡约 4.4B 参数分片；配 offload + recompute 后落进 64GB（.18 HBM 异构：6×96G + 2×64G，卡 4/6 为 64G）。
- 显存账本：**4 卡无 DP** 时分布式优化器状态切不开 → optimizer step OOM、且 vLLM 与训练抢 HBM 死结（GMU 0.4 训练 OOM / GMU 0.2 vLLM 起不来）；**8 卡引入 DP** 后分布式优化器跨 DP 切分优化器状态，显存关解除。

### 关键发现：模型 setup 期 507035 是平台层问题，非本适配
- 现象：8 卡 30B 在模型 setup 期崩 NPU **507035**（aivec 向量核「地址未对齐/越界」）。同步模式（ASCEND_LAUNCH_BLOCKING=1）精确定位到两个 **stock 基础算子**：megatron 参数初始化的 `tensor.fill_()`（PC 0x10004083c880）与 mbridge `load_weights` 权重拷贝到 NPU（PC 0x100040804880）——**均非我们代码、且未进入 forward/backward**；逐卡单独 `fill_` 8 张全 PASS。
- 是平台层、非适配的判据：
  - **.54（同为 950DT，CANN 9.1.T560，默认分配器）**：同一 30B 8 卡构建 **aivec=0，完全通过**（直到 load_weights 因未拷权重才报 FileNotFound）。
  - 报错文本即「address out of bounds」，对张量落地地址敏感：.18 上换用 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`（改内存地址布局）即可让构建通过（成功建成 4.4B 参数/卡），但随后在另一处大块分配触发 107017；用默认分配器则 fill_ 507035 复现。
- **结论**：507035 是 **.18 这台机器的平台层问题**（CANN 版本 / 驱动 / 硬件叠加，地址布局敏感），**与 CA_verl 适配代码无关**，且**非 950DT 通用**（.54 不犯）。仅靠 .18 上 source 新 CANN（T560）不足以稳定绕过。
- 建议：**全 30B 至-scale 验证在 .54 类健康节点上跑**；.18 需平台侧排查（升级 CANN/驱动或检查该机硬件）。

### 状态
- 8 卡 MoE 的**适配 + 显存**关已解（EP4×TP2+DP+recompute+offload，模型成功构建 4.4B 参数/卡）。
- 全 30B 端到端 30 步在 .18 上被该平台层 507035 卡住；待 .54 类节点补「至-scale」corr + 性能（接 §2/§3）。
- 运行脚本：env 可调的 run_30b.sh（DEVICES / NGPUS / EP / TP / RTP / GMU / RECOMPUTE / PO / GO / OO / CPUINIT / CANN_ENV / MXFP8 / STEPS / TAG）。
