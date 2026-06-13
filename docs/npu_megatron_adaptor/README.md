# verl × Ascend NPU(950DT):MegatronAdaptor + TransformerEngineNPU + vLLM RL 闭环

本目录固化在 Ascend 950DT 上用 **MegatronAdaptor + TransformerEngineNPU** 跑通 verl 训练与 GRPO RL 全闭环的全部成果、补丁与复现脚本。

> 硬件:Ascend 950DT(soc_version=260,vllm-ascend 识别为 A5 设备型) · CANN 9.1.T551 · torch_npu 2.8.0 · mcore 0.17.2 · TE-NPU 2.13.0 · verl 0.9.0.dev

## 总进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 后端 | `megatron_adaptor` 薄壳引擎注册进 verl EngineRegistry | ✅ |
| 兼容 | mcore 0.17 兼容修复(`ModelType.encoder_and_decoder` 移除) | ✅ |
| 训练 | SFT 单卡 bf16 + mxfp8 冒烟(Qwen3-0.6B) | ✅ |
| 低精度 | mxfp8 vs bf16:吞吐 **MFU +12.6%**、收敛齐平(val 差 0.8%)、显存 +1.7G | ✅ |
| 多卡 | TP2 + sequence-parallel,bf16/mxfp8 全通,显存按切分减半 | ✅ |
| Rollout | vllm-ascend A5 GQA cache bug 修复,Qwen3-0.6B 推理输出正确 | ✅ |
| RL 闭环 | GRPO 全链路工程打通(actor + vllm rollout + 权重同步 + ref) | ✅ 工程通 |
| 质量 | reward 信号、rollout_probs_diff 数值一致性 | ⏳ 待查 |

## 核心代码改动(已在 commit 2566c33 / 本 commit)

### verl 仓内
- `verl/workers/engine/megatron_adaptor/`:薄壳引擎,`backend="megatron_adaptor", device="npu"`,LM/Value 双注册,`import megatron_adaptor` 一次性打 patch。
- `verl/workers/engine/__init__.py`:文件顶部 `import megatron_adaptor`(须先于 `.fsdp` 传递 import megatron),`except Exception`(torch_npu 无 CANN 时抛非 ImportError)。
- `verl/workers/config/engine.py`:`McoreEngineConfig` strategy 断言放宽收 `megatron_adaptor`。
- `verl/utils/megatron_utils.py`:① `config.fp8` 条件化(保留 `override_transformer_config` 显式 fp8);② 3 处 `ModelType.encoder_and_decoder` 加 `getattr` 守卫(mcore 0.17 移除了该枚举)。
- `verl/trainer/config/engine/megatron_adaptor.yaml`、`tests/workers/test_megatron_adaptor_registry_on_cpu.py`(4 用例)。

### 环境侧(不在 verl 仓,部署时需手动应用)
- **`vllm_ascend_a5_gqa_nd.patch`**:修 vllm-ascend 3 月内部 A5 构建的 GQA KV cache bug。根因:写路径用 `npu_scatter_pa_kv_cache`(要 NZ 布局),读路径 `_get_fia_params` 用 ND view,二者矛盾。修法:强制 A5 GQA 走标准 ND `_npu_reshape_and_cache`(与 ND 读路径自洽)。应用:`patch -p0 < vllm_ascend_a5_gqa_nd.patch` 于 site-packages(改 `attention/attention_v1.py`)。**局限**:只修 GQA;Qwen3.5(`qwen3_next` 混合架构)的 linear attention 层仍需完整内部构建。
- **mbridge**(site-packages `mbridge/core/util.py`):2 处 `encoder_and_decoder` 同样加 `getattr` 守卫(mcore 0.17)。
- **运行时环境**:
  - `source /usr/local/Ascend/nnal/atb/set_env.sh`(vllm rollout 需 libatb.so)
  - `export LD_PRELOAD=$CONDA_ENV/lib/libstdc++.so.6`(`datasets`/pyarrow 会先载系统旧 libstdc++,致 torch_npu sqlite3 依赖的 `CXXABI_1.3.15` 加载失败)
  - 启动脚本勿用 `set -u`(华为 vendor `set_env.sh` 引用未定义变量 CMAKE_PREFIX_PATH/ZSH_VERSION)

## 实测数据

### mxfp8 vs bf16(Qwen3-0.6B,饱和负载 batch256/max_token8192,单卡)
| 指标 | bf16 | mxfp8 |
|---|---|---|
| 平均 MFU | 0.2301 | 0.2592(**+12.6%**) |
| val loss | — | 收敛齐平,差 0.8% |
| 峰值显存 | 32.4G | 34.1G(+1.7G,双轴 fp8 反向缓存) |

mxfp8 为真量化:`MXFP8MatMul` 走 `npu_dynamic_mx_quant_with_dual_axis` + `npu_quant_matmul`,fp8 进 fp8 算,前反向全覆盖(非伪量化)。`fp8_param=True`(权重 fp8 存储)被 TE-NPU `QuantizedTensor` 不支持 `Module._apply` 设备搬运阻塞,列为 TE-NPU 工作项。

### GRPO 全闭环(2 步,2 卡,vllm rollout)
全链路 timing 均有效执行:`gen`(vllm) / `update_weights`(megatron→vllm resharding 2.4s) / `ref` / `old_log_prob` / `update_actor`(megatron_adaptor)。EXIT=0。
**两个待查质量问题**:
1. reward 全 0 → actor 梯度 0、未真更新(GSM8K reward_fn 未匹配,response 打满 256 被截;配置/小模型问题,非适配层)。
2. `rollout_probs_diff_mean:0.55` 偏大,vllm 与 actor 概率对不齐,**需验证是否 ND patch 引入的数值偏差**。

## 复现

环境(conda env `b84_verl_017`)+ 三仓 editable(mcore 0.17.2 / TE-NPU / megatron_adaptor)+ verl `pip -e`。详见 `../../REPORT_NPU_MEGATRON_ADAPTOR.md`。

```bash
# SFT 训练冒烟(bf16 / 加 fp8 参数即 mxfp8)
bash scripts/run_smoke_sft.sh 0
bash scripts/run_smoke_sft.sh 0 +engine.override_transformer_config.fp8=hybrid +engine.override_transformer_config.fp8_recipe=mxfp8

# vllm-ascend 推理冒烟(需先 patch + source NNAL)
python scripts/test_vllm_ascend.py

# GRPO 2 步 RL 闭环
NPUS=0,1 bash scripts/run_grpo_smoke.sh
```

## 待办
- [ ] 修 GSM8K reward_fn,让 reward 有信号,验证健康收敛
- [ ] 验证 `rollout_probs_diff`(patch 前后对比),确认 ND patch 数值安全
- [ ] `fp8_param=True` 权重 fp8 存储(TE-NPU QuantizedTensor 设备搬运修复)
- [ ] Qwen3.5(qwen3_next 混合架构):linear attention 层需完整内部 vllm-ascend A5 构建
