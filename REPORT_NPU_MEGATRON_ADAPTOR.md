# verl × Ascend NPU:MegatronAdaptor + TransformerEngineNPU 适配报告

> 分支:`npu_megatron_adaptor` · 日期:2026-06-12 · 验证硬件:Ascend 950DT(CANN 9.1,torch_npu 2.8.0)

## 1. 目标

在 verl 中引入基于 **MegatronAdaptor**(Ascend 轻量 monkey-patch 层,~546 行)+ **TransformerEngineNPU**(drop-in 替换 `transformer_engine`)的 NPU 后端,替代重量级 MindSpeed 路线,实现:

1. **轻量敏捷跟进新版 megatron-core**(本次:0.17.2,MindSpeed 路线锁 0.12.1)
2. **低精度训练优先**:打通 mxfp8(`MXFP8BlockScaling`,E8M0 块缩放)
3. **对 verl 核心零侵入**:复用 `EngineRegistry` 插件机制

## 2. 软件栈

| 组件 | 版本 | 说明 |
|---|---|---|
| megatron-core | 0.17.2(源码 editable,`core_r0.17.0` 分支) | metadata 要求 py>=3.12,代码实测 py3.10 兼容(全量 compileall 零错误),`--ignore-requires-python` 安装 |
| transformer_engine | 2.13.0(TransformerEngineNPU,editable) | 纯 Python + torch_npu kernel,无 C++ 编译 |
| megatron_adaptor | 0.1.0(editable) | `import` 即生效 |
| verl | 0.9.0.dev(base commit 41a5244) | 本分支 |
| torch / torch_npu | 2.8.0 / 2.8.0.post4.dev | |
| mbridge / megatron-bridge | 0.15.1 / 0.3.1 | HF↔mcore 权重桥 |

## 3. 代码改动清单

### 新增
| 文件 | 说明 |
|---|---|
| `verl/workers/engine/megatron_adaptor/transformer_impl.py` | 薄壳引擎(~80 行):`backend="megatron_adaptor", device="npu"`,LM/Value 双注册,继承通用 `MegatronEngine*`。所有 patch 在 `import megatron_adaptor` 时一次完成,无 per-engine repatch |
| `verl/workers/engine/megatron_adaptor/__init__.py` | 导出 |
| `verl/trainer/config/engine/megatron_adaptor.yaml` | 引擎配置(megatron.yaml + `strategy: megatron_adaptor`) |
| `tests/workers/test_megatron_adaptor_registry_on_cpu.py` | 4 个 CPU 测试:registry 解析、不遮蔽 megatron backend、配置类接受/拒绝 strategy |

### 修改
| 文件 | 改动 | 原因 |
|---|---|---|
| `verl/workers/engine/__init__.py` | 文件顶部 try-import `megatron_adaptor`(`except Exception`) | `.fsdp` 会传递 import megatron,patch 必须抢先;torch_npu 在无 CANN 环境抛非 ImportError(UnicodeDecodeError),宽接防止炸掉非 NPU 路径 |
| `verl/workers/config/engine.py` | `McoreEngineConfig` strategy 断言放宽至 `["megatron", "megatron_adaptor"]` | 原断言在配置阶段直接堵死新 backend(审查发现的 P0) |
| `verl/utils/megatron_utils.py` | ① `config.fp8 = None` 条件化(仅未显式配置时清除);② 3 处 `ModelType.encoder_and_decoder` 加 `getattr` 守卫 | ① 解锁 fp8:用户经 `override_transformer_config` 显式开启的 fp8 得以保留,对未配置者零行为变化;② mcore 0.17 删除了该枚举成员 |

### 环境侧(未进仓,需部署时注意)
- **mbridge 0.15.1 的 `core/util.py` 2 处 `ModelType.encoder_and_decoder`** 同样需要 getattr 守卫(本验证中直改 site-packages;正式方案应向 mbridge 上游提 PR 或在 adaptor 中补 patch)
- `LD_PRELOAD=$CONDA_ENV/lib/libstdc++.so.6`:`datasets`(pyarrow)会先加载系统旧 libstdc++,导致 torch_npu 的 sqlite3 依赖(`libicui18n` 需 `CXXABI_1.3.15`)加载失败

## 4. 使用方式

```bash
torchrun ... -m verl.trainer.sft_trainer \
    engine=megatron_adaptor \
    optim=megatron \
    ... # bf16 默认
# 开启 mxfp8:
    +engine.override_transformer_config.fp8=hybrid \
    +engine.override_transformer_config.fp8_recipe=mxfp8
```

## 5. 验证结果(Qwen3-0.6B,GSM8K SFT,单卡 950DT,TP1/PP1/CP1)

### 5.1 冒烟(2 步训练 + 全量验证)

| 指标 | bf16 | mxfp8 | 偏差 |
|---|---|---|---|
| step 1 loss | 3.2595 | 3.5477 | +8.8% |
| step 2 loss | 1.5560 | 1.6399 | +5.4% |
| val loss | 0.7817 | 0.9076 | +16% |
| 峰值显存 | 15.1 GB | 16.4 GB | |

loss 差异证明 fp8 路径真实生效(非静默关闭);下降趋势一致。收敛性结论需 100+ 步曲线(待做)。

### 5.2 单元/集成验证
- MXFP8 GEMM 前向+反向微测:loss 有限、梯度无 NaN(`npu_dynamic_mx_quant_with_dual_axis` + `npu_quant_matmul` 实证)
- `megatron.core.fp8_utils.get_fp8_recipe` 确认被 adaptor patch 接管
- pytest 4 用例全过;registry 双引擎解析正确
- 性能参考:单步 ~1-2 s(batch 32,~7k token/步,MFU 0.04-0.09,远未打满)

### 5.3 算子覆盖
TE-NPU 全量引用 34 个 `torch_npu.npu_*` 算子,本机绑定 **0 缺失**。已实证:mxfp8 量化/GEMM、fusion attention、rms_norm、rope。未实测(阶段 2):MoE(`npu_grouped_matmul`、token permute/unpermute)、TP 通信融合(`npu_quant_mm_reduce_scatter`)。

## 6. 已知风险与待办

| 项 | 状态 |
|---|---|
| MegatronAdaptor 将 TE 版本伪装为 2.2.0(`patches/requirements/requirements.py`) | mxfp8 分支无版本 gate,本次不受影响;长期建议改为透传真实版本 |
| ref/rollout worker 经 `oc.select` 继承 actor 的 `override_transformer_config`,会跟随开启 fp8(logprob 噪声) | 阶段 2 决策:forward_only 引擎是否强制回退 bf16 |
| CP>1 时薄壳无 repatch 钩子(adaptor feature 按默认 args 选择) | 阶段 2 验证 CP 场景 |
| mbridge 上游未适配 mcore 0.17 | 已本地修;建议提 PR |
| 100+ 步收敛对比、多卡 TP/EP、MoE GroupedLinear mxfp8、vllm-ascend rollout | 阶段 2 |
