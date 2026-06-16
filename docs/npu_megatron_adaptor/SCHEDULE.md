# CA_verl 交付倒排表（周 · 6/9 → 7/30 · 目标模型 Qwen3 / Qwen3.5 MoE）

> 锚点：7/30（周四）交付 ← 预留 10 个工作日社区审核（7/17–7/30）← **verl PR 最晚 7/17 提交**。起算 6/9。

| 周 | 日期(2026) | 主线 | 状态 |
|---|---|---|---|
| **W1** | 6/8–6/14 | 立项勘察 + 基础适配（megatron_adaptor backend / fp8 解锁 / mcore0.17 兼容）+ dense SFT & mxfp8 +12.6% MFU + A5 修复 + bshd RL | ✅ |
| **W2** | 6/15–6/21 | thd 适配修复 + .18 环境就绪 + dense thd 精度对照 | 🔵 进行中 |
| **W3** | 6/22–6/28 | dense 精度+性能收尾交付数据；**启动 MoE 适配**——GroupedLinear mxfp8 实测、token dispatch/permute、EP、MoE thd 路径排雷 | ⏳ |
| **W4** | 6/29–7/5 | **MoE 端到端打通**（Qwen3 MoE）+ 精度/性能验证，修规模下新 gap | ⏳ |
| **W5** | 7/6–7/12 | MoE 多卡（EP+TP）规模化验证 + 追加验证缓冲 + 整理交付物 + verl/TE-NPU PR 准备 | ⏳ |
| **W6** | 7/13–7/19 | **提交 verl 社区 PR（≤7/17）** + TE-NPU 剩余 PR；审核窗口开启 | 🔒 |
| **W7** | 7/20–7/26 | verl 社区审核 + 响应反馈 | 🔒 审核 |
| **W8** | 7/27–7/30 | 审核收尾 → 🎯 **7/30 交付** | 🎯 |

## 要点
- **范围**：dense（W1–W2）是地基，**MoE（W3–W5）是交付主体且基本未动**——工期最大不确定性。
- **缓冲**：W5 含追加验证；PR 比硬线 7/17 提前几天；MoE 顺利可更早提 PR、拉长审核窗口。
- **建议**：W3 一开始先做 MoE 最小冒烟探雷，尽早暴露隐藏 gap（类似 dense-thd 的两个根因）。

## 已交付/在途的修复
- TE-NPU **padding_causal**（thd 注意力因果掩码）：PR #93（Bruce-rl-hw → Ascend 上游）
- TE-NPU **mxfp8 columnwise**（thd backward wgrad）：fork 分支（暂不提 PR）
- verl：megatron_adaptor backend + fp8 解锁 + mcore0.17 兼容（待整理为社区 PR）
