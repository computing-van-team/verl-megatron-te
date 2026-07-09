# Multi-node mxfp8 (Option B) — Progress & Reference

## Goal

CA_verl: adapt verl to Ascend 950DT NPU via **MegatronAdaptor + TransformerEngineNPU (TE-NPU) + megatron-core 0.17**, for **mxfp8 low-precision RL post-training** on **Qwen3-30B-A3B**.

- Model weights: **bf16 base**. mxfp8 is the compute/comm precision in **training** (MXFP8BlockScaling), not a quantized checkpoint.
- Rollout: **bf16 inference** (we do NOT ship a quantized model).
- Endgame: **2-node / 16-card** to escape the single-node 8-card mxfp8 memory wall.

## What works (validated)

- Single-node **8-card full Qwen3-30B-A3B GRPO, bf16, EP4×TP2** — end-to-end, logprob corr **0.971** (on 141.62.17.180).
- mxfp8 mine-sweeping passed: dense + MoE × bf16 + mxfp8 × thd.
- GroupedTensor mxfp8 fix (TransformerEngineNPU) pushed as PR.

## Single-node memory wall (why 2-node)

8-card mxfp8 step2: ~74G train + ~17.6G vLLM ≈ 91.6G > ~96G HBM. Root = megatron optimizer persistent buckets. Config levers exhausted → pursue 2-node/16-card (training spread over 16 cards lowers per-card residency).

## Two stacks

- **Old stack** `CA_verl` — torch_npu 2.8 / CANN 9.1.T551. Colocate works single-node. vllm-ascend 0.1.dev (Mar 2026); 2-node MoE rollout **assumed** unsupported (NOT source-verified — see Open Questions).
- **New stack "Option B"** `CA_verl_new` — torch_npu 2.10 / vllm 0.20.3.dev0 / vllm-ascend 0.19.1rc2 / CANN 9.1.T500. Cloned from q00887491's `verl_vllm020` env + layered our editables (megatron-core 0.17.2, TE-NPU 2.13, megatron_adaptor, verl_new). vllm-ascend 0.19 **has** multi-node MoE rollout.

## Option B single-node smoke — breakthrough chain

Launcher: `/home/CA_verl/launch_new_1node_envfull.sh` (replicates q's working single-node env + our params). Blockers solved in order:

1. **507033 / "context can not be null"** @ `vLLMHttpServer.launch_server`. Root: rollout DP>1 (RTP=4 → 2 replicas) under `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` → both replicas' rank-0 worker claim global device 0. **Fix: single replica `RTP=8` (1:1 with the 8 training cards).** Minimal tests ruled out driver device-sharing limits, megatron_adaptor, and thread-context; `replica.py`/`vllm_async_server.py` are byte-identical to q's; our editables do NOT load in the rollout process.

2. **uniform_ EZ1008** @ dummy weight init. `param.uniform_()` → `aclnnInplaceUniform` / `StatelessRandomUniformV2` tiling **fails on arch35 (950)** — plog `random_tiling_arch35.cpp CheckTensor`, `Execution_Error(EZ1008)`.
   - Dead-end: `rollout.quantization=ascend` cleared it (quantized dummy init skips the float `uniform_`) but that forces **quantized rollout** and requires a pre-quantized ModelSlim config — **contradicts our bf16 target**. Dropped.
   - **Correct fix (TODO): keep bf16 rollout; monkey-patch vLLM `initialize_dummy_weights` to avoid the failing RNG op** (dummy values are immediately overwritten by the training→rollout weight sync in colocate; `.normal_()` / `.zero_()` / constant init suffices).

## Open questions

- **torch2.8 multi-node NOT definitively confirmed incapable.** 2-node rollout on the old stack failed with HCCL errors (`EI0006` socket timeout, `HcclAllocComResource`, vector-core timeout). These are **comms-layer** errors that may be **HCCL config** (IFNAME / ports / timeout / buffer), not a hard vllm-ascend capability gap. Source-verify old vllm-ascend for multi-node MoE rollout before fully committing to Option B.

## Machine status (2026-07-09, 141.62.17.x A5 fleet)

| Node | State | Notes |
|------|-------|-------|
| .188 | BUSY | user `s00574452` deepseek pretrain, 8/8 cards. Has `CA_verl_new` ready. |
| .114 | FREE | card0 sensor `Alarm` (80E3A207) but **compute verified OK**. Has torch2.10 base `verl_vllm020` + sources; missing `verl_new`/`mbridge`/assembled `CA_verl_new`. |
| .180 | FREE | no HW alarm, validated 8-card machine. Only old torch2.8 `CA_verl` env (no torch2.10 base). |

## Reference: q00887491's multi-node (2-node) scripts & assets

Read-only, on the 141.62.17.x shared paths; **do not disturb their running jobs**.

- 2-node **mxfp8** train script: `/mnt/share/q00887491/logs/20260629_2145/qcs-30b_A5_8_end_magatron_2nodes_mxfp8_prec.sh`
- 2-node **bf16** train script: `/mnt/share/q00887491/logs/20260707_2058/30b_A5_8_end_magatron_2nodes_bf16_prec.sh`
- ray-start scripts (per-run): `qcs-rays-start.sh` inside each `logs/<timestamp>/`; also `ray_start_mxfp8.sh`, `ray_start_bf16.sh`
- **working single-node bf16** reference dir: `/mnt/share/q00887491/logs/20260629_1655_bf16_1n_ok/`
- **working 2-node run log**: `/mnt/share/q00887491/logs/20260707_2058/run_20260707_2058.log`
- q's verl (A5-0611): `/mnt/share/q00887491/projects/A5-0611/verl`
- q's env: conda env `verl_vllm020` (present on .188 / .114)
- CANN for new stack: `/mnt/share/l00606955/9.1.T500.B010/cann-9.1.T500/set_env.sh`
- pre-quantized models (for quantized rollout — **not needed for our bf16 goal**): `/mnt/share/q00887491/models/Qwen3-30B-MoE-MXFP8`, `/mnt/share/z00804445/models/Qwen3-30B-MoE-MXFP8`

### q's key 2-node config (mxfp8)

- **NNODES=2**; train `TP1 PP2 EP8 ETP1` VPP=null DP8; `all_offload=True` + `optimizer_offload_fraction=1`; `fp8=e4m3` + `fp8_recipe=mxfp8` + `fp8_reuse_quantized_weight=True`; dist_checkpointing = `Qwen3-30B-A3B-Base-dist`.
- **Rollout**: `gen_tp=2 gen_dp=1 gen_ep=2`, `quantization=ascend` (their choice — quantized rollout), `enforce_eager=False`, `free_cache_engine=True`, `gpu_mem=0.65`, `max_num_batched_tokens=8192`, `max_num_seqs=512`.
- **Env**: `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`, `VLLM_ASCEND_ENABLE_NZ=0`, `TASK_QUEUE_ENABLE=2`, `CPU_AFFINITY_CONF=1`, `LCAL_COMM_ID=127.0.0.1:27001`, `MULTI_STREAM_MEMORY_REUSE=1`, full HCCL socket settings (`HCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` on the 141.6 iface, `HCCL_*_SOCKET_PORT_RANGE=auto`, `HCCL_CONNECT_TIMEOUT/EXEC_TIMEOUT=3000`, `HCCL_BUFFSIZE=200`).

## Next steps

1. Validate the **bf16 `initialize_dummy_weights` patch** (minimal op test: which init op works on arch35) — needs torch2.10 base (available on .114).
2. Decide **old-stack-2-node vs Option B** by source-verifying old vllm-ascend multi-node MoE rollout.
3. Finish single-node Option B smoke with the bf16-rollout patch → then **2-node / 16-card**.
