#!/usr/bin/env bash
# CA_verl full-30B TE-NPU single-node, clean bf16-vs-mxfp8 comparison.
# Same layout EP4 x TP2 x PP2 + precision-aware optimizer offload.
# Toggle: MXFP8=0|1  TAG=...  STEPS=N
export DEVICES=0,1,2,3,4,5,6,7 NGPUS=8 TP=2 EP=4 RTP=4 GMU=0.4 BS=16 MICRO=1
export RECOMPUTE=1 CPUINIT=1 PO=True GO=True OO=True
export STEPS=${STEPS:-3}
export MXFP8=${MXFP8:-0}
export TAG=${TAG:-te30b}
export CANN_ENV=/home/CA_verl/cann/CANN_9.1.T551.B011_0526_5patch/cann-9.1.T551/set_env.sh
# HCCL timeouts: mxfp8 first-forward lazy init (fp8 quant + EP all-to-all + grouped GEMM
# first compile) can exceed the 120s default and trip PP p2p socket timeout (EI0006).
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-900}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-900}
bash /home/CA_verl/run_30b.sh \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
