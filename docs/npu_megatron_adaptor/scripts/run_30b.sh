#!/usr/bin/env bash
# CA_verl 30B-A3B GRPO, env-tunable. Env:
#   DEVICES=0,1,2,3  NGPUS=4  EP=2  TP=2  RTP=4  GMU=0.4  RECOMPUTE=1  STEPS=N  MXFP8=1  TAG=...
set -xeo pipefail
E=/home/miniconda3/envs/CA_verl
source ${CANN_ENV:-/home/d00568668/cann/CANN_9.1.T551.B011_0526_5patch/cann-9.1.T551/set_env.sh}
[ -f /home/CA_verl/nnal/atb/set_env.sh ] && source /home/CA_verl/nnal/atb/set_env.sh
export LD_PRELOAD=$E/lib/libstdc++.so.6
export ASCEND_RT_VISIBLE_DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
export HF_HUB_OFFLINE=1
export HF_HOME=/home/CA_verl/.hf_home
export VLLM_ASCEND_ENABLE_NZ=0
export PATH=$E/bin:$PATH

EXTRA=()
if [ "${MXFP8:-0}" = "1" ]; then
  EXTRA+=(
    "+actor_rollout_ref.actor.megatron.override_transformer_config.fp8=hybrid"
    "+actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=mxfp8"
  )
fi
if [ "${CPUINIT:-0}" = "1" ]; then EXTRA+=("+actor_rollout_ref.actor.megatron.override_transformer_config.use_cpu_initialization=True"); fi
if [ "${RECOMPUTE:-0}" = "1" ]; then
  EXTRA+=(
    "+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full"
    "+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform"
    "+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1"
  )
fi

$E/bin/python -m verl.trainer.main_ppo \
    model_engine=megatron \
    ++ray_kwargs.ray_init.address=local \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/CA_verl/data/gsm8k_rl/train.parquet \
    data.val_files=/home/CA_verl/data/gsm8k_rl/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=/home/CA_verl/models/Qwen3-30B-A3B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=megatron_adaptor \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP:-2} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP:-2} \
    actor_rollout_ref.actor.megatron.param_offload=${PO:-True} \
    actor_rollout_ref.actor.megatron.grad_offload=${GO:-True} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${OO:-True} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${RTP:-4} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GMU:-0.4} \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    trainer.n_gpus_per_node=${NGPUS:-8} \
    trainer.nnodes=1 \
    trainer.logger=console \
    trainer.project_name=CA_verl_validate \
    trainer.experiment_name=grpo-qwen3-30b-${TAG:-run} \
    trainer.total_training_steps=${STEPS:-30} \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "${EXTRA[@]}" "$@"
