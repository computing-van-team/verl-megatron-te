#!/usr/bin/env bash
# Saturated full-30B GRPO (aligned to m00881603): prompt 1024 + response 2048,
# use_dynamic_bsz to pack tokens so MoE GEMMs saturate (tok/exp >= mxfp8 knee).
set -xeo pipefail
E=/home/miniconda3/envs/CA_verl
source ${CANN_ENV:-/home/CA_verl/cann/CANN_9.1.T551.B011_0526_5patch/cann-9.1.T551/set_env.sh}
[ -f /home/CA_verl/nnal/atb/set_env.sh ] && source /home/CA_verl/nnal/atb/set_env.sh
export LD_PRELOAD=$E/lib/libstdc++.so.6
export LD_LIBRARY_PATH=$E/lib/python3.10/site-packages/torch/lib:$E/lib/python3.10/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
export HF_HUB_OFFLINE=1
export HF_HOME=/home/CA_verl/.hf_home
export VLLM_ASCEND_ENABLE_NZ=0
export PATH=$E/bin:$PATH
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-900}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-900}

PLEN=${PLEN:-1024}; RLEN=${RLEN:-2048}
ACT_TOK=$(( (PLEN + RLEN) * 4 ))
INF_TOK=$(( (PLEN + RLEN) * 8 ))

EXTRA=()
if [ "${MXFP8:-0}" = "1" ]; then
  EXTRA+=(
    "+actor_rollout_ref.actor.megatron.override_transformer_config.fp8=hybrid"
    "+actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=mxfp8"
  )
fi
if [ "${CPUINIT:-1}" = "1" ]; then EXTRA+=("+actor_rollout_ref.actor.megatron.override_transformer_config.use_cpu_initialization=True"); fi
if [ "${RECOMPUTE:-1}" = "1" ]; then
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
    data.train_batch_size=${BS:-16} \
    data.max_prompt_length=${PLEN} \
    data.max_response_length=${RLEN} \
    actor_rollout_ref.model.path=${MODEL:-/home/CA_verl/models/Qwen3-30B-A3B} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=megatron_adaptor \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI:-4} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACT_TOK} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP:-4} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP:-2} \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${RTP:-4} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GMU:-0.5} \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_model_len=$(( PLEN + RLEN )) \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INF_TOK} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INF_TOK} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    trainer.n_gpus_per_node=${NGPUS:-8} \
    trainer.nnodes=1 \
    trainer.logger=console \
    trainer.project_name=CA_verl_validate \
    trainer.experiment_name=grpo-qwen3-30b-sat-${TAG:-run} \
    trainer.total_training_steps=${STEPS:-3} \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "${EXTRA[@]}" "$@"
