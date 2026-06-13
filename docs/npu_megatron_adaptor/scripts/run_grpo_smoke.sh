#!/usr/bin/env bash
set -xeo pipefail   # NOTE: no -u; vendor set_env.sh reference unbound vars (CMAKE_PREFIX_PATH/ZSH_VERSION)
E=/home/miniconda3/envs/b84_verl_017
source /home/d00568668/cann/CANN_9.1.T551.B011_0526_5patch/cann-9.1.T551/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_PRELOAD=$E/lib/libstdc++.so.6
export ASCEND_RT_VISIBLE_DEVICES=${NPUS:-0,1}
export HF_HUB_OFFLINE=1
export PATH=$E/bin:$PATH

$E/bin/python -m verl.trainer.main_ppo \
    model_engine=megatron \
    ++ray_kwargs.ray_init.address=local \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/b84412626/data/gsm8k_rl/train.parquet \
    data.val_files=/home/b84412626/data/gsm8k_rl/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=/home/b84412626/models/Qwen3-0.6B \
    actor_rollout_ref.actor.strategy=megatron_adaptor \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.logger=console \
    trainer.project_name=verl_npu_smoke \
    trainer.experiment_name=grpo-qwen3-0.6b-megatron-adaptor-vllm \
    trainer.total_training_steps=2 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "$@"
