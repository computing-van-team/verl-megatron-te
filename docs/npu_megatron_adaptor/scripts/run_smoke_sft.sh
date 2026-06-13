#!/usr/bin/env bash
# Smoke test: verl SFT trainer with backend=megatron_adaptor on Ascend NPU
# Usage: bash run_smoke_sft.sh [NPU_ID] [FP8_ARGS...]
set -xeuo pipefail

NPU_ID=${1:-0}          # single id "0" or list "0,1"
shift || true
NPROC=${NPROC:-1}
TP_SIZE=${TP_SIZE:-1}

E=/home/miniconda3/envs/b84_verl_017
source /home/d00568668/cann/CANN_9.1.T551.B011_0526_5patch/cann-9.1.T551/set_env.sh
export LD_PRELOAD=$E/lib/libstdc++.so.6
export ASCEND_RT_VISIBLE_DEVICES=$NPU_ID
export HF_HUB_OFFLINE=1
export PATH=$E/bin:$PATH

MODEL_PATH=/home/b84412626/models/Qwen3-0.6B
DATA_DIR=/home/b84412626/data/gsm8k_sft
CKPTS=/home/b84412626/runs/smoke-sft-megatron-adaptor

mkdir -p "$CKPTS"

$E/bin/torchrun --standalone --nnodes=1 --nproc-per-node=$NPROC \
    -m verl.trainer.sft_trainer \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.pad_mode=no_padding \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=2048 \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    model=hf_model \
    model.path=$MODEL_PATH \
    model.use_remove_padding=True \
    engine=megatron_adaptor \
    engine.tensor_model_parallel_size=$TP_SIZE \
    engine.pipeline_model_parallel_size=1 \
    engine.context_parallel_size=1 \
    engine.use_mbridge=True \
    optim=megatron \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.2 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=1e-6 \
    trainer.test_freq=after_each_epoch \
    trainer.save_freq=-1 \
    trainer.logger=['console'] \
    trainer.project_name=verl_npu_smoke \
    trainer.experiment_name=qwen3-0.6b-megatron-adaptor \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    trainer.default_local_dir=$CKPTS \
    trainer.resume_mode=disable \
    "$@"
