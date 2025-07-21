#!/usr/bin/env bash
# exit on any error
# set -xeuo pipefail 

# Wandb API key
export WANDB_API_KEY=XXXXXXX

# Project and experiment names  
project_name='astracodev2'  
exp_name='qwen2.5-1.5B_deepseek_full_run_2'

# Log dir
rollout_data_dir='outputs/train/qwen2.5-1.5B_deepseek_full_run_2'
validation_data_dir='outputs/test/qwen2.5-1.5B_deepseek_full_run_2'

# Define paths
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_DIR="data/cf"  

PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$DATA_DIR/astratrain_python.parquet \
  data.val_files=$DATA_DIR/astratest_python.parquet \
  data.train_batch_size=64 \
  data.val_batch_size=64 \
  data.max_prompt_length=2048 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  reward_model.reward_manager=dapo \
  +reward_model.max_resp_len=8192 \
  +reward_model.overlong_buffer_cfg.enable=True \
  +reward_model.overlong_buffer_cfg.log=True \
  +reward_model.overlong_buffer_cfg.len=2048 \
  +reward_model.overlong_buffer_cfg.penalty_factor=0.5 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.4 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.n=32 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.temperature=1.2 \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  custom_reward_function.path=verl/utils/reward_score/sandbox_fusion/__init__.py \
  custom_reward_function.name=compute_score \
  trainer.logger="['console', 'wandb']" \
  trainer.project_name=$project_name \
  trainer.experiment_name=$exp_name \
  trainer.rollout_data_dir=$rollout_data_dir \
  trainer.validation_data_dir=$validation_data_dir \
  trainer.val_before_train=True \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=45 \
  trainer.test_freq=45 \
  trainer.total_epochs=10 \
  2>&1 | tee verl_train.log
