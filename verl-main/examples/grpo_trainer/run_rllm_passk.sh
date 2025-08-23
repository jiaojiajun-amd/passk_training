#!/bin/bash
set -x


export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

# Find the directory where rllm package is located
RLLM_DIR=$(python3 -c "import rllm; import os; print(os.path.dirname(os.path.dirname(rllm.__file__)))")

MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B



python examples/data_preprocess/rllm.py --train_dataset_name "deepscaler" --local_dir data/rllm

## Commonly
max_prompt_length=$((1024 * 4))

#--------------------------------------

## Entropy_Coeff
entropy_coeff=0.001

#----------Adaptive Entropy----------#
target_entropy=0.28
beta=-7.5


warmup_beta="use_default"            ##default: use_default
manual_global_step=0                 ##default: 0

use_adaptive_entropy=True
set_ref_entropy=False


#----------Entropy Gain Clip---------#
# use_entropy_gain_clip=Ture
use_entropy_gain_clip=True

entropy_clip_mode="high"             ##default: "high"
entropy_clip_ratio=0.2


#------------Pg Loss Clip-------------#
pg_loss_clip=True


## +Clip-Higher
clip_ratio_low=0.2
clip_ratio_high=0.25


## +Token-level Loss
loss_agg_mode="token-mean"


## +Dynamic Sampling
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10


#Did NOT Added#
## - [x]  + Overlong Filtering
## - [x]  + Soft Overlong Punishment



## Extending response length

# RES_LENGTH=24576 
# RES_LENGTH=16384
RES_LENGTH=8192

TP=1
SP=1

MAX_TOKEN_LEN=$(((RES_LENGTH + max_prompt_length + 1000) / SP))

calculate_kl_loss=True
use_kl_loss=True
kl_loss_coef=0.001


train_file_path = data/rllm/deepscaler.parquet
val_file_path = data/rllm/aime24.parquet




# Train over 4 nodes, 8 A100-80GB GPUs per node.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=passk_training \
    data.train_files=${train_file_path} \
    data.val_files=${val_file_path} \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=$RES_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    +actor_rollout_ref.actor.calculate_kl_loss=${calculate_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    +actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + $RES_LENGTH)) \
    ++actor_rollout_ref.rollout.enable_chunked_prefill=True \
    ++actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKEN_LEN \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='passk-training-rllm' \
    trainer.experiment_name='1.5b-8k-passk' \
    ++trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 