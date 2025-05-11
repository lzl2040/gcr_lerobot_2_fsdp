#!/bin/bash

# 默认参数值
NNODES=1
NPROC_PER_NODE=2
JOB_NAME="test-ft"
OPTIMIZER_LR=1e-4
SCHEDULER_WARMUP_STEPS=10000
SCHEDULER_DECAY_STEPS=1500000
NODE_RANK=0


# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --job_name)
            JOB_NAME="$2"
            shift 2
            ;;
        --optimizer_lr)
            OPTIMIZER_LR="$2"
            shift 2
            ;;
        --scheduler_warmup_steps)
            SCHEDULER_WARMUP_STEPS="$2"
            shift 2
            ;;
        --scheduler_decay_steps)
            SCHEDULER_DECAY_STEPS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [[ -z "$JOB_NAME" ]]; then
    echo "错误：必须指定 --job_name"
    exit 1
fi

# 固定输出目录（根据需求修改）
FIXED_OUTPUT_DIR="original_qw"
# 执行训练命令
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=9985 \
    lerobot/scripts/fsdp_train.py \
    --policy.type="qwen" \
    --output_dir="$FIXED_OUTPUT_DIR" \
    --dataset.repo_id="whatever" \
    --dataset.processor="/Data/lzl/qwen2.5_vl_7b/Qwen2.5-VL-7B-Instruct" \
    --dataset.parent_dir="/Data/lerobot_data/simulated" \
    --dataset.data_mix="libero" \
    --policy.scheduler_warmup_steps=$SCHEDULER_WARMUP_STEPS \
    --policy.scheduler_decay_steps=$SCHEDULER_DECAY_STEPS \
    --policy.optimizer_lr=$OPTIMIZER_LR \
    --policy.train_main_layers=0 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --wandb.enable=false \
    --wandb.project="pi0first" \
    --job_name="$JOB_NAME" \
    --log_dir="logs"