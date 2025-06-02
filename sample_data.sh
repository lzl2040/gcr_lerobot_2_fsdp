# torchrun --nnodes=1 \
#     --nproc_per_node=4 \
#     --master_port=9911 \
python \
    lerobot/scripts/data_traverse.py \
    --policy.type="pi0" \
    --save_freq=100 \
    --dataset.repo_id="whatever" \
    --dataset.processor="/mnt/wangxiaofa/qwen_params/Qwen2.5-VL-7B-Instruct/" \
    --dataset.parent_dir="/mnt/wangxiaofa/robot_dataset/lerobot-format/" \
    --dataset.data_mix="oxe_magic_soup_plus" \
    --dataset.use_history_state=false \
    --dataset.sample_ratio=0.4 \
    --output_dir="qwen_flow" \
    --batch_size=1
    