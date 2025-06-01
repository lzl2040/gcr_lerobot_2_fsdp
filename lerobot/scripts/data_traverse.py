#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
import os
import glob
import json
import functools
from pathlib import Path
from datetime import datetime
from pprint import pformat
from termcolor import colored
from typing import Any
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    always_wrap_policy
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.lerobot_dataset_example import MultiDatasetforDistTraining, extra_collate_fn
from lerobot.common.datasets.sampler import EpisodeAwareSampler, DistEpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy
import numpy as np

def save_item(preprocess_root, new_obs_image_keys, item, data_id):
    # 构建目标文件夹路径
    folder_path = os.path.join(preprocess_root, f"data_{data_id:010d}")
    
    # 如果文件夹不存在则创建
    os.makedirs(folder_path, exist_ok=True)

    # 拆分数据
    action = item['action']                  # shape: (50, 7)
    state = item['observation.state']     # shape: (1, 8)

    # 保存各个 tensor 为 pt 文件
    torch.save(action, os.path.join(folder_path, 'action.pt'))
    torch.save(state, os.path.join(folder_path, 'state.pt'))
    for key in new_obs_image_keys:
        img_data = item[key]
        save_name = key.replace(".", "_")
        img_save_path = os.path.join(folder_path, f'{save_name}.npz')
        np.savez_compressed(img_save_path, image=img_data.numpy())
    
    json_path = os.path.join(folder_path, "frame_pad.json")
    is_pad_frame = item["is_pad_frame"]
    with open(json_path, 'w') as f:
        json.dump(is_pad_frame, f, indent=4)
    source_path = os.path.join(folder_path, "data_source.txt")
    with open(source_path, "w") as f:
        f.write(item["source"])
    print(f"Data saved in {folder_path}")


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    # 初始化分布式环境
    
    # 初始化配置
    cfg.validate()
    # 设置随机种子
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # 数据集初始化
    
    step = 1
    seed = cfg.seed
            
    image_transforms = (ImageTransforms(cfg.dataset.image_transforms))
    print(image_transforms)
    dataset = MultiDatasetforDistTraining(
        cfg=cfg, 
        image_transforms=image_transforms,
        seed=seed,
        data_mix=cfg.dataset.data_mix,
        vla2root_json="vla2root.json",
        # vla2root_json="vla2root_bak_single.json"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=16,
        # collate_fn=extra_collate_fn,
        pin_memory=False,
    )
    
    # dataloader_iter = cycle(dataloader)
    
    print("Starting training loop...")
        
    
    img_obs_key = dataset.new_obs_image_keys
    step = 1
    data_num = 0
    preprocess_root = "/mnt/wangxiaofa/robot_dataset/lerobot-format/pre_openx_agi"
    os.makedirs(preprocess_root, exist_ok=True)
    # preprocess_root = "/Data/lzl/debug_process_data"
    for batch in dataloader:
        batch_start = time.perf_counter()
        # batch = next(dataloader_iter)
        data_time = time.perf_counter() - batch_start
        
        bs = batch["action"].shape[0]
        for b in range(bs):
            item = {}
            item["action"] = batch["action"][b]
            item["observation.state"] = batch["observation.state"][b]
            is_pad_frame = {}
            for key in img_obs_key:
                img_data = batch[key][b]
                # print(img_data.shape)
                is_pad_frame[key] = batch["is_pad_frame"][key][b].item()
                item[key] = img_data
            item["is_pad_frame"] = is_pad_frame
            item["source"] = batch["source"][b]
            save_item(preprocess_root, img_obs_key, item, data_num)
            data_num += 1
            print(f"Step={step} data num={data_num}")
                
        
        step += 1
        print(f"Step={step}, data_time = {data_time}")

if __name__ == "__main__":
    # # 设置环境变量
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    # os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    os.environ['WANDB_API_KEY'] = '9e1c3ac77856b8ebb5573c4e1e250c84aabfb904'
    
    # 启动训练
    train()