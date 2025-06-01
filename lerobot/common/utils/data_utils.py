import torch
import json
import os
def tensor_to_list(obj):
    """
    递归地将包含 torch.Tensor 的结构转换为纯 Python 数据结构。
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj

def save_to_json(data, path):
    """
    保存数据为 JSON 文件。
    """
    converted_data = tensor_to_list(data)
    
    # 创建路径中不存在的文件夹
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存为 JSON 文件
    with open(path, 'w') as f:
        json.dump(converted_data, f, indent=4)