import torch
import numpy as np
import os
import json
import safetensors.torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

# === 仅需修改这里 ===
MODEL_PATH = "/home/robot/lerobot-main/outputs/train/018000/pretrained_model"

def load_stats(model_path):
    # (复用之前的智能加载逻辑)
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        if "normalization_stats" in cfg:
            return cfg["normalization_stats"]["action"]
    
    stats_path = os.path.join(model_path, "statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)["action"]
    return None

def main():
    print("🔍 开始诊断...")
    
    # 1. 检查统计数据
    stats = load_stats(MODEL_PATH)
    if stats:
        mean = np.array(stats["mean"])
        std = np.array(stats.get("std", stats.get("scale")))
        print(f"✅ 统计数据形状: {mean.shape}")
        print(f"   Mean (前6 - 左手?): {mean[:6]}")
        print(f"   Mean (中6 - 右手?): {mean[7:13]}") # 假设14维结构
        print(f"   Mean (后6 - 右手?): {mean[13:19] if len(mean)>14 else '无'}")
        
        # 关键判断：Mean 值通常代表机械臂的"平均姿态"
        # 如果是弧度，值应该在 -3.14 ~ 3.14 之间
        # 如果是角度，值可能很大
        if np.max(np.abs(mean)) > 7:
            print("💡 提示: 统计数据看起来像是【角度 (Degree)】")
        else:
            print("💡 提示: 统计数据看起来像是【弧度 (Radian)】")
    else:
        print("❌ 未找到统计数据！")

    # 2. 检查模型配置
    config = PreTrainedConfig.from_pretrained(MODEL_PATH)
    print(f"✅ 模型配置:")
    print(f"   Action Dimension: {config.action_feature.shape if config.action_feature else 'Unknown'}")
    print(f"   Input Features: {list(config.input_features.keys())}")
    
    # 检查 state 的维度定义
    if "observation.state" in config.input_features:
        print(f"   State Expectation: {config.input_features['observation.state'].shape}")
    
if __name__ == "__main__":
    main()