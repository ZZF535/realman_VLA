import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy

# ================= 配置部分 =================
# 替换为你存放上述文件的文件夹路径
pretrained_policy_path = "/path/to/your/model_folder" 
# ===========================================

def main():
    # 1. 自动加载模型、权重和归一化统计数据
    # from_pretrained 会读取 config.json 并实例化 ACTPolicy
    # 它还会自动加载 preprocessor 和 postprocessor 的统计数据
    print(f"正在加载模型: {pretrained_policy_path} ...")
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    
    # 2. 设置设备 (GPU 或 CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    policy.eval() # 切换到评估模式
    print(f"模型已加载到: {device}")

    # 3. 构造伪造的输入数据 (Dummy Data)
    # 根据你的 config.json，我们需要构造以下维度的输入：
    # 图像: [Batch, Channel, Height, Width] -> [1, 3, 480, 640]
    # 状态: [Batch, State_Dim] -> [1, 26]
    
    print("构造测试输入数据...")
    # 注意：LeRobot 期望图像数据是 0-1 之间的 float32 张量
    dummy_observation = {
        "observation.images.image_top": torch.rand(1, 3, 480, 640).to(device),
        "observation.images.image_left_wrist": torch.rand(1, 3, 480, 640).to(device),
        "observation.images.image_right_wrist": torch.rand(1, 3, 480, 640).to(device),
        "observation.state": torch.randn(1, 26).to(device)
    }

    # 4. 执行推理
    print("开始推理...")
    with torch.no_grad():
        # select_action 会自动处理：
        # -> 预处理 (归一化)
        # -> 模型前向传播 (ACT 生成动作块)
        # -> 后处理 (反归一化)
        action = policy.select_action(dummy_observation)

    # 5. 输出结果
    # ACT 输出通常是一个动作块 (Action Chunk)，但 select_action 默认可能只返回当前步
    # 具体取决于 policy 的内部实现，但在 eval 模式下通常直接可用
    print("\n推理成功！")
    print(f"输出动作形状: {action.shape}") 
    print(f"输出动作数据 (前5维): {action[0, :5]}") # 打印一部分看看

if __name__ == "__main__":
    main()