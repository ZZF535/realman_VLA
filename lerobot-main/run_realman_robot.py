import torch
import time
import numpy as np
from pathlib import Path

# --- LeRobot 核心组件 ---
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

# --- 你的自定义机器人组件 ---
# 假设你的 Realman 代码在 src/lerobot/robots/realman/realman.py
from src.lerobot.robots.realman.realman import Realman
from src.lerobot.robots.realman.configuration_realman import RealmanConfig
from src.lerobot.robots.bi_base_robot.bi_base_robot import BiBaseRobot, BiBaseRobotConfig

# ================= 配置区域 =================
# 1. 模型路径
CHECKPOINT_PATH = "/home/robot/lerobot-main/outputs/train/checkpoints/last/pretrained_model"

# 2. 机器人 IP 配置
LEFT_IP = "169.254.128.18"
RIGHT_IP = "169.254.128.19"
ROBOT_PORT = 8080

# 3. 相机配置 (序列号)
CAMERA_SERIALS = {
    "image_top": "346522073032",
    "image_left_wrist": "243722073715",
    "image_right_wrist": "346522074543",
}

# 4. 运行参数
FPS = 15            # 降低帧率防卡死
VELOCITY = 5        # 安全速度 5%
FREQUENCY = 5       # 控制频率 5Hz
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def build_robot():
    """手动组装双臂机器人和相机"""
    print("🤖 正在初始化机器人和相机...")
    
    # 1. 配置相机
    cameras = {}
    for name, serial in CAMERA_SERIALS.items():
        config = IntelRealSenseCameraConfig(
            serial_number=serial,
            width=640,
            height=480,
            fps=FPS,
            use_depth=False
        )
        cameras[name] = IntelRealSenseCamera(config)

    # 2. 配置机械臂 (左 & 右)
    left_config = RealmanConfig(ip=LEFT_IP, port=ROBOT_PORT, velocity=VELOCITY)
    right_config = RealmanConfig(ip=RIGHT_IP, port=ROBOT_PORT, velocity=VELOCITY)
    
    left_arm = Realman(left_config)
    right_arm = Realman(right_config)

    # 3. 组装双臂机器人
    # 注意：这里我们不需要复杂的 BiBaseRobotConfig，只需传入实例
    robot = BiBaseRobot(BiBaseRobotConfig(), left_arm, right_arm, cameras)
    
    return robot

def prepare_batch(robot_obs, device):
    """
    数据预处理：
    1. 补全缺失的相机图像 (黑帧)
    2. 补齐状态向量维度 (14 -> 26)
    3. 归一化/转 Tensor (由 select_action 内部自动处理，这里只需整理格式)
    """
    batch = {}
    
    # --- A. 处理图像 ---
    # 你的模型需要 top, left, right 三个视角
    required_images = ["image_top", "image_left_wrist", "image_right_wrist"]
    
    for cam_key in required_images:
        # 对应 robot_obs 里的键名 (BiBaseRobot 通常返回 'image_top' 等)
        if cam_key in robot_obs and robot_obs[cam_key] is not None:
            # 正常数据: [H, W, C] -> 转 Tensor [C, H, W] -> 归一化 0-1
            img = torch.from_numpy(robot_obs[cam_key]).float().permute(2, 0, 1) / 255.0
        else:
            # 🔴 补丁: 丢帧/未连接时，补全黑帧
            # print(f"⚠️ 补全黑帧: {cam_key}")
            img = torch.zeros((3, 480, 640), dtype=torch.float32)
            
        # 增加 Batch 维度: [C, H, W] -> [1, C, H, W]
        # 注意键名要加上 "observation.images." 前缀以匹配模型
        batch[f"observation.images.{cam_key}"] = img.unsqueeze(0).to(device)

    # --- B. 处理状态 (State) ---
    # 机器人返回的是 14 维 (左7 + 右7)
    state = robot_obs["observation.state"]
    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(state).float()
    
    state = state.to(device)
    
    # 🔴 补丁: 维度补齐 14 -> 26 (补两个 6维 零向量)
    if state.shape[-1] == 14:
        left_part = state[:7]
        right_part = state[7:]
        zeros = torch.zeros(6, device=device)
        state = torch.cat([left_part, zeros, right_part, zeros])
        
    # 增加 Batch 维度: [26] -> [1, 26]
    batch["observation.state"] = state.unsqueeze(0)
    
    return batch

def main():
    # 1. 加载模型
    print(f"🚀 正在加载模型: {CHECKPOINT_PATH}")
    policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
    policy.to(DEVICE)
    policy.eval()
    print("✅ 模型加载完毕！")

    # 2. 连接机器人
    robot = build_robot()
    robot.connect()
    print("✅ 机器人连接成功！(可视化已禁用)")
    
    # ⚠️ 禁用可视化以防报错
    robot.visualizer = None 

    print(f"🏁 开始推理循环 (Ctrl+C 停止)... 频率: {FREQUENCY}Hz")
    period = 1.0 / FREQUENCY
    
    try:
        while True:
            loop_start = time.perf_counter()

            # --- 1. 获取观测 ---
            obs = robot.get_observation()
            
            # --- 2. 预处理 (转 Tensor + 补齐维度) ---
            batch = prepare_batch(obs, DEVICE)
            
            # --- 3. 模型推理 ---
            with torch.no_grad():
                # select_action 内部会自动应用 normalize_inputs
                action = policy.select_action(batch)
            
            # 此时 action 是 [1, 26] 的 Tensor，且已反归一化
            action = action.squeeze(0).cpu().numpy() # -> [26]
            
            # --- 4. 解析动作 (提取前7和后7，组成字典) ---
            # 索引映射: 左臂 [0:7], 右臂 [13:20] (跳过中间的6维位姿)
            action_dict = {}
            
            # 左臂
            left_names = robot.left_robot.config.joint_names # ['joint_1', ...]
            for i, name in enumerate(left_names):
                action_dict[f"left_{name}"] = action[i]
                
            # 右臂
            right_names = robot.right_robot.config.joint_names
            for i, name in enumerate(right_names):
                action_dict[f"right_{name}"] = action[13 + i]

            # --- 5. 发送指令 ---
            robot.send_action(action_dict)

            # --- 6. 维持频率 ---
            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\n🛑 停止运行...")
    finally:
        robot.disconnect()
        print("👋 机器人已断开连接。")

if __name__ == "__main__":
    main()