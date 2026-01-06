import torch
import time
import numpy as np
import os
import cv2
from lerobot.policies.act.modeling_act import ACTPolicy
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
from safetensors.torch import load_file

# ================= 配置区 =================
MODEL_PATH = "/home/robot/lerobot-main/outputs/train/checkpoints/last/pretrained_model"
LEFT_IP, RIGHT_IP = "169.254.128.18", "169.254.128.19"
CAM_INDICES = {"image_top": 8, "image_left_wrist": 0, "image_right_wrist": 14}
# ==========================================

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    
    # 1. 加载模型
    policy = ACTPolicy.from_pretrained(MODEL_PATH)
    policy.eval()

    # --- 🚀 v2.1 归一化参数对齐补丁 (跨版本核心修复) ---
    print("🔧 正在检测模型结构并注入归一化参数...")
    sd = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
    
    with torch.no_grad():
        # 1. 注入状态归一化 (State)
        if hasattr(policy, 'normalize_inputs'):
            # 针对旧版 policy.normalize_inputs.buffer_observation_state
            if "normalize_inputs.buffer_observation_state.mean" in sd:
                policy.normalize_inputs.buffer_observation_state.mean.copy_(sd["normalize_inputs.buffer_observation_state.mean"])
                policy.normalize_inputs.buffer_observation_state.std.copy_(sd["normalize_inputs.buffer_observation_state.std"])
                print("  ✅ State Normalization aligned.")

            # 2. 注入图像归一化 (Images)
            # 遍历所有可能的图像键名
            for cam in ["image_top", "image_left_wrist", "image_right_wrist"]:
                old_key_mean = f"normalize_inputs.buffer_observation_images_{cam}.mean"
                old_key_std = f"normalize_inputs.buffer_observation_images_{cam}.std"
                attr_name = f"buffer_observation_images_{cam}"
                
                if old_key_mean in sd and hasattr(policy.normalize_inputs, attr_name):
                    getattr(policy.normalize_inputs, attr_name).mean.copy_(sd[old_key_mean])
                    getattr(policy.normalize_inputs, attr_name).std.copy_(sd[old_key_std])
                    print(f"  ✅ {cam} Normalization aligned.")

        # 3. 注入动作去归一化 (Action)
        if hasattr(policy, 'unnormalize_outputs'):
            if "unnormalize_outputs.buffer_action.mean" in sd:
                policy.unnormalize_outputs.buffer_action.mean.copy_(sd["unnormalize_outputs.buffer_action.mean"])
                policy.unnormalize_outputs.buffer_action.std.copy_(sd["unnormalize_outputs.buffer_action.std"])
                print("  ✅ Action Unnormalization aligned.")

    # 2. 硬件初始化
    arm_l = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    arm_r = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    arm_l.rm_create_robot_arm(LEFT_IP, 8080)
    arm_r.rm_create_robot_arm(RIGHT_IP, 8080)
    arm_l.rm_set_arm_run_mode(0) # 模式0更安全
    arm_r.rm_set_arm_run_mode(0)
    
    caps = {n: cv2.VideoCapture(i) for n, i in CAM_INDICES.items()}

    try:
        while True:
            # --- 3. 获取观测 (角度 -> 弧度) ---
            _, j_l_deg = arm_l.rm_get_joint_degree() # 睿尔曼原生返回度
            _, j_r_deg = arm_r.rm_get_joint_degree()
            
            # 将前6轴度转为弧度喂给模型
            j_l_rad = np.deg2rad(j_l_deg).tolist()
            j_r_rad = np.deg2rad(j_r_deg).tolist()
            
            # 夹爪通常不需要转弧度，保持原始比例 (如果是0-1000)
            _, grip_l = arm_l.rm_get_rm_plus_state_info()
            _, grip_r = arm_r.rm_get_rm_plus_state_info()
            g_l, g_r = grip_l['pos'][0], grip_r['pos'][0]

            # 构造 26 维输入 [左7 + 位姿6] + [右7 + 位姿6]
            l_7 = torch.tensor(j_l_rad + [g_l])
            r_7 = torch.tensor(j_r_rad + [g_r])
            full_state = torch.cat([l_7, torch.zeros(6), r_7, torch.zeros(6)]).unsqueeze(0)

            # 构造图像
            batch = {"observation.state": full_state}
            for name, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).float()
                    batch[f"observation.images.{name}"] = img.permute(2, 0, 1).unsqueeze(0) / 255.0

            # --- 4. 推理 ---
            with torch.no_grad():
                action = policy.select_action(batch).squeeze(0).cpu().numpy()

            # --- 5. 下发动作 (弧度 -> 角度) ---
            # 左臂: 关节 0-5 是弧度，转回度
            l_joints_deg = np.rad2deg(action[0:6]).tolist()
            l_grip = int(np.clip(action[6], 0, 1000))
            
            # 右臂: 关节 13-18 是弧度，转回度
            r_joints_deg = np.rad2deg(action[13:19]).tolist()
            r_grip = int(np.clip(action[19], 0, 1000))

            # 阻塞式下发，观察动作
            ret_r = arm_r.rm_movej(r_joints_deg, 20, 0, 1, 1) 
            arm_r.rm_set_gripper_position(r_grip, False, 1)

            print(f"R_J1: {j_r_deg[0]:.2f}° -> CMD: {r_joints_deg[0]:.2f}° | Ret: {ret_r}")

    except KeyboardInterrupt:
        pass
    finally:
        arm_l.rm_destroy()
        arm_r.rm_destroy()
        for cap in caps.values(): cap.release()

if __name__ == "__main__":
    main()