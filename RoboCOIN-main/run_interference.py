# import torch
# import time
# import numpy as np
# import os
# from pathlib import Path
# import cv2

# # 基础组件
# from lerobot.policies.act.modeling_act import ACTPolicy
# from lerobot.cameras.realsense.camera_realsense import RealSenseCamera, RealSenseCameraConfig
# from src.lerobot.robots.realman.realman import Realman
# from src.lerobot.robots.realman.configuration_realman import RealmanConfig
# from Robotic_Arm.rm_robot_interface import *

# # ================= 配置区 =================
# MODEL_PATH = "/home/robot/lerobot-main/outputs/train/checkpoints/last/pretrained_model"
# LEFT_IP, RIGHT_IP = "169.254.128.18", "169.254.128.19"
# # 相机序列号
# CAMERAS = {
#     "image_top": "346522073032",
#     "image_left_wrist": "243722073715",
#     "image_right_wrist": "346522074543",
# }
# # ==========================================


# def main():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    
#     # 1. 加载模型
#     print(f"🚀 正在加载原生模型: {MODEL_PATH}")
#     policy = ACTPolicy.from_pretrained(MODEL_PATH)
#     policy.eval()

#     # 2. 按照官方 Demo 实例化和连接
#     arm_l = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#     arm_r = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    
#     handle_l = arm_l.rm_create_robot_arm(LEFT_IP, 8080)
#     handle_r = arm_r.rm_create_robot_arm(RIGHT_IP, 8080)
    
#     if handle_l.id == -1 or handle_r.id == -1:
#         print("❌ 机械臂连接失败，请检查 IP")
#         return
    
#     print(f"✅ 连接成功! ID: Left({handle_l.id}), Right({handle_r.id})")

#     # 强制切换到轨迹模式 (Mode 0) 确保 movej 生效
#     arm_l.rm_set_arm_run_mode(0)
#     arm_r.rm_set_arm_run_mode(0)

#     # 3. 相机初始化
#     caps = {n: cv2.VideoCapture(i) for n, i in CAMERAS.items()}

#     try:
#         print("🏁 开始同步推理 (25Hz)...")
#         while True:
#             start_time = time.perf_counter()
            
#             # --- 4. 获取观测 (原生 API 调用) ---
#             # 直接获取角度
#             _, joints_l = arm_l.rm_get_joint_degree()
#             _, joints_r = arm_r.rm_get_joint_degree()
            
#             # 获取夹爪位置
#             _, grip_l_info = arm_l.rm_get_rm_plus_state_info()
#             _, grip_r_info = arm_r.rm_get_rm_plus_state_info()
#             pos_l = grip_l_info.get('pos', [0])[0]
#             pos_r = grip_r_info.get('pos', [0])[0]
            
#             # 构造 26 维状态 (13+13 结构)
#             s_l_7 = np.array(joints_l + [pos_l])
#             s_r_7 = np.array(joints_r + [pos_r])
#             p_zeros = torch.zeros(6)
            
#             full_state = torch.cat([
#                 torch.from_numpy(s_l_7).float(), p_zeros,
#                 torch.from_numpy(s_r_7).float(), p_zeros
#             ]).unsqueeze(0)
            
#             # 构造图像输入
#             batch = {"observation.state": full_state}
#             for name, cap in caps.items():
#                 ret, frame = cap.read()
#                 if ret:
#                     img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).float()
#                     batch[f"observation.images.{name}"] = img.permute(2, 0, 1).unsqueeze(0) / 255.0

#             # --- 5. 推理 ---
#             with torch.no_grad():
#                 action = policy.select_action(batch).squeeze(0).cpu().numpy()

#             # --- 6. 双臂同步并行下发 ---
#             l_joints_cmd = action[0:6].tolist()
#             l_gripper_cmd = int(np.clip(action[6], 0, 1000))
            
#             r_joints_cmd = action[13:19].tolist()
#             r_gripper_cmd = int(np.clip(action[19], 0, 1000))

#             # 🚀 关键：同步非阻塞下发 (block=0)
#             # v=20 速度, r=0 角度
#             arm_l.rm_movej(l_joints_cmd, 20, 0, 1, 0)
#             arm_r.rm_movej(r_joints_cmd, 20, 0, 1, 0)
            
#             # 夹爪下发
#             arm_l.rm_set_gripper_position(l_gripper_cmd, False, 1)
#             arm_r.rm_set_gripper_position(r_gripper_cmd, False, 1)

#             print(f"L_J1: {joints_l[0]:.1f}->{l_joints_cmd[0]:.1f} | R_J1: {joints_r[0]:.1f}->{r_joints_cmd[0]:.1f}")

#             # 维持频率在 25Hz 左右
#             elapsed = time.perf_counter() - start_time
#             time.sleep(max(0, 0.04 - elapsed))

#     except KeyboardInterrupt:
#         print("\n🛑 停止")
#     finally:
#         arm_l.rm_delete_robot_arm() # 按照 Demo 清理
#         arm_r.rm_delete_robot_arm()
#         for cap in caps.values(): cap.release()

# if __name__ == "__main__":
#     main()

# import torch
# import time
# import numpy as np
# import os
# import cv2
# from pathlib import Path

# # 基础组件
# from lerobot.policies.act.modeling_act import ACTPolicy
# from lerobot.cameras.realsense.camera_realsense import RealSenseCamera, RealSenseCameraConfig
# # 机器人 SDK 接口
# from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

# # ================= 配置区 =================
# # MODEL_PATH = "/home/robot/lerobot-main/outputs/train/checkpoints/last/pretrained_model"
# MODEL_PATH = "/home/robot/lerobot-main/outputs/train/last/pretrained_model"
# LEFT_IP, RIGHT_IP = "169.254.128.18", "169.254.128.19"

# CAMERAS_CONFIG = {
#     "image_top": "346522073032",
#     "image_left_wrist": "243722073715",
#     "image_right_wrist": "346522074543",
# }

# TARGET_KEYS = {
#     "image_top": "observation.images.image_top",
#     "image_left_wrist": "observation.images.image_left_wrist",
#     "image_right_wrist": "observation.images.image_right_wrist"
# }
# # ==========================================

# def main():
#     # 1. 设备环境自动适配
#     try:
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#             # 针对 RTX 50 系列的潜在驱动问题，强制同步报错
#             os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#             print(f"🚀 检测到 GPU: {torch.cuda.get_device_name(0)}，尝试开启显卡加速")
#         else:
#             raise Exception("No CUDA")
#     except:
#         device = torch.device("cpu")
#         torch.set_num_threads(os.cpu_count())
#         print("⚠️ 显卡驱动不兼容或无显卡，切换至 CPU 极致优化模式")

#     # 2. 加载模型
#     print(f"📦 正在加载模型至 {device}...")
#     policy = ACTPolicy.from_pretrained(MODEL_PATH)
#     policy.to(device)
    
#     # 验证模型加载在CPU还是GPU上了
#     param_iterator = iter(policy.parameters())
#     p = next(param_iterator)
#     print("policy_param_device",p.device)
    
#     policy.eval()

#     # 3. 机械臂初始化
#     arm_l = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#     arm_r = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    
#     handle_l = arm_l.rm_create_robot_arm(LEFT_IP, 8080)
#     handle_r = arm_r.rm_create_robot_arm(RIGHT_IP, 8080)
    
#     if handle_l.id == -1 or handle_r.id == -1:
#         print("❌ 机械臂连接失败，请检查 IP")
#         return

#     # 🚀 关键：强制激活机械臂控制模式 (防止连接成功但无法运动)
#     # arm_l.rm_set_arm_run_mode(0) # 0 代表程序控制模式
#     # arm_r.rm_set_arm_run_mode(0)
#     # print("✅ 机械臂控制权限已激活")

#     # 4. 相机初始化
#     caps = {}
#     for name, sn in CAMERAS_CONFIG.items():
#         try:
#             # 统一使用中等分辨率提高推理频率
#             config = RealSenseCameraConfig(serial_number_or_name=sn, fps=30, width=640, height=480)
#             caps[name] = RealSenseCamera(config)
#             caps[name].connect()
#             print(f"📸 相机 {name} 就绪")
#         except Exception as e:
#             print(f"❌ 相机 {name} 失败: {e}")

#     print("\n🏁 开始推理循环...")
#     last_vis_time = 0

#     try:
#         while True:
#             loop_start = time.perf_counter()
            
#             # --- 5. 获取硬件状态 ---
#             _, joints_l = arm_l.rm_get_joint_degree() 
#             _, joints_r = arm_r.rm_get_joint_degree()
            
#             # 构造输入 state (ACT通常是 14维或26维，这里根据你之前 26维逻辑)
#             # 假设只关心 6轴+夹爪，其余补0
#             state_data = np.concatenate([
#                 np.array(joints_l + [0]), # 7
#                 np.zeros(6),              # 6
#                 np.array(joints_r + [0]), # 7
#                 np.zeros(6)               # 6
#             ])
#             full_state = torch.from_numpy(state_data).float().unsqueeze(0).to(device)
            
#             # --- 6. 获取并处理图像 ---
#             batch = {"observation.state": full_state}
#             vis_frames = []
#             do_vis = (time.time() - last_vis_time > 2.0)

#             for name, cap in caps.items():
#                 # frame_rgb = cap.read()
#                 # if frame_rgb is not None:
#                 #     # 预处理：Resize -> Tensor -> Normalize
#                 #     small_rgb = cv2.resize(frame_rgb, (320, 240))
#                 #     img_tensor = torch.from_numpy(small_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
#                 #     batch[TARGET_KEYS[name]] = img_tensor.to(device)
                    
#                 #     if do_vis:
#                 #         vis_frames.append(cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR))
#                 # else:
#                 #     batch[TARGET_KEYS[name]] = torch.zeros((1, 3, 240, 320)).to(device)
#                 frame_rgb = cap.read()
#                 if frame_rgb is not None:
#                     # 不 resize，直接用原始 640×480
#                     img = frame_rgb  # shape (480, 640, 3)
#                     img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
#                     batch[TARGET_KEYS[name]] = img_tensor.to(device)

#                     if do_vis:
#                         vis_frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#                 else:
#                     batch[TARGET_KEYS[name]] = torch.zeros((1, 3, 480, 640)).to(device)




#             if do_vis and len(vis_frames) == 3:
#                 cv2.imwrite("vis_debug.jpg", np.hstack(vis_frames))
#                 last_vis_time = time.time()

#             # --- 7. 推理 (带报错捕获) ---
#             try:
#                 with torch.no_grad():
#                     # 得到输出并转回 CPU numpy
#                     output_action = policy.select_action(batch).squeeze(0).cpu().numpy()
#             except Exception as e:
#                 print(f"❌ 推理崩溃: {e}")
#                 break

#             # --- 8. 下发动作与调试打印 ---
#             #l_cmd = np.rad2deg(output_action[0:6].tolist())
#             #r_cmd = np.rad2deg(output_action[13:19].tolist())
            
#             l_cmd = output_action[0:6].tolist()
#             r_cmd = output_action[13:19].tolist()
            
#             print(f"👉右手: {r_cmd}")

#             l_grip_pos = int(output_action[6])
#             r_grip_pos = int(output_action[19])
#             # print(f"￥￥左手夹爪￥￥r_grip_pos:{l_grip_pos}")    
#             print(f"***右手夹爪***r_grip_pos:{r_grip_pos}") 
                
#             # 🛑 核心调试：计算动作变化量
#             l_diff = np.abs(np.array(l_cmd) - np.array(joints_l)).mean()
#             r_diff = np.abs(np.array(r_cmd) - np.array(joints_r)).mean()

#             # 执行运动 (v=60, block=0)
#             # arm_l.rm_movej(l_cmd[0:6], v=25, r=0, connect=1, block=0)
#             # arm_r.rm_movej(r_cmd[0:6], v=25, r=0, connect=1, block=0)
            
#             arm_l.rm_movej_canfd(l_cmd[0:6], False,0,0,0)
#             arm_r.rm_movej_canfd(r_cmd[0:6], False,0,0,0)           
            
            
            
#             l_grip_result = arm_l.rm_set_gripper_position(l_grip_pos, False, 1)
#             r_grip_result_= arm_r.rm_set_gripper_position(r_grip_pos, False, 1)
#             # print(f"￥￥左手夹爪执行结果￥￥:{l_grip_result}")  
#             print(f"***右手夹爪执行结果***:{r_grip_result_}") 



#             # 打印监控
#             fps = 1.0 / (time.perf_counter() - loop_start)
#             print(f"FPS: {fps:4.1f} | L_Diff: {l_diff:6.4f} | R_Diff: {r_diff:6.4f} | 指令: {r_cmd[0]:.2f}")

#     except KeyboardInterrupt:
#         print("\n👋 停止运行")
#     finally:
#         arm_l.rm_delete_robot_arm()
#         arm_r.rm_delete_robot_arm()
#         for cap in caps.values():
#             cap.disconnect()

# if __name__ == "__main__":
#     main()

import torch
import time
import numpy as np
import os
import cv2
from pathlib import Path

# 基础组件
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera, RealSenseCameraConfig
# 机器人 SDK 接口
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

# ================= 配置区 =================
# MODEL_PATH = "/home/robot/lerobot-main/outputs/train/checkpoints/last/pretrained_model"
MODEL_PATH = "/home/robot/lerobot-main/outputs/train/005000/pretrained_model"
LEFT_IP, RIGHT_IP = "169.254.128.18", "169.254.128.19"

CAMERAS_CONFIG = {
    "image_top": "346522073032",
    "image_left_wrist": "243722073715",
    "image_right_wrist": "346522074543",
}

TARGET_KEYS = {
    "image_top": "observation.images.image_top",
    "image_left_wrist": "observation.images.image_left_wrist",
    "image_right_wrist": "observation.images.image_right_wrist"
}
# ==========================================

def main():
    # 1. 设备环境自动适配
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            print(f"🚀 检测到 GPU: {torch.cuda.get_device_name(0)}，尝试开启显卡加速")
        else:
            raise Exception("No CUDA")
    except:
        device = torch.device("cpu")
        torch.set_num_threads(os.cpu_count())
        print("⚠️ 显卡驱动不兼容或无显卡，切换至 CPU 极致优化模式")

    # 2. 加载模型
    print(f"📦 正在加载模型至 {device}...")
    try:
        policy = ACTPolicy.from_pretrained(MODEL_PATH)
        policy.to(device)
        policy.eval()
        
        # 🚨【关键】启用 Temporal Ensemble 平滑动作，减少抖动
        # ACT 论文推荐系数 0.01，值越小越平滑
        if policy.config.temporal_ensemble_coeff is None:
            print("⚠️ 检测到未启用 Temporal Ensemble，正在手动启用...")
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            policy.config.temporal_ensemble_coeff = 0.01
            policy.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff=0.01, 
                chunk_size=policy.config.chunk_size
            )
            print("✅ Temporal Ensemble 已启用 (coeff=0.01)")
        
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 机械臂初始化
    arm_l = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    arm_r = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    
    handle_l = arm_l.rm_create_robot_arm(LEFT_IP, 8080)
    handle_r = arm_r.rm_create_robot_arm(RIGHT_IP, 8080)
    
    if handle_l.id == -1 or handle_r.id == -1:
        print("❌ 机械臂连接失败，请检查 IP")
        return

    # 4. 相机初始化
    caps = {}
    for name, sn in CAMERAS_CONFIG.items():
        try:
            # 统一使用中等分辨率提高推理频率
            config = RealSenseCameraConfig(serial_number_or_name=sn, fps=30, width=640, height=480)
            caps[name] = RealSenseCamera(config)
            caps[name].connect()
            print(f"📸 相机 {name} 就绪")
        except Exception as e:
            print(f"❌ 相机 {name} 失败: {e}")

    print("\n🏁 开始推理循环...")
    last_vis_time = 0
    
    # 🚨【关键】每次开始新任务前必须重置策略状态！
    # 这会清空 action queue 或重置 temporal ensembler
    policy.reset()
    print("✅ 策略状态已重置")

    try:
        while True:
            loop_start = time.perf_counter()
            
            # --- 5. 获取硬件状态 ---
            # 获取角度数据
            _, joints_l_deg = arm_l.rm_get_joint_degree() 
            _, joints_r_deg = arm_r.rm_get_joint_degree()
            
            # 🚨【核心修复】：模型训练时使用的是角度，所以不需要转弧度！
            # 直接使用角度构造输入 state
            state_data = np.concatenate([
                np.array(joints_l_deg + [0]),  # 7 (角度)
                np.zeros(6),                    # 6 (位姿占位)
                np.array(joints_r_deg + [0]),  # 7 (角度)
                np.zeros(6)                     # 6 (位姿占位)
            ])
            full_state = torch.from_numpy(state_data).float().unsqueeze(0).to(device)
            
            # --- 6. 获取并处理图像 ---
            batch = {"observation.state": full_state}
            vis_frames = []
            do_vis = (time.time() - last_vis_time > 2.0)

            for name, cap in caps.items():
                # 🚨【核心修复】：RealSenseCamera.read() 默认返回 RGB，不是 BGR！
                # 查看 camera_realsense.py 第 274 行：rs.format.rgb8
                frame_rgb = cap.read()
                
                if frame_rgb is not None:
                    # 相机已经是 RGB，模型也是 RGB 训练的，直接用！
                    # 不需要任何颜色转换
                    
                    # 转 Tensor (归一化到 0-1)
                    img_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    batch[TARGET_KEYS[name]] = img_tensor.to(device)

                    if do_vis:
                        # 可视化存储时转 BGR (因为 cv2.imwrite 需要 BGR)
                        vis_frames.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                else:
                    # 丢帧补黑图 (注意尺寸 480, 640)
                    batch[TARGET_KEYS[name]] = torch.zeros((1, 3, 480, 640)).to(device)

            if do_vis and len(vis_frames) == 3:
                cv2.imwrite("vis_debug.jpg", np.hstack(vis_frames))
                last_vis_time = time.time()

            # --- 7. 推理 ---
            try:
                with torch.no_grad():
                    output_action = policy.select_action(batch).squeeze(0).cpu().numpy()
            except Exception as e:
                print(f"❌ 推理崩溃: {e}")
                break

            # --- 8. 下发动作 ---
            # 🚨【核心逻辑】：输出直接用角度 (根据日志观测)
            l_cmd = output_action[0:6].tolist()
            r_cmd = output_action[13:19].tolist()
            
            l_grip_pos = int(output_action[6])
            r_grip_pos = int(output_action[19])
            
            # --- 9. 计算误差 ---
            # 模型输出和当前关节角度的差值（单位：角度）
            l_diff = np.abs(np.array(l_cmd) - np.array(joints_l_deg)).mean()
            r_diff = np.abs(np.array(r_cmd) - np.array(joints_r_deg)).mean()

            print(f"👉右手指令: {r_cmd[0]:.2f} | 夹爪: {r_grip_pos}")

            # 执行 (发送角度)
            arm_l.rm_movej_canfd(l_cmd, False, 0, 0, 0)
            arm_r.rm_movej_canfd(r_cmd, False, 0, 0, 0)
            
            arm_l.rm_set_gripper_position(l_grip_pos, False, 1)
            arm_r.rm_set_gripper_position(r_grip_pos, False, 1)

            # 打印监控
            elapsed = time.perf_counter() - loop_start
            fps = 1.0 / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:4.1f} | L_Diff: {l_diff:6.2f} | R_Diff: {r_diff:6.2f} | 指令: {r_cmd[0]:.2f}")
            
            # 🚨【关键】控制执行频率，防止动作过快导致抖动
            # 目标频率 30Hz（与数据采集频率一致）
            TARGET_DT = 1.0 / 30.0
            sleep_time = TARGET_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n👋 停止运行")
    finally:
        arm_l.rm_delete_robot_arm()
        arm_r.rm_delete_robot_arm()
        for cap in caps.values():
            cap.disconnect()

if __name__ == "__main__":
    main()