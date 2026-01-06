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

# 

# import torch
# import time
# import numpy as np
# import os
# import cv2
# import json
# import safetensors.torch
# from pathlib import Path

# from transformers import BartTokenizerFast
# from lerobot.configs.policies import PreTrainedConfig

# try:
#     from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
# except ImportError:
#     from lerobot.policies.pi0.modeling_pi0 import PI0Policy as XVLAPolicy

# from lerobot.cameras.realsense.camera_realsense import RealSenseCamera, RealSenseCameraConfig
# from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

# # ======================= [配置中心] =======================
# MODEL_PATH = "/home/robot/lerobot-main/outputs/train/008000/pretrained_model"
# LEFT_IP = "169.254.128.18"
# RIGHT_IP = "169.254.128.19"
# TASK_INSTRUCTION = "Pick the bottle to the basket_soda"

# CAMERAS_CONFIG = {
#     "image_top": "346522073032",
#     "image_left_wrist": "243722073715",
#     "image_right_wrist": "346522074543",
# }
# TARGET_KEYS = {
#     "image_top": "observation.images.image",
#     "image_right_wrist": "observation.images.image2",
#     "image_left_wrist": "observation.images.empty_camera_0"
# }

# # --- 调试参数 ---
# DRY_RUN = False             # ⚠️ 实战模式
# SMOOTH_FACTOR = 0.3         # 平滑系数

# # [固定参数]
# FORCE_RAD_TO_DEG = False
# FORCE_COMPACT_INDEX = False
# # ========================================================

# def load_stats(model_path, device):
#     print("🔍 正在加载反归一化数据...")
#     stats_file_name = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
#     stats_path = os.path.join(model_path, stats_file_name)
#     if not os.path.exists(stats_path):
#         stats_path = os.path.join(os.path.dirname(model_path.rstrip("/")), stats_file_name)

#     if os.path.exists(stats_path):
#         try:
#             tensors = safetensors.torch.load_file(stats_path)
#             mean, scale = None, None
#             for key in tensors.keys():
#                 if "action" in key and "mean" in key: mean = tensors[key]
#                 if "action" in key and ("scale" in key or "std" in key): scale = tensors[key]

#             if mean is not None:
#                 mean = mean.to(device=device, dtype=torch.float32)
#                 scale = scale.to(device=device, dtype=torch.float32)
#                 print(f"✅ 统计数据加载成功 | Mean Shape: {mean.shape}")
#                 return mean, scale
#         except Exception as e:
#             print(f"❌ 读取失败: {e}")

#     return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)

# # [保留] 中心裁剪函数
# def center_crop_and_resize(img, target_size=224):
#     h, w, _ = img.shape
#     min_dim = min(h, w)
#     top = (h - min_dim) // 2
#     left = (w - min_dim) // 2
#     img_cropped = img[top:top+min_dim, left:left+min_dim]
#     img_resized = cv2.resize(img_cropped, (target_size, target_size))
#     return img_resized

# def main():
#     try:
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#             print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
#         else:
#             raise Exception
#     except:
#         device = torch.device("cpu")

#     # 1. 加载统计数据
#     action_mean, action_std = load_stats(MODEL_PATH, device)

#     # 2. 加载模型
#     print(f"📦 正在加载模型...")
#     try:
#         config = PreTrainedConfig.from_pretrained(MODEL_PATH)
#         config.max_len_seq = 2048
#         config.dtype = "float32"
#         policy = XVLAPolicy(config)

#         model_file = os.path.join(MODEL_PATH, "model.safetensors")
#         state_dict = safetensors.torch.load_file(model_file)

#         pos_emb_key = "model.transformer.pos_emb"
#         if pos_emb_key in state_dict:
#             old_emb = state_dict[pos_emb_key]
#             new_emb_placeholder = policy.model.transformer.pos_emb.data.clone()
#             new_emb_placeholder[:, :old_emb.shape[1], :] = old_emb
#             state_dict[pos_emb_key] = new_emb_placeholder

#         enc_k = "model.vlm.language_model.model.encoder.embed_tokens.weight"
#         shared_k = "model.vlm.language_model.model.shared.weight"
#         if enc_k in state_dict and shared_k not in state_dict:
#             state_dict[shared_k] = state_dict[enc_k]

#         policy.load_state_dict(state_dict, strict=False)
#         policy.to(dtype=torch.float32, device=device)
#         policy.eval()
#         print("✅ 模型加载成功")
#     except Exception as e:
#         print(f"❌ 模型加载失败: {e}")
#         return

#     # 3. 分词器
#     try:
#         tokenizer = BartTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)
#         text_tokens = tokenizer(TASK_INSTRUCTION, return_tensors="pt", max_length=policy.config.tokenizer_max_length, truncation=True, padding="max_length")["input_ids"].to(device)
#     except:
#         return

#     # 4. 硬件初始化
#     arm_l = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#     arm_r = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#     handle_l = arm_l.rm_create_robot_arm(LEFT_IP, 8080)
#     handle_r = arm_r.rm_create_robot_arm(RIGHT_IP, 8080)

#     if handle_l.id == -1 or handle_r.id == -1:
#         print("❌ 机械臂连接失败")
#         return

#     caps = {}
#     for name, sn in CAMERAS_CONFIG.items():
#         try:
#             cfg = RealSenseCameraConfig(serial_number_or_name=sn, fps=30, width=424, height=240)
#             caps[name] = RealSenseCamera(cfg)
#             caps[name].connect()
#             print(f"📸 {name} 就绪")
#         except: pass

#     # 5. 变量初始化
#     _, curr_l = arm_l.rm_get_joint_degree()
#     _, curr_r = arm_r.rm_get_joint_degree()
#     l_cmd_smooth = np.array(curr_l, dtype=np.float32)
#     r_cmd_smooth = np.array(curr_r, dtype=np.float32)

#     last_l_grip_cmd = -1
#     last_r_grip_cmd = -1

#     # [关键修复] ImageNet 标准化参数 (PyTorch 标准)
#     IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
#     IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

#     print(f"\n🏁 系统启动 | DRY_RUN={DRY_RUN} | Vision=RGB+CenterCrop+Norm")
#     time.sleep(1)

#     try:
#         while True:
#             loop_start = time.perf_counter()

#             # --- State ---
#             _, joints_l = arm_l.rm_get_joint_degree()
#             _, joints_r = arm_r.rm_get_joint_degree()

#             pad_6 = np.zeros(6)
#             state_data = np.concatenate([
#                 np.array(joints_l), [0],  # Index 0-6
#                 pad_6,                    # Index 7-12
#                 np.array(joints_r), [0],  # Index 13-19
#                 pad_6                     # Index 20-25
#             ])

#             full_state = torch.from_numpy(state_data).float().unsqueeze(0).to(device)
#             batch = { "observation.state": full_state, "observation.language.tokens": text_tokens }

#             vis_frames = []

#             # --- 视觉处理核心循环 ---
#             for name, cap in caps.items():
#                 frame = cap.read() # OpenCV 返回 BGR
#                 if frame is not None:
#                     # 1. [关键] 中心裁剪 (保持物体比例)
#                     img_cropped = center_crop_and_resize(frame, 224)

#                     # 2. [给人类] 可视化 (BGR)
#                     vis_frames.append(img_cropped)

#                     # 3. [关键] BGR 转 RGB (给模型)
#                     img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)

#                     # 4. [关键] 转 Tensor 并 ImageNet 标准化
#                     img_tensor = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
#                     img_tensor = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD

#                     batch[TARGET_KEYS[name]] = img_tensor
#                 else:
#                     batch[TARGET_KEYS[name]] = torch.zeros((1,3,224,224)).to(device)
#                     vis_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

#             # --- 显示可视化窗口 (检查颜色和比例) ---
#             if len(vis_frames) > 0:
#                 # 这里显示的是 BGR，所以橘色应该是橘色
#                 cv2.imshow("Corrected View (BGR for Humans)", np.hstack(vis_frames))
#                 if cv2.waitKey(1) & 0xFF == ord('q'): break

#             # --- 推理 ---
#             try:
#                 with torch.no_grad():
#                     raw_action = policy.select_action(batch).squeeze(0)
#                     if raw_action.shape[0] != action_mean.shape[0]:
#                         curr_mean = action_mean[:raw_action.shape[0]]
#                         curr_std = action_std[:raw_action.shape[0]]
#                     else:
#                         curr_mean = action_mean
#                         curr_std = action_std

#                     real_action = raw_action * curr_std + curr_mean
#                     output_action = real_action.float().cpu().numpy()
#             except Exception as e:
#                 print(f"❌ 推理崩溃: {e}")
#                 break

#             # --- 解析 ---
#             target_l = output_action[0:6]
#             l_grip_raw = output_action[6]
#             target_r = output_action[13:19]
#             r_grip_raw = output_action[19]

#             l_cmd_smooth = l_cmd_smooth * (1 - SMOOTH_FACTOR) + target_l * SMOOTH_FACTOR
#             r_cmd_smooth = r_cmd_smooth * (1 - SMOOTH_FACTOR) + target_r * SMOOTH_FACTOR

#             # [关键] 夹爪放大系数调整：从 *100 改为 *10
#             # 假设 Raw 输出在 0-100 之间，乘以 10 映射到 0-1000
#             l_grip_pos = int(np.clip(l_grip_raw * 10, 0, 1000))
#             r_grip_pos = int(np.clip(r_grip_raw * 10, 0, 1000))

#             # --- 打印 ---
#             # fps = 1.0 / (time.perf_counter() - loop_start)
#             fps = 30.0
#             print(f"\rFPS: {fps:.1f} | L_Grip: {l_grip_pos:<4} | R_Grip: {r_grip_pos:<4} | R_J1: {r_cmd_smooth[0]:.1f}", end="")

#             # --- 执行 ---
#             if not DRY_RUN:
#                 arm_l.rm_movej_canfd(l_cmd_smooth.tolist(), False, 0, 0, 0)
#                 arm_r.rm_movej_canfd(r_cmd_smooth.tolist(), False, 0, 0, 0)

#                 if abs(l_grip_pos - last_l_grip_cmd) > 20:
#                     arm_l.rm_set_gripper_position(l_grip_pos, False, 1)
#                     last_l_grip_cmd = l_grip_pos

#                 if abs(r_grip_pos - last_r_grip_cmd) > 20:
#                     arm_r.rm_set_gripper_position(r_grip_pos, False, 1)
#                     last_r_grip_cmd = r_grip_pos
#             else:
#                 time.sleep(0.05)

#     except KeyboardInterrupt:
#         print("\n👋 停止运行")
#     finally:
#         arm_l.rm_delete_robot_arm()
#         arm_r.rm_delete_robot_arm()
#         for cap in caps.values():
#             cap.disconnect()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import torch
import time
import numpy as np
import os
import cv2
import safetensors.torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from transformers import BartTokenizerFast
from lerobot.configs.policies import PreTrainedConfig
try:
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
except ImportError:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy as XVLAPolicy

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera, RealSenseCameraConfig
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

# ================= 配置类 =================
@dataclass
class InferenceConfig:
    # 路径配置
    model_path: str = "/home/robot/lerobot-main/outputs/train/008000/pretrained_model"
    
    # 硬件配置
    left_ip: str = "169.254.128.18"
    right_ip: str = "169.254.128.19"
    cameras_config: Dict[str, str] = None
    
    # 任务配置
    task_instruction: str = "Pick the bottle to the basket_soda"
    
    # 平滑系数
    smooth_factor: float = 0.3
    dry_run: bool = True
    loop_rate: float = 30.0
    
    def __post_init__(self):
        if self.cameras_config is None:
            self.cameras_config = {
                "image_top": "346522073032",
                "image_left_wrist": "243722073715",
                "image_right_wrist": "346522074543",
            }

# ================= 推理核心类 =================
class Inference:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 设备: {self.device}")
        
        # 1. 加载统计数据
        self.action_mean, self.action_std = self._load_output_stats()
        self.state_mean, self.state_std = self._load_input_stats()

        # 2. 加载模型
        self.policy = self._load_model()
        self.text_tokens = self._load_tokenizer()
        
        # 3. 初始化硬件
        self.arm_l, self.arm_r = self._init_robots()
        self.caps = self._init_cameras()
        self._init_runtime_state()
        
        # 图像归一化标准 (0.5)
        self.IMG_MEAN = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self.IMG_STD = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        
        self.target_keys = {
            "image_top": "observation.images.image",
            "image_right_wrist": "observation.images.image2",
            "image_left_wrist": "observation.images.empty_camera_0"
        }

    def _load_output_stats(self):
        print("🔍 加载输出统计 (Action)...")
        stats_file = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        paths = [
            os.path.join(self.cfg.model_path, stats_file),
            os.path.join(os.path.dirname(self.cfg.model_path.rstrip("/")), stats_file)
        ]
        
        for path in paths:
            if os.path.exists(path):
                try:
                    tensors = safetensors.torch.load_file(path)
                    mean, scale = None, None
                    # 暴力查找 Key
                    for key in tensors.keys():
                        if "action" in key and "mean" in key: mean = tensors[key]
                        if "action" in key and ("scale" in key or "std" in key): scale = tensors[key]
                    
                    if mean is not None:
                        mean = mean.to(self.device, dtype=torch.float32)
                        scale = scale.to(self.device, dtype=torch.float32)
                        # 确保是 [1, 26]
                        if mean.ndim == 1: mean = mean.unsqueeze(0)
                        if scale.ndim == 1: scale = scale.unsqueeze(0)
                        print(f"✅ 输出统计加载成功 | Shape: {mean.shape}")
                        return mean, scale
                except Exception as e:
                    print(f"❌ 输出统计读取失败: {e}")
        
        print("⚠️ 未找到输出统计！使用默认值")
        return torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)

    def _load_input_stats(self):
        print("🔍 加载输入统计 (State)...")
        stats_file = "policy_preprocessor_step_7_normalizer_processor.safetensors"
        paths = [
            os.path.join(self.cfg.model_path, stats_file),
            os.path.join(os.path.dirname(self.cfg.model_path.rstrip("/")), stats_file)
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    tensors = safetensors.torch.load_file(path)
                    mean, scale = None, None
                    for key in tensors.keys():
                        if "observation.state" in key and "mean" in key: mean = tensors[key]
                        if "observation.state" in key and ("scale" in key or "std" in key): scale = tensors[key]

                    if mean is not None:
                        mean = mean.to(self.device, dtype=torch.float32)
                        scale = scale.to(self.device, dtype=torch.float32)
                        if mean.ndim == 1: mean = mean.unsqueeze(0)
                        if scale.ndim == 1: scale = scale.unsqueeze(0)
                        print(f"✅ 输入统计加载成功 | Mean: {mean[0,:6].cpu().numpy()}")
                        return mean, scale
                except Exception as e: print(f"❌ 失败: {e}")
        print("⚠️ 未找到输入统计！")
        return torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)

    def _load_model(self):
        print(f"📦 加载模型...")
        config = PreTrainedConfig.from_pretrained(self.cfg.model_path)
        config.max_len_seq = 2048
        config.dtype = "float32"
        policy = XVLAPolicy(config)
        state_dict = safetensors.torch.load_file(os.path.join(self.cfg.model_path, "model.safetensors"))
        self._patch_state_dict(policy, state_dict)
        policy.load_state_dict(state_dict, strict=False)
        policy.to(dtype=torch.float32, device=self.device).eval()
        return policy

    def _patch_state_dict(self, policy, state_dict):
        pos_key = "model.transformer.pos_emb"
        if pos_key in state_dict:
            old = state_dict[pos_key]
            new = policy.model.transformer.pos_emb.data.clone()
            if old.shape != new.shape:
                new[:, :old.shape[1], :] = old
                state_dict[pos_key] = new
        enc_k = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shared_k = "model.vlm.language_model.model.shared.weight"
        if enc_k in state_dict and shared_k not in state_dict: state_dict[shared_k] = state_dict[enc_k]

    def _load_tokenizer(self):
        try:
            tokenizer = BartTokenizerFast.from_pretrained(self.cfg.model_path, local_files_only=True)
            return tokenizer(self.cfg.task_instruction, return_tensors="pt", 
                max_length=self.policy.config.tokenizer_max_length, 
                truncation=True, padding="max_length")["input_ids"].to(self.device)
        except: return None

    def _init_robots(self):
        arm_l = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        arm_r = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        hl = arm_l.rm_create_robot_arm(self.cfg.left_ip, 8080)
        hr = arm_r.rm_create_robot_arm(self.cfg.right_ip, 8080)
        if hl.id == -1 or hr.id == -1: raise RuntimeError("❌ 连接失败")
        return arm_l, arm_r

    def _init_cameras(self):
        caps = {}
        for name, sn in self.cfg.cameras_config.items():
            try:
                cfg = RealSenseCameraConfig(serial_number_or_name=sn, fps=30, width=640, height=480)
                cam = RealSenseCamera(cfg)
                cam.connect()
                caps[name] = cam
                print(f"📸 {name} 就绪")
            except: pass
        return caps

    def _init_runtime_state(self):
        _, curr_l = self.arm_l.rm_get_joint_degree()
        _, curr_r = self.arm_r.rm_get_joint_degree()
        self.cmd_smooth_l = np.array(curr_l, dtype=np.float32)
        self.cmd_smooth_r = np.array(curr_r, dtype=np.float32)
        self.last_grip_l = -1
        self.last_grip_r = -1

    def _process_image(self, img_bgr):
        h, w, _ = img_bgr.shape
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        img = img_bgr[top:top+min_dim, left:left+min_dim]
        img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0).to(self.device) / 255.0
        tensor = (tensor - self.IMG_MEAN) / self.IMG_STD
        return tensor, img

    def get_observation(self):
        _, jl = self.arm_l.rm_get_joint_degree()
        _, jr = self.arm_r.rm_get_joint_degree()
        
        jl_np = np.array(jl)
        jr_np = np.array(jr)
        pad = np.zeros(6)
        state = np.concatenate([jl_np, [0], pad, jr_np, [0], pad])
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # 归一化
        normalized_state = (state_tensor - self.state_mean) / self.state_std
        
        batch = {
            "observation.state": normalized_state,
            "observation.language.tokens": self.text_tokens
        }
        
        vis_frames = []
        for name, cap in self.caps.items():
            frame = cap.read()
            if frame is not None:
                tensor, vis_img = self._process_image(frame)
                batch[self.target_keys[name]] = tensor
                vis_frames.append(vis_img)
            else:
                batch[self.target_keys[name]] = torch.zeros((1,3,224,224), device=self.device)
                vis_frames.append(np.zeros((224,224,3), dtype=np.uint8))
        return batch, vis_frames

    def execute_action(self, action):
        target_l = action[0:6]
        target_r = action[13:19]
        grip_l = int(np.clip(action[6] * 10, 0, 1000))
        grip_r = int(np.clip(action[19] * 10, 0, 1000))
        
        self.cmd_smooth_l = self.cmd_smooth_l * (1 - self.cfg.smooth_factor) + target_l * self.cfg.smooth_factor
        self.cmd_smooth_r = self.cmd_smooth_r * (1 - self.cfg.smooth_factor) + target_r * self.cfg.smooth_factor
        
        print(f"\rL_Grip: {grip_l:<4} | R_Grip: {grip_r:<4} | R_J1: {self.cmd_smooth_r[0]:.1f}", end="")
        
        if not self.cfg.dry_run:
            self.arm_l.rm_movej_canfd(self.cmd_smooth_l.tolist(), False, 0, 0, 0)
            self.arm_r.rm_movej_canfd(self.cmd_smooth_r.tolist(), False, 0, 0, 0)
            if abs(grip_l - self.last_grip_l) > 20:
                self.arm_l.rm_set_gripper_position(grip_l, False, 1)
                self.last_grip_l = grip_l
            if abs(grip_r - self.last_grip_r) > 20:
                self.arm_r.rm_set_gripper_position(grip_r, False, 1)
                self.last_grip_r = grip_r

    def run(self):
        print(f"\n🏁 开始推理循环 | Rate: {self.cfg.loop_rate}Hz")
        while True:
            loop_start = time.perf_counter()
            batch, vis_frames = self.get_observation()
            
            if vis_frames:
                cv2.imshow("Robot View", np.hstack(vis_frames))
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            with torch.no_grad():
                raw_action = self.policy.select_action(batch).squeeze(0)
                
                # ----------------------------------------------------
                # 🛡️ 调试 & 暴力修复区
                # ----------------------------------------------------
                # 如果这一行打印出来了，说明代码更新成功了！
                # print(f"DEBUG SHAPE: Action={raw_action.shape} | Stats={self.action_mean.shape}")

                # 暴力修正：只要最后一个维度不是 26，就一定是维度反了，立刻转置！
                # 针对 [26, 20] 这种情况
                if raw_action.ndim > 1 and raw_action.shape[-1] != 26:
                    raw_action = raw_action.t()
                
                # 针对可能的 [26] (1D) 情况，强制升维到 [1, 26] 以防万一
                if raw_action.ndim == 1 and raw_action.shape[0] == 26:
                    raw_action = raw_action.unsqueeze(0)

                # ----------------------------------------------------
                
                curr_mean = self.action_mean
                curr_std = self.action_std
                
                real_action = raw_action * curr_std + curr_mean
                action_numpy = real_action.float().cpu().numpy()
            
            # 取第一帧执行
            # 如果 raw_action 变成了 [20, 26]，取 [0] 就是第一帧 [26]
            self.execute_action(action_numpy[0])
            
            time.sleep(max(0, (1.0 / self.cfg.loop_rate) - (time.perf_counter() - loop_start)))

    def cleanup(self):
        print("\n🧹 清理...")
        self.arm_l.rm_delete_robot_arm()
        self.arm_r.rm_delete_robot_arm()
        for cap in self.caps.values(): cap.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config = InferenceConfig(task_instruction="Pick the bottle to the basket_soda")
    inference = Inference(config)
    inference.run()