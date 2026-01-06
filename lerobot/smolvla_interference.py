import torch
import time
import numpy as np
import os
import cv2
import json
import safetensors.torch
from pathlib import Path

from transformers import BartTokenizerFast, AutoTokenizer
from lerobot.configs.policies import PreTrainedConfig

# ======================= [配置中心] =======================
MODEL_PATH = "/home/robot/lerobot-main/outputs/train/020000/pretrained_model"
LEFT_IP = "169.254.128.18"
RIGHT_IP = "169.254.128.19"
TASK_INSTRUCTION = "把...放进..._打开洗衣机取出衣服_把衣服放到洗衣机里"

CAMERAS_CONFIG = {
    "image_top": "346522073032",
    "image_left_wrist": "243722073715",
    "image_right_wrist": "346522074543",
}
TARGET_KEYS = {
    "image_top": "observation.images.image",
    "image_right_wrist": "observation.images.image2",
    "image_left_wrist": "observation.images.empty_camera_0"
}

# --- 调试参数 ---
DRY_RUN = True             
SMOOTH_FACTOR = 0.3         
TARGET_IMAGE_SIZE = 384     # SigLIP 默认 384

FORCE_RAD_TO_DEG = False     
FORCE_COMPACT_INDEX = False  
# ========================================================

# --- 动态导入策略类 ---
# 为了防止 LeRobot 版本差异，我们尝试导入所有可能的类
try:
    from lerobot.policies.factory import make_policy
    USE_FACTORY = True
except ImportError:
    try:
        from lerobot.policies.factory import make_policy
        USE_FACTORY = True
    except ImportError:
        USE_FACTORY = False

# 尝试导入具体的策略类作为备选
PolicyClasses = {}
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    PolicyClasses['SmolVLAConfig'] = SmolVLAPolicy
except ImportError:
    pass

try:
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    PolicyClasses['XVLAConfig'] = XVLAPolicy
except ImportError:
    pass

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera, RealSenseCameraConfig
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


def load_stats(model_path, device):
    print("🔍 正在加载反归一化数据...")
    stats_file_name = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    stats_path = os.path.join(model_path, stats_file_name)
    if not os.path.exists(stats_path):
        stats_path = os.path.join(os.path.dirname(model_path.rstrip("/")), stats_file_name)

    if os.path.exists(stats_path):
        try:
            tensors = safetensors.torch.load_file(stats_path)
            mean, scale = None, None
            for key in tensors.keys():
                if "action" in key and "mean" in key: mean = tensors[key]
                if "action" in key and ("scale" in key or "std" in key): scale = tensors[key]

            if mean is not None:
                mean = mean.to(device=device, dtype=torch.float32)
                scale = scale.to(device=device, dtype=torch.float32)
                print(f"✅ 统计数据加载成功 | Mean Shape: {mean.shape}")
                return mean, scale
        except Exception as e:
            print(f"❌ 读取失败: {e}")
    
    return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)

def center_crop_and_resize(img, target_size):
    h, w, _ = img.shape
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    img_cropped = img[top:top+min_dim, left:left+min_dim]
    img_resized = cv2.resize(img_cropped, (target_size, target_size))
    return img_resized

def main():
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        else:
            raise Exception
    except:
        device = torch.device("cpu")

    action_mean, action_std = load_stats(MODEL_PATH, device)

    print(f"📦 正在加载 SmolVLA 模型...")
    policy = None
    
    # --- 1. 优先尝试 Factory 加载 (自动识别类) ---
    if USE_FACTORY:
        try:
            # [修正] 直接传路径作为位置参数，兼容性最好
            policy = make_policy(MODEL_PATH)
            policy.to(device)
            policy.eval()
            print("✅ 通过 Factory 加载成功")
        except Exception as e:
            print(f"⚠️ Factory 加载失败 ({e})，尝试手动加载...")
            policy = None

    # --- 2. 手动加载回退 ---
    if policy is None:
        try:
            config = PreTrainedConfig.from_pretrained(MODEL_PATH)
            config_class_name = config.__class__.__name__
            print(f"ℹ️ 识别到配置类: {config_class_name}")

            # 根据 Config 类型选择正确的 Policy 类
            if config_class_name in PolicyClasses:
                PolicyClass = PolicyClasses[config_class_name]
                print(f"ℹ️ 使用策略类: {PolicyClass.__name__}")
            else:
                # 最后的尝试：如果找不到对应的类，且名字里有 Smol，尝试动态导入
                if 'Smol' in config_class_name:
                     from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as PolicyClass
                else:
                     from lerobot.policies.xvla.modeling_xvla import XVLAPolicy as PolicyClass
            
            # 更新分辨率
            global TARGET_IMAGE_SIZE
            if hasattr(config, "image_size") and config.image_size:
                TARGET_IMAGE_SIZE = config.image_size
                print(f"ℹ️ Config Image Size: {TARGET_IMAGE_SIZE}")

            policy = PolicyClass(config)
            
            model_file = os.path.join(MODEL_PATH, "model.safetensors")
            state_dict = safetensors.torch.load_file(model_file)
            
            # 权重修复逻辑
            if "model.transformer.pos_emb" in state_dict:
                old_emb = state_dict["model.transformer.pos_emb"]
                new_emb = policy.model.transformer.pos_emb.data.clone()
                new_emb[:, :old_emb.shape[1], :] = old_emb
                state_dict["model.transformer.pos_emb"] = new_emb
            
            policy.load_state_dict(state_dict, strict=False)
            policy.to(dtype=torch.float32, device=device)
            policy.eval()
            print("✅ 手动加载成功")
        except Exception as e:
            print(f"❌ 致命错误: 模型加载完全失败: {e}")
            return

    # 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        max_len = getattr(policy.config, "tokenizer_max_length", 128) 
        text_tokens = tokenizer(TASK_INSTRUCTION, return_tensors="pt", max_length=max_len, truncation=True, padding="max_length")["input_ids"].to(device)
    except:
        print("❌ 分词器加载失败")
        return

    # 硬件连接
    arm_l = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    arm_r = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle_l = arm_l.rm_create_robot_arm(LEFT_IP, 8080)
    handle_r = arm_r.rm_create_robot_arm(RIGHT_IP, 8080)
    
    if handle_l.id == -1 or handle_r.id == -1:
        print("❌ 机械臂连接失败")
        return

    caps = {}
    for name, sn in CAMERAS_CONFIG.items():
        try:
            cfg = RealSenseCameraConfig(serial_number_or_name=sn, fps=30, width=640, height=480)
            caps[name] = RealSenseCamera(cfg)
            caps[name].connect()
            print(f"📸 {name} 就绪")
        except: pass

    _, curr_l = arm_l.rm_get_joint_degree()
    _, curr_r = arm_r.rm_get_joint_degree()
    l_cmd_smooth = np.array(curr_l, dtype=np.float32)
    r_cmd_smooth = np.array(curr_r, dtype=np.float32)
    
    last_l_grip_cmd = -1 
    last_r_grip_cmd = -1

    # SmolVLA/SigLIP 归一化参数 (Mean=0.5, Std=0.5)
    SIGLIP_MEAN = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    SIGLIP_STD = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    print(f"\n🏁 系统启动 | SMOOTH={SMOOTH_FACTOR} | Size={TARGET_IMAGE_SIZE}")
    time.sleep(1)

    try:
        while True:
            loop_start = time.perf_counter()
            
            _, joints_l = arm_l.rm_get_joint_degree() 
            _, joints_r = arm_r.rm_get_joint_degree()
            
            # State
            pad_6 = np.zeros(6)
            state_data = np.concatenate([
                np.array(joints_l), [0],  # 0-6
                pad_6,                    # 7-12
                np.array(joints_r), [0],  # 13-19
                pad_6                     # 20-25
            ])

            full_state = torch.from_numpy(state_data).float().unsqueeze(0).to(device)
            batch = { "observation.state": full_state, "observation.language.tokens": text_tokens }
            
            vis_frames = []
            for name, cap in caps.items():
                frame = cap.read()
                if frame is not None:
                    # 1. Center Crop + Resize
                    img = center_crop_and_resize(frame, TARGET_IMAGE_SIZE)
                    vis_frames.append(img)
                    
                    # 2. BGR -> RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 3. Norm (SigLIP style)
                    img_tensor = torch.from_numpy(img_rgb).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
                    img_tensor = (img_tensor - SIGLIP_MEAN) / SIGLIP_STD
                    
                    batch[TARGET_KEYS[name]] = img_tensor
                else:
                    batch[TARGET_KEYS[name]] = torch.zeros((1,3,TARGET_IMAGE_SIZE,TARGET_IMAGE_SIZE)).to(device)
                    vis_frames.append(np.zeros((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3), dtype=np.uint8))

            if len(vis_frames) > 0:
                debug_view = cv2.resize(np.hstack(vis_frames), (224*3, 224))
                cv2.imshow("SmolVLA View", debug_view)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            # 推理
            try:
                with torch.no_grad():
                    raw_action = policy.select_action(batch).squeeze(0)
                    if raw_action.shape[0] != action_mean.shape[0]:
                        limit = min(raw_action.shape[0], action_mean.shape[0])
                        curr_mean, curr_std = action_mean[:limit], action_std[:limit]
                        raw_action = raw_action[:limit]
                    else:
                        curr_mean, curr_std = action_mean, action_std

                    real_action = raw_action * curr_std + curr_mean
                    output_action = real_action.float().cpu().numpy()
            except Exception as e:
                print(f"❌ 推理崩溃: {e}")
                break

            target_l = output_action[0:6]
            l_grip_raw = output_action[6]
            target_r = output_action[13:19]
            r_grip_raw = output_action[19]

            l_cmd_smooth = l_cmd_smooth * (1 - SMOOTH_FACTOR) + target_l * SMOOTH_FACTOR
            r_cmd_smooth = r_cmd_smooth * (1 - SMOOTH_FACTOR) + target_r * SMOOTH_FACTOR
            
            l_grip_pos = int(np.clip(l_grip_raw * 10, 0, 1000))
            r_grip_pos = int(np.clip(r_grip_raw * 10, 0, 1000))

            fps = 1.0 / (time.perf_counter() - loop_start)
            print(f"\rFPS: {fps:.1f} | L: {l_grip_pos:<4} | R: {r_grip_pos:<4}", end="")

            if not DRY_RUN:
                arm_l.rm_movej_canfd(l_cmd_smooth.tolist(), False, 0, 0, 0)
                arm_r.rm_movej_canfd(r_cmd_smooth.tolist(), False, 0, 0, 0)
                
                if abs(l_grip_pos - last_l_grip_cmd) > 20: 
                    arm_l.rm_set_gripper_position(l_grip_pos, False, 1)
                    last_l_grip_cmd = l_grip_pos

                if abs(r_grip_pos - last_r_grip_cmd) > 20:
                    arm_r.rm_set_gripper_position(r_grip_pos, False, 1)
                    last_r_grip_cmd = r_grip_pos
            else:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n👋 停止运行")
    finally:
        arm_l.rm_delete_robot_arm()
        arm_r.rm_delete_robot_arm()
        for cap in caps.values(): cap.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()