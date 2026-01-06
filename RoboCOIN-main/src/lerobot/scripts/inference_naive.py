import draccus
import imageio
import numpy as np
import os
import threading
import time
import torch
import traceback
from dataclasses import dataclass, field
from sshkeyboard import listen_keyboard, stop_listening
from typing import List

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.server.helpers import (
    map_robot_keys_to_lerobot_features,
    raw_observation_to_observation,
)
from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.cameras.dummy.configuration_dummy import DummyCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots import (
    dummy,
    bi_dummy,
    piper,
    bi_piper,
    realman,
    bi_realman,
)


@dataclass
class LocalRobotClientConfig:
    robot: RobotConfig
    task: str = "place the bowl on the plate"
    pretrained_path: str = ""
    device: str = "cuda:0"
    repo_id: str = "piper/place_the_bowl_on_the_plate_filtered"
    result_dir: str = "results/"
    frequency: int = 30
    camera_keys: List[str] = field(default_factory=lambda: [
        'front'
    ])
    

class VideoRecorder:
    def __init__(
        self,
        save_dir,
        fps: int = 30,
    ):
        self.save_dir = save_dir
        self.fps = fps
        self._frames = []

        os.makedirs(self.save_dir, exist_ok=True)

    def add(self, frame):
        if isinstance(frame, list):
            # [(H, W, C), ...] -> (H, W * N, C)
            frame = np.concatenate(frame, axis=1)
        self._frames.append(frame)
    
    def save(self, task, success):
        save_path = os.path.join(self.save_dir, f"{task.replace('.', '')}_{'success' if success else 'failed'}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        print(f'Saving video to {save_path}...')
        imageio.mimwrite(save_path, self._frames, fps=self.fps)
        self._frames = []


class KeyboardListener:
    def __init__(self):
        self._listener = threading.Thread(target=listen_keyboard, args=(self._on_press,))
        self._listener.daemon = True

        self._quit = False
        self._success = None
    
    def listen(self):
        self._listener.start()
    
    def reset(self):
        self._quit = False
        self._success = None
    
    def _on_press(self, key):
        if key == 'q':
            self._quit = True
        
        elif key == 'y':
            self._success = True
            stop_listening()
        
        elif key == 'n':
            self._success = False
            stop_listening()


class LocalRobotClient:
#    def __init__(self, config: LocalRobotClientConfig):
#       self.config = config
#
 #       self.video_recorder = VideoRecorder(config.result_dir, fps=config.frequency)
  #      self.keyboard_listener = KeyboardListener()
#
 #       # self.dataset = LeRobotDataset(repo_id=config.repo_id)
  #      from lerobot.policies.factory import get_policy_class
   #     
    #    print(f"🚀 直接加载预训练模型: {config.pretrained_path}")
     #   
      #  # 1. 获取策略类 (如 ACTPolicy)
       # policy_cls = get_policy_class(policy_config.type)
        
        ## 2. 直接从路径加载 (自动读取 config.json 和 model.safetensors)
       # self.policy = policy_cls.from_pretrained(config.pretrained_path)
        
        ## 3. 移动到设备并设为评估模式
       # self.policy.to(policy_config.device)
        #self.policy.eval()
        
       # print("✅ 策略加载成功！(跳过了数据集元数据检查)")
        #policy_config = PreTrainedConfig.from_pretrained(self.config.pretrained_path)
        #policy_config.pretrained_path = self.config.pretrained_path
       # self.policy = make_policy(policy_config, ds_meta=self.dataset.meta)
        #self.policy.to(config.device)
#
 #       self.robot = make_robot_from_config(config.robot)
#
 #       self._is_finished = False
    def __init__(self, config: LocalRobotClientConfig):
        self.config = config
        self.video_recorder = VideoRecorder(config.result_dir, fps=config.frequency)
        self.keyboard_listener = KeyboardListener()

        # =========================================================
        # ⚡️ 终极修复 V4：智能键名映射 + 强制 CPU
        # =========================================================
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class
        import safetensors.torch
        import torch
        from pathlib import Path

        print(f"🚀 准备加载模型: {config.pretrained_path}")
        model_dir = Path(config.pretrained_path)
        
        # 1. 加载配置
        policy_config = PreTrainedConfig.from_pretrained(config.pretrained_path)
        policy_config.pretrained_path = config.pretrained_path
        
        # 2. 初始化策略
        print(f"ℹ️ 策略类型: {policy_config.type}")
        policy_cls = get_policy_class(policy_config.type)
        self.policy = policy_cls.from_pretrained(config.pretrained_path)
        
        # --- 🔧 辅助函数：键名重映射 ---
        def remap_keys(state_dict, module_prefix):
            new_sd = {}
            for k, v in state_dict.items():
                # 解析后缀 (mean/std/min/max)
                if k.endswith(".mean"): stat, suffix_len = "mean", 5
                elif k.endswith(".std"): stat, suffix_len = "std", 4
                elif k.endswith(".min"): stat, suffix_len = "min", 4
                elif k.endswith(".max"): stat, suffix_len = "max", 4
                else: continue
                
                # 获取特征名 (去除后缀)
                feature_name = k[:-suffix_len] # e.g. "observation.images.image_top"
                
                # 转换格式: 点 -> 下划线, 增加 buffer_ 前缀
                feature_slug = feature_name.replace(".", "_")
                new_key = f"{module_prefix}.buffer_{feature_slug}.{stat}"
                
                print(f"   映射: {k} -> {new_key}")
                new_sd[new_key] = v
            return new_sd
        # -----------------------------------

        # 3. 加载并打补丁
        print("🔧 正在智能修复统计数据...")
        
        # (A) 输入统计 -> normalize_inputs
        pre_files = list(model_dir.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
        if pre_files:
            print(f"   -> 加载输入统计: {pre_files[0].name}")
            pre_sd = safetensors.torch.load_file(pre_files[0])
            patch_sd = remap_keys(pre_sd, "normalize_inputs")
            self.policy.load_state_dict(patch_sd, strict=False)
        
        # (B) 输出统计 -> unnormalize_outputs
        post_files = list(model_dir.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
        if post_files:
            print(f"   -> 加载输出统计: {post_files[0].name}")
            post_sd = safetensors.torch.load_file(post_files[0])
            patch_sd = remap_keys(post_sd, "unnormalize_outputs")
            self.policy.load_state_dict(patch_sd, strict=False)

        # 4. 强制 CPU (RTX 5080 兼容)
        print("⚠️ 强制切换至 CPU 模式...")
        self.policy.to("cpu")
        self.policy.eval()
        
        print("✅ 策略加载完毕！")
        # =========================================================

        self.robot = make_robot_from_config(config.robot)
        self._is_finished = False
    
    
    
    
    def start(self):
        self.keyboard_listener.listen()
        self.robot.connect()
        time.sleep(5)
    
    def control_loop(self):
        # while not self._is_finished:
        #     obs = self._prepare_observation(self.robot.get_observation())
        #     with torch.inference_mode():
        #         action = self.policy.select_action(obs)[0]
        #     obs = self.robot.get_observation()
        #     state = None
        #     action = self._prepare_action(action, state)
        #   print('Prepared action:', action)
        self.robot.visualizer = None
        
        while not self._is_finished:
            start_time = time.perf_counter()
            
            # 1. 获取观测
            obs = self._prepare_observation(self.robot.get_observation())
            
            # 2. 模型推理 -> 得到 26维 Tensor
            action = self.policy.select_action(obs)[0]
            
            # 3. 🔥 核心修复: Tensor 转 Dictionary
            # BiBaseRobot 需要字典才能分发左右臂动作
            # 我们根据之前的分析：左臂是前7个，右臂是第13-20个 (跳过中间的位姿)
            
            # 转为 CPU numpy 数组方便处理
            act_np = action.to("cpu").numpy()
            
            action_dict = {}
            
            # --- 左臂 (Index 0-6) ---
            # 我们直接去读 robot.left_robot.motor_names 或者 _motors_ft.keys()
            # 这样绝对不会错！
            try:
                # 尝试获取电机映射键名
                left_motor_keys = list(self.robot.left_robot._motors_ft.keys())
                right_motor_keys = list(self.robot.right_robot._motors_ft.keys())
            except AttributeError:
                # 如果是旧版代码没有 _motors_ft，回退到 joint_names
                left_motor_keys = self.robot.left_robot.config.joint_names
                right_motor_keys = self.robot.right_robot.config.joint_names

            # 填充左臂数据
            for i, key_name in enumerate(left_motor_keys):
                # 如果 key_name 已经是 "joint_1_pos"，那加上 "left_" 就是 "left_joint_1_pos"
                # 这正是 BiBaseRobot 剥离前缀后想要的名字
                action_dict[f"left_{key_name}"] = act_np[i]
                
            # 填充右臂数据 (从 index 13 开始)
            for i, key_name in enumerate(right_motor_keys):
                action_dict[f"right_{key_name}"] = act_np[13 + i]
            
            # 4. 发送字典给机器人
            # 此时 action_dict 是一个包含所有关节数据的字典，BiBaseRobot 能读懂
            self.robot.send_action(action_dict)
            # self.robot.send_action(action)
            self._after_action()
            time.sleep(1 / self.config.frequency)

    def stop(self):
        self.robot.disconnect()
    
    # def _prepare_observation(self, observation):
    #     observation['task'] = self.config.task
    #     observation = raw_observation_to_observation(
    #         observation, 
    #         map_robot_keys_to_lerobot_features(self.robot),
    #         self.policy.config.image_features,
    #         device=self.config.device,
    #     )
        
    #     import torch
    #     state = observation["observation.state"]
        
    #     # 检查是否是 14 维 (2臂 * 7关节)
    #     if state.shape[-1] == 14:
    #         print("⚠️ 检测到状态维度为 14，正在自动补齐 12 维速度数据...")
    #         batch_size = state.shape[0] if state.dim() > 1 else 1
    #         device = state.device
    #         dtype = state.dtype
            
    #         # 构造 12 个 0 (假设两臂各 6 个速度)
    #         zeros = torch.zeros((12,), device=device, dtype=dtype)
    #         if state.dim() > 1:
    #             zeros = zeros.unsqueeze(0).repeat(batch_size, 1)
                
    #         # 这里的拼接顺序很重要！通常 ACT 是 [left_pos, left_vel, right_pos, right_vel]
    #         # 现在的 14 维是 [left_pos, right_pos] (假设顺序)
    #         # 我们需要把它拆开，分别插进去
            
    #         left_pos = state[..., :7]  # 前7个
    #         right_pos = state[..., 7:] # 后7个
            
    #         left_vel = torch.zeros((6,), device=device, dtype=dtype)
    #         right_vel = torch.zeros((6,), device=device, dtype=dtype)
            
    #         if state.dim() > 1:
    #             left_vel = left_vel.unsqueeze(0).repeat(batch_size, 1)
    #             right_vel = right_vel.unsqueeze(0).repeat(batch_size, 1)
            
    #         # 拼成 [lp, lv, rp, rv] = 7+6+7+6 = 26
    #         new_state = torch.cat([left_pos, left_vel, right_pos, right_vel], dim=-1)
            
    #         observation["observation.state"] = new_state
            
    #     return observation
        # return observation
        
        
    # def _prepare_observation(self, robot_obs):
    #     from lerobot.scripts.server.helpers import raw_observation_to_observation
        
    #     # 1. 混合对象 (HybridFeature) - 解决配置兼容性
    #     class HybridFeature(dict):
    #         def __getattr__(self, name):
    #             if name in self: return self[name]
    #             raise AttributeError(f"No attribute {name}")

    #     input_features_hybrid = {}
        
    #     for key, feature in self.policy.config.input_features.items():
    #         h_feat = HybridFeature()
    #         # 复制数据
    #         if isinstance(feature, dict):
    #             h_feat.update(feature)
    #         else:
    #             for attr in ["type", "dtype", "shape", "names"]:
    #                 if hasattr(feature, attr):
    #                     val = getattr(feature, attr)
    #                     if val is not None: h_feat[attr] = val
                            
    #         # 补全 dtype
    #         if "dtype" not in h_feat:
    #             f_type = h_feat.get("type", getattr(feature, "type", None))
    #             h_feat["dtype"] = "video" if (f_type == "image" or "image" in key) else "float32"
            
    #         # 补全 names (重要：这里我们只请求机器人能提供的 14 个数据)
    #         if key == "observation.state" and "names" not in h_feat:
    #             try:
    #                 left_names = [f"left_{n}_pos" for n in self.robot.left_robot.config.joint_names]
    #                 right_names = [f"right_{n}_pos" for n in self.robot.right_robot.config.joint_names]
    #                 h_feat["names"] = left_names + right_names
    #             except AttributeError:
    #                 base_names = [f"joint_{i+1}_pos" for i in range(6)] + ["gripper_pos"]
    #                 h_feat["names"] = [f"left_{n}" for n in base_names] + [f"right_{n}" for n in base_names]

    #         input_features_hybrid[key] = h_feat

    #     # 2. 生成初步观测值 (此时 state 是 14 维)
    #     observation = raw_observation_to_observation(
    #         robot_obs,
    #         input_features_hybrid,
    #         input_features_hybrid,
    #         "cpu" 
    #     )
        
    #     observation['task'] = self.config.task
        
    #     # 3. 强制 CPU
    #     for key in observation:
    #         if hasattr(observation[key], "to"):
    #             observation[key] = observation[key].to("cpu")
        
    #     import torch
    #     state = observation["observation.state"]
        
    #     # =========================================================
    #     # ⚡️ 核心修复：维度补齐 (14 -> 26) 
    #     # =========================================================
    #     if state.shape[-1] == 14:
    #         # print("⚠️ 检测到 14 维数据，正在自动补齐位姿数据(0)...")
    #         batch_size = state.shape[0] if state.dim() > 1 else 1
    #         device = "cpu"
    #         dtype = state.dtype
            
    #         # 切分左右臂数据 (各7维: 6关节 + 1夹爪)
    #         # 假设顺序是 [Left_Joints(6), Left_Gripper(1), Right_Joints(6), Right_Gripper(1)]
    #         left_part = state[..., :7]
    #         right_part = state[..., 7:]
            
    #         # 创建 6 维零向量 (代表缺失的位姿)
    #         zeros = torch.zeros((6,), device=device, dtype=dtype)
    #         if state.dim() > 1:
    #             zeros = zeros.unsqueeze(0).repeat(batch_size, 1)
            
    #         # 重新拼接为 26 维: [L(7), L_Pose(0), R(7), R_Pose(0)]
    #         new_state = torch.cat([left_part, zeros, right_part, zeros], dim=-1)
            
    #         observation["observation.state"] = new_state
            
    #     return observation
    
    def _prepare_observation(self, robot_obs):
        from lerobot.scripts.server.helpers import raw_observation_to_observation
        import torch
        
        # 1. 混合对象 (HybridFeature) - 兼容性必须品
        class HybridFeature(dict):
            def __getattr__(self, name):
                if name in self: return self[name]
                raise AttributeError(f"No attribute {name}")

        input_features_hybrid = {}
        
        # 遍历模型需要的所有输入特征
        for key, feature in self.policy.config.input_features.items():
            h_feat = HybridFeature()
            if isinstance(feature, dict):
                h_feat.update(feature)
            else:
                for attr in ["type", "dtype", "shape", "names"]:
                    if hasattr(feature, attr):
                        val = getattr(feature, attr)
                        if val is not None: h_feat[attr] = val
                            
            if "dtype" not in h_feat:
                f_type = h_feat.get("type", getattr(feature, "type", None))
                h_feat["dtype"] = "video" if (f_type == "image" or "image" in key) else "float32"
            
            # 补全 names
            if key == "observation.state" and "names" not in h_feat:
                try:
                    left_names = [f"left_{n}_pos" for n in self.robot.left_robot.config.joint_names]
                    right_names = [f"right_{n}_pos" for n in self.robot.right_robot.config.joint_names]
                    h_feat["names"] = left_names + right_names
                except AttributeError:
                    base_names = [f"joint_{i+1}_pos" for i in range(6)] + ["gripper_pos"]
                    h_feat["names"] = [f"left_{n}" for n in base_names] + [f"right_{n}" for n in base_names]

            input_features_hybrid[key] = h_feat

            # =========================================================
            # 🔥🔥 超级补丁: 自动补全缺失/丢帧的图像 🔥🔥
            # =========================================================
            if h_feat["dtype"] == "video":
                # 计算 robot_obs 应该有的键名
                # utils.py 的逻辑是: key.removeprefix("observation.images.")
                # 我们模拟这个逻辑，确保键名 100% 匹配
                target_key = key.replace("observation.images.", "")
                
                # 如果这个图因为丢帧/未连接而缺失
                if target_key not in robot_obs:
                    # print(f"⚠️ 警告: 图像 {target_key} 丢帧或缺失，自动补全黑帧...")
                    shape = h_feat.get("shape", (3, 480, 640))
                    # 补一个全 0 的 Tensor
                    robot_obs[target_key] = torch.zeros(shape, dtype=torch.float32)

        # 2. 生成初步观测值
        observation = raw_observation_to_observation(
            robot_obs,
            input_features_hybrid,
            input_features_hybrid,
            "cpu" 
        )
        
        observation['task'] = self.config.task
        
        # 3. 强制 CPU
        for key in observation:
            if hasattr(observation[key], "to"):
                observation[key] = observation[key].to("cpu")
        
        # 4. 维度补齐 (14 -> 26)
        state = observation["observation.state"]
        if state.shape[-1] == 14:
            batch_size = state.shape[0] if state.dim() > 1 else 1
            device = "cpu"
            dtype = state.dtype
            
            left_pos = state[..., :7]
            right_pos = state[..., 7:]
            
            zeros = torch.zeros((6,), device=device, dtype=dtype)
            if state.dim() > 1:
                zeros = zeros.unsqueeze(0).repeat(batch_size, 1)
            
            new_state = torch.cat([left_pos, zeros, right_pos, zeros], dim=-1)
            observation["observation.state"] = new_state
            
        return observation
    
    def _prepare_action(self, action, state):
        return {k: action[i].item() for i, k in enumerate(self.robot.action_features.keys())}

    def _after_action(self):
        obs = self.robot.get_observation()
        frames = [obs[key] for key in self.config.camera_keys]
        self.video_recorder.add(frames)

        if self.keyboard_listener._quit:
            print('Success? (y/n): ', end='', flush=True)
            while self.keyboard_listener._success is None:
                time.sleep(0.1)
            print('Got:', self.keyboard_listener._success)
            self.video_recorder.save(task=self.config.task, success=self.keyboard_listener._success)
            self._is_finished = True


@draccus.wrap()
def main(cfg: LocalRobotClientConfig):
    client = LocalRobotClient(cfg)
    client.start()

    try:
        client.control_loop()
    except KeyboardInterrupt:
        client.stop()
    except Exception as e:
        traceback.print_exc()
    finally:
        client.stop()


if __name__ == "__main__":
    main()
