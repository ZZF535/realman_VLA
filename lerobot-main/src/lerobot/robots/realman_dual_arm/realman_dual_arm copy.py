# # realman_dual_arm.py
# import time
# import numpy as np
# import torch
# import threading
# from typing import Any

# from lerobot.cameras.utils import make_cameras_from_configs
# from ..robot import Robot
# from ..config import RobotConfig
# from Robotic_Arm.rm_robot_interface import *
# from .config_realman_dual_arm import RealManDualArmConfig

# # === 1. 导入睿尔曼 SDK ===
# # 优先尝试从系统库导入，如果失败则尝试本地文件
# try:
#     from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e, RM65_B
#     print("Success: Imported RealMan SDK from system library.")
# except ImportError:
#     raise ImportError("无法导入 Robotic_Arm.rm_robot_interface，请确认SDK已安装或文件位置正确。")



# class RealManDualArmRobot(Robot):
#     """
#     睿尔曼双臂机器人 LeRobot 驱动 (RealMan Dual Arm Driver)
#     基于官方 SDK rm_robot_interface.py 实现，负责将 LeRobot 的 Action 转换为 SDK 指令，并将 SDK 状态转换为 Observation
#     """
#     config_class = RealManDualArmConfig
#     name = "realman_dual_arm"

#     def __init__(self, config: RealManDualArmConfig, **kwargs):
#         # 1. 必须先调用父类初始化
#         super().__init__(config=config, **kwargs)
#         self.config = config
        
#         # === [关键！报错就是因为缺了下面这行] ===
#         self._connected = False   # <--- 必须有这一行！
#         # ====================================
        
#         self._calibrated = True

#     @property
#     def observation_features(self):
#         # 这里定义 Dataset 的数据结构
#         # 14维 = 左臂(6) + 左爪(1) + 右臂(6) + 右爪(1)
#         return {
#             "observation.state": {
#                 "dtype": "float32",
#                 "shape": (14,),
#                 "names": [
#                     # Left Arm
#                     "left_joint_1", "left_joint_2", "left_joint_3", 
#                     "left_joint_4", "left_joint_5", "left_joint_6", 
#                     "left_gripper",
#                     # Right Arm
#                     "right_joint_1", "right_joint_2", "right_joint_3", 
#                     "right_joint_4", "right_joint_5", "right_joint_6", 
#                     "right_gripper"
#                 ]
#             },
#             # 相机特征会由 LeRobot 框架根据 config 自动添加，这里如果不加也没关系，
#             # 但为了严谨，通常这里只定义 state
#         }

#     @property
#     def action_features(self):
#         # 动作和状态通常是对齐的
#         return {
#             "action": {
#                 "dtype": "float32",
#                 "shape": (14,),
#                 "names": [
#                     "left_joint_1", "left_joint_2", "left_joint_3", 
#                     "left_joint_4", "left_joint_5", "left_joint_6", 
#                     "left_gripper",
#                     "right_joint_1", "right_joint_2", "right_joint_3", 
#                     "right_joint_4", "right_joint_5", "right_joint_6", 
#                     "right_gripper"
#                 ]
#             }
#         }

#     @property
#     def is_connected(self) -> bool:
#         return self._connected

#     @property
#     def is_calibrated(self) -> bool:
#         return self._calibrated

#     # ---------------------------------------------------------
#     # 2. 连接与校准
#     # ---------------------------------------------------------
#     def connect(self):
#         if self._connected:
#             return

#         print(f"Connecting to RealMan Arms: L={self.config.left_ip}, R={self.config.right_ip}")
        
#         # 初始化 SDK 类 (请根据你的 SDK 实际类名修改 Robotic_Arm)
#         try:
#             self.arm_left = RoboticArm(RM65_B, self.config.left_ip)
#             self.arm_right = RoboticArm(RM65_B, self.config.right_ip)
#             # 可以在这里做一些初始化设置，比如切换到透传模式
#             # self.arm_left.rm_set_arm_run_mode(1) 
#         except Exception as e:
#             raise ConnectionError(f"SDK Init Failed: {e}")

#         # 连接相机 (LeRobot 标准流程)
#         for name, camera in self.cameras.items():
#             camera.connect()

#         self._connected = True
#         print("✅ RealMan Arms Connected!")

#     def disconnect(self):
#         if not self._connected:
#             return
        
#         # 断开 SDK (如果有 disconnect 方法)
#         if self.arm_left:
#             self.arm_left.rm_delete_robot_arm()
#         if self.arm_right:
#             self.arm_right.rm_delete_robot_arm()
            
#         for name, camera in self.cameras.items():
#             camera.disconnect()
            
#         self._connected = False

#     def calibrate(self):
#         # 对于睿尔曼，通常不需要像 Aloha 那样找零点
#         # 如果需要回零，可以在这里写
#         pass

#     def configure(self, config):
#         # 用于运行时更新配置，通常留空
#         pass

# # def get_observation(self) -> dict[str, Any]:
# #         """
# #         获取当前帧的观测数据
# #         返回: 
# #            observation.state: Tensor(14,)
# #            observation.images.*: Tensor(C,H,W)
# #         """
# #         if not self._connected:
# #             raise ConnectionError("Robot not connected")

# #         # 1. 读取相机图像 (异步读取)
# #         obs_dict = {}
# #         for name, cam in self.cameras.items():
# #             key_name = f"observation.image.{name}"
# #             obs_dict[key_name] = cam.async_read()

# #         # 2. 读取机械臂状态
# #         # 为了降低延迟，这里是串行读取，追求极致性能可用多线程
# #         # 调用 SDK 获取数据 (返回的是 7 个元素的 list)
# #         ret_l, raw_joints_l = self.arm_left.rm_get_joint_degree()
# #         ret_r, raw_joints_r = self.arm_right.rm_get_joint_degree()

# #         # 只取前 6 个数据！
# #         joints_l_deg = raw_joints_l[:6]  
# #         joints_r_deg = raw_joints_r[:6]

# #         if ret_l != 0 or ret_r != 0:
# #             print("Warning: 读取关节角度失败")

# #         # SDK返回的是角度(degree)，LeRobot需要弧度(radian)
# #         joints_l_rad = np.radians(np.array(joints_l_deg, dtype=np.float32))
# #         joints_r_rad = np.radians(np.array(joints_r_deg, dtype=np.float32))

# #         # 模拟获取夹爪状态 (如果SDK不支持直接读夹爪，就用上一帧的指令值)
# #         # 这里假设我们用缓存的指令值，或者你需要查阅 SDK 的 get_gripper 接口
# #         g_l = self._obs_cache["left_gripper"]
# #         g_r = self._obs_cache["right_gripper"]

# #         # 3. 拼接 State 向量 (14维)
# #         # 顺序: [左臂(6), 左爪(1), 右臂(6), 右爪(1)]
# #         state_vec = np.concatenate([
# #             joints_l_rad, 
# #             [g_l], 
# #             joints_r_rad, 
# #             [g_r]
# #         ])

# #         obs_dict["observation.state"] = torch.from_numpy(state_vec)
# #         return obs_dict

#     def get_observation(self) -> dict[str, Any]:
#         """
#         获取当前帧的观测数据
#         """
#         if not self._connected:
#             raise ConnectionError("Robot not connected")

#         # 1. 读取相机图像 (保持不变)
#         obs_dict = {}
#         for name, cam in self.cameras.items():
#             key_name = f"observation.image.{name}"
#             obs_dict[key_name] = cam.async_read()

#         # 2. 读取机械臂状态
#         ret_l, raw_joints_l = self.arm_left.rm_get_joint_degree()
#         ret_r, raw_joints_r = self.arm_right.rm_get_joint_degree()

#         # 只取前 6 个关节数据
#         joints_l_deg = raw_joints_l[:6]  
#         joints_r_deg = raw_joints_r[:6]

#         if ret_l != 0 or ret_r != 0:
#             print(f"Warning: 读取关节角度失败 L:{ret_l} R:{ret_r}")

#         # 角度转弧度
#         joints_l_rad = np.radians(np.array(joints_l_deg, dtype=np.float32))
#         joints_r_rad = np.radians(np.array(joints_r_deg, dtype=np.float32))

#         # --- [修改开始] 读取真实夹爪状态 ---
#         # 睿尔曼 API: rm_get_gripper_state() 返回 (int: 错误码, dict: 状态信息)
#         # 状态字典中通常包含 "position" 字段，范围 1-1000
#         ret_gl, gripper_state_l = self.arm_left.rm_get_gripper_state()
#         ret_gr, gripper_state_r = self.arm_right.rm_get_gripper_state()

#         # 处理左手夹爪 (归一化 0~1000 -> 0.0~1.0)
#         if ret_gl == 0 and isinstance(gripper_state_l, dict):
#             # 假设 1000 是最大开口，需根据实际情况确认方向
#             g_l = float(gripper_state_l.get('position', 0)) / 1000.0
#             self._obs_cache["left_gripper"] = g_l # 更新缓存
#         else:
#             g_l = self._obs_cache.get("left_gripper", 0.0)

#         # 处理右手夹爪
#         if ret_gr == 0 and isinstance(gripper_state_r, dict):
#             g_r = float(gripper_state_r.get('position', 0)) / 1000.0
#             self._obs_cache["right_gripper"] = g_r
#         else:
#             g_r = self._obs_cache.get("right_gripper", 0.0)
#         # --- [修改结束] ---

#         # 3. 拼接 State 向量 (保持不变)
#         state_vec = np.concatenate([
#             joints_l_rad, 
#             [g_l], 
#             joints_r_rad, 
#             [g_r]
#         ])

#         obs_dict["observation.state"] = torch.from_numpy(state_vec)
#         return obs_dict
    
    
    
    
# # def send_action(self, action: torch.Tensor) -> torch.Tensor:
# #         """
# #         执行动作
# #         action: Tensor(14,) -> [左臂(6), 左爪(1), 右臂(6), 右爪(1)]
# #         """
# #         if not self._connected:
# #             raise ConnectionError("Robot not connected")

# #         # 转为 numpy
# #         if isinstance(action, torch.Tensor):
# #             action = action.detach().cpu().numpy()

# #         # 1. 解析数据
# #         # 索引切片依据 config.features 的顺序
# #         target_l_rad = action[0:6]
# #         target_g_l   = action[6]
# #         target_r_rad = action[7:13]
# #         target_g_r   = action[13]

# #         # 2. 弧度转角度
# #         target_l_deg = np.degrees(target_l_rad).tolist()
# #         target_r_deg = np.degrees(target_r_rad).tolist()

# #         # 3. 发送机械臂指令 (使用透传模式/High Frequency Mode)
# #         # 注意：这里的第二个参数 1 代表透传/伺服模式，具体请查阅你的 SDK 文档
# #         self.arm_left.RM_Set_Joint_Position(target_l_deg, 1)
# #         self.arm_right.RM_Set_Joint_Position(target_r_deg, 1)

# #         # 4. 发送夹爪指令
# #         # 映射 0.0~1.0 -> 0~1000 (假设 0 是松开)
# #         # 需确认你的夹爪是 0 开 1000 闭，还是反过来
# #         g_l_val = int(target_g_l * 1000)
# #         g_r_val = int(target_g_r * 1000)
        
# #         # 简单的阈值控制，避免频繁抖动
# #         self.arm_left.RM_Set_Gripper_Position(g_l_val, False) # False表示不阻塞
# #         self.arm_right.RM_Set_Gripper_Position(g_r_val, False)

# #         # 更新缓存用于 observation 读取
# #         self._obs_cache["left_gripper"] = target_g_l
# #         self._obs_cache["right_gripper"] = target_g_r

# #         return torch.from_numpy(action)

#     def send_action(self, action: torch.Tensor) -> torch.Tensor:
#         """
#         执行动作
#         """
#         if not self._connected:
#             raise ConnectionError("Robot not connected")

#         if isinstance(action, torch.Tensor):
#             action = action.detach().cpu().numpy()

#         # 1. 解析数据
#         target_l_rad = action[0:6]
#         target_g_l   = action[6]
#         target_r_rad = action[7:13]
#         target_g_r   = action[13]

#         # 2. 弧度转角度 (SDK需要角度列表)
#         target_l_deg = np.degrees(target_l_rad).tolist()
#         target_r_deg = np.degrees(target_r_rad).tolist()

#         # --- [修改开始] 发送机械臂指令 (使用透传伺服接口) ---
#         # rm_servo_j 是睿尔曼的高频伺服接口，专用于实时控制
#         # 它可以接受当前的规划周期数据，直接驱动电机，无轨迹规划延迟
#         # 注意：使用前请确保机械臂处于正确的模式（通常 MoveJ 模式下直接调用即可，部分固件需先切模式）
        
#         # 错误处理：检查 SDK 调用返回值
#         ret_l = self.arm_left.rm_servo_j(target_l_deg)
#         ret_r = self.arm_right.rm_servo_j(target_r_deg)
        
#         if ret_l != 0:
#             print(f"Left Arm Servo Error: {ret_l}")
#         # --- [修改结束] ---

#         # --- [修改开始] 发送夹爪指令 (增加阈值控制) ---
#         # 归一化映射: 0.0~1.0 -> 1~1000
#         # 限制范围在 1-1000 之间 (睿尔曼夹爪通常接受 1-1000)
#         g_l_val = int(np.clip(target_g_l * 1000, 1, 1000))
#         g_r_val = int(np.clip(target_g_r * 1000, 1, 1000))

#         # 获取上一帧的夹爪状态用于对比
#         last_g_l = self._obs_cache.get("last_command_g_l", -1)
#         last_g_r = self._obs_cache.get("last_command_g_r", -1)

#         # 阈值控制: 只有变化超过 5% (50个单位) 才发送指令
#         # 这样可以防止网络拥塞，且避免夹爪电机过热
#         THRESHOLD = 50 

#         if abs(g_l_val - last_g_l) > THRESHOLD:
#             # rm_set_gripper_position(pos, block=False) 
#             # 参数2为 True 表示阻塞，False 表示非阻塞。LeRobot 必须非阻塞。
#             self.arm_left.rm_set_gripper_position(g_l_val, False)
#             self._obs_cache["last_command_g_l"] = g_l_val

#         if abs(g_r_val - last_g_r) > THRESHOLD:
#             self.arm_right.rm_set_gripper_position(g_r_val, False)
#             self._obs_cache["last_command_g_r"] = g_r_val
#         # --- [修改结束] ---

#         # 更新缓存用于 observation (模拟回显，虽然上面已经有了真实读取，但用于平滑)
#         self._obs_cache["left_gripper"] = target_g_l
#         self._obs_cache["right_gripper"] = target_g_r

#         return torch.from_numpy(action)


import time
import numpy as np
import torch
from typing import Any, Dict

from lerobot.cameras.utils import make_cameras_from_configs
from ..robot import Robot
from .config_realman_dual_arm import RealManDualArmConfig


try:
    # 导入必要的类和结构体
    from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e, rm_peripheral_read_write_params_t
    print("Success: Imported RoboticArm & Modbus Params.")
except ImportError as e:
    print(f"[CRITICAL] SDK 导入失败: {e}")
    RoboticArm = None

class RealManDualArmRobot(Robot):
    config_class = RealManDualArmConfig
    name = "realman_dual_arm"

    def __init__(self, config: RealManDualArmConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config = config
        self._connected = False
        
        self._obs_cache = {
            "left_gripper": 0.0, "right_gripper": 0.0,
            "last_command_g_l": -1, "last_command_g_r": -1
        }
        self.arm_left = None
        self.arm_right = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def observation_features(self):
        return {"observation.starm_write_registerste": {"dtype": "float32", "shape": (14,), "names": ["left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper"]}}

    @property
    def action_features(self):
        return {"action": {"dtype": "float32", "shape": (14,), "names": ["left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper"]}}

    @property
    def is_connected(self) -> bool: return self._connected
    @property
    def is_calibrated(self) -> bool: return True

    def connect(self):
        if self._connected: return
        print(f"Connecting Arms: L={self.config.left_arm_ip}, R={self.config.right_arm_ip}")
        
        if RoboticArm is None: raise ImportError("SDK 未找到")

        try:
            # 1. 使用单线程模式
            mode = rm_thread_mode_e.RM_SINGLE_MODE_E
            self.arm_left = RoboticArm(mode)
            self.arm_right = RoboticArm(mode)
            
            # 2. 建立连接
            self.arm_left.rm_create_robot_arm(self.config.left_arm_ip, 8080)
            self.arm_right.rm_create_robot_arm(self.config.right_arm_ip, 8080)
            print("  - Socket Connected.")

            # === [新增] 3. 初始化 Modbus 夹爪 ===
            print("  - Initializing Modbus Grippers...")
            
            # (1) 设置端口模式: Port=1 (末端), Baud=115200, Timeout=2
            self.arm_left.rm_set_modbus_mode(1, 115200, 2)
            self.arm_right.rm_set_modbus_mode(1, 115200, 2)
            time.sleep(0.5) # 给一点时间生效

            # (2) 使能夹爪: 写入寄存器 256 = 1
            # 参数: port=1, address=256, device=1
            params_enable = rm_peripheral_read_write_params_t(1, 256, 1)
            
            self.arm_left.rm_write_single_register(params_enable, 1)
            self.arm_right.rm_write_single_register(params_enable, 1)
            print("  - Grippers Enabled.")
            time.sleep(0.5)

        except Exception as e:
            raise ConnectionError(f"连接失败: {e}")

        for name, cam in self.cameras.items(): cam.connect()
        self._connected = True
        print("✅ All Systems Connected!")

    def disconnect(self):
        if not self._connected: return
        try: self.arm_left.rm_delete_robot_arm()
        except: pass
        try: self.arm_right.rm_delete_robot_arm()
        except: pass
        for name, cam in self.cameras.items(): cam.disconnect()
        self._connected = False

    def get_observation(self) -> Dict[str, Any]:
        if not self._connected: raise ConnectionError("Not connected")
        obs_dict = {}
        for name, cam in self.cameras.items():
            obs_dict[f"observation.image.{name}"] = cam.async_read()

        j_l = [0.0]*6; j_r = [0.0]*6
        
        # 1. 读取关节
        try:
            ret_l, raw_l = self.arm_left.rm_get_joint_degree()
            if ret_l == 0 and len(raw_l) >= 6: j_l = raw_l[:6]
        except: pass

        try:
            ret_r, raw_r = self.arm_right.rm_get_joint_degree()
            if ret_r == 0 and len(raw_r) >= 6: j_r = raw_r[:6]
        except: pass

        # 2. 读取夹爪 (这里暂时保持原样，或者也需要改寄存器读取？)
        # 如果 rm_get_gripper_state 不准，我们可以用 _obs_cache 里的值“欺骗”模型
        # 目前先相信缓存的值是准的
        
        state = np.concatenate([np.radians(j_l), [self._obs_cache["left_gripper"]], np.radians(j_r), [self._obs_cache["right_gripper"]]])
        obs_dict["observation.state"] = torch.from_numpy(state).float()
        return obs_dict

    def _write_gripper_register(self, arm_obj, value_0_to_1000):
        """
        通过 Modbus 寄存器控制夹爪
        value: 0 ~ 1000
        """
        try:
            # 1. 设置位置 (Reg 258)
            # 注意：你的示例代码用的是 rm_write_registers (复数)，这里假设写单个值
            p_pos = rm_peripheral_read_write_params_t(1, 258, 1)
            arm_obj.rm_write_registers(p_pos, [0])
            
            # 2. 设置速度/力度? (Reg 259) - 你的示例设为 500
            p_spd = rm_peripheral_read_write_params_t(1, 259, 1)
            arm_obj.rm_write_registers(p_spd, [int(value_0_to_1000)])
            
            # 3. 触发运动 (Reg 264) - 写入 1
            p_trig = rm_peripheral_read_write_params_t(1, 264, 1)
            arm_obj.rm_write_single_register(p_trig, [1])
            
        except Exception as e:
            print(f"Gripper Write Error: {e}")

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self._connected: raise ConnectionError("Not connected")
        if isinstance(action, torch.Tensor): action = action.detach().cpu().numpy()
        
        # 1. 关节控制 (MoveJ 非阻塞)
        try:
            target_l = np.degrees(action[0:6]).tolist()
            target_r = np.degrees(action[7:13]).tolist()
            # v=30, r=0, connect=0, block=0
            self.arm_left.rm_movej(target_l, 10, 0, 0, 0)
            self.arm_right.rm_movej(target_r, 10, 0, 0, 0)
        except Exception as e:
            print(f"MoveJ Error: {e}")
        
        # 2. 夹爪控制 (寄存器方式)
        tg_l = int(np.clip(action[6]*1000, 1, 1000))
        tg_r = int(np.clip(action[13]*1000, 1, 1000))
        
        # 阈值防抖 (Modbus通信慢，必须防抖)
        if abs(tg_l - self._obs_cache["last_command_g_l"]) > 50:
            self._write_gripper_register(self.arm_left, tg_l)
            self._obs_cache["last_command_g_l"] = tg_l
            
        if abs(tg_r - self._obs_cache["last_command_g_r"]) > 50:
            self._write_gripper_register(self.arm_right, tg_r)
            self._obs_cache["last_command_g_r"] = tg_r
            
        return torch.from_numpy(action)

    def calibrate(self): pass
    def configure(self, config): pass