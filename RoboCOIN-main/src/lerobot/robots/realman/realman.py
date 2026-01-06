# # """
# # Realman robot implementation.
# # """

# # import importlib
# # import numpy as np
# # import time
# # from ..base_robot import BaseRobot
# # from .configuration_realman import RealmanConfig


# # class Realman(BaseRobot):
# #     """
# #     Realman robot implementation.
# #     Params:
# #     - config: RealmanConfig
# #     """

# #     config_class = RealmanConfig
# #     name = "realman"

# #     def __init__(self, config: RealmanConfig) -> None:
# #         super().__init__(config)
# #         self.config = config

# #     def _check_dependency(self) -> None:
# #         """
# #         Check for dependencies required by the Realman robot.
# #         Raises ImportError if the required package is not found.
# #         """
# #         if importlib.util.find_spec("Robotic_Arm") is None:
# #             raise ImportError(
# #                 "Realman robot requires the Robotic_Arm package. "
# #                 "Please install it using 'pip install Robotic_Arm'."
# #             )
    
# #     def _connect_arm(self) -> None:
# #         """
# #         Connect to the Realman robot arm.
# #         Initializes the RoboticArm interface and creates a robot arm handle.
# #         """
# #         from Robotic_Arm.rm_robot_interface import (
# #             RoboticArm, 
# #             rm_thread_mode_e,
# #         )
# #         self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# #         self.handle = self.arm.rm_create_robot_arm(self.config.ip, self.config.port)
# #         self.arm.rm_set_arm_run_mode(1)
    
# #     def _disconnect_arm(self) -> None:
# #         """
# #         Disconnect from the Realman robot arm.
# #         Destroys the robot arm handle.
# #         """
# #         ret_code = self.arm.rm_destroy()
# #         if ret_code != 0:
# #             raise RuntimeError(f'Failed to disconnect: {ret_code}')
    
# #     def _set_joint_state(self, state: np.ndarray) -> None:
# #         """
# #         Set the joint state of the Realman robot.
# #         Uses the RoboticArm interface to move the joints and set the gripper position.
# #         Raises RuntimeError if the movement fails.
# #         Params:
# #         - state: np.ndarray of joint positions
# #         """
# #         state = list(state)
# #         success = self.arm.rm_movej(state[:-1], v=self.config.velocity, r=0, connect=0, block=self.config.block)

# #         if success != 0:
# #             raise RuntimeError(f'Failed movej')
# #         success = self.arm.rm_set_gripper_position(int(state[-1]), block=self.config.block, timeout=3)
# #         if success != 0:
# #             raise RuntimeError('Failed set gripper')

# #         if not self.config.block:
# #             time.sleep(self.config.wait_second)
    
# #     def _get_joint_state(self) -> np.ndarray:
# #         """
# #         Get the joint state of the Realman robot.
# #         Uses the RoboticArm interface to retrieve the current joint and gripper states.
# #         Raises RuntimeError if retrieval fails.
# #         Returns:
# #         - state: np.ndarray of joint positions
# #         """
# #         ret_code, joint = self.arm.rm_get_joint_degree()
# #         if ret_code != 0:
# #             raise RuntimeError(f'Failed to get joint state: {ret_code}')
# #         ret_code, grip = self.arm.rm_get_gripper_state()
# #         grip = grip['actpos']
# #         if ret_code != 0:
# #             raise RuntimeError(f'Failed to get gripper state: {ret_code}')
# #         return np.array(joint + [grip])
    
# #     def _set_ee_state(self, state: np.ndarray) -> None:
# #         """
# #         Set the end-effector state of the Realman robot.
# #         Uses the RoboticArm interface to compute inverse kinematics and set joint states accordingly.
# #         Raises RuntimeError if inverse kinematics fails.
# #         Params:
# #         - state: np.ndarray of end-effector positions
# #         """
# #         from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t
# #         state = list(state)
# #         ret_code, joint = self.arm.rm_algo_inverse_kinematics(rm_inverse_kinematics_params_t(
# #             q_in=self._get_joint_state()[:-1],
# #             q_pose=state[:-1],
# #             flag=1
# #         ))
# #         if ret_code != 0:
# #             print('IK error:', ret_code)
# #         self._set_joint_state(joint + [state[-1]])

# #     def _get_ee_state(self) -> np.ndarray:
# #         """
# #         Get the end-effector state of the Realman robot.
# #         Uses the RoboticArm interface to compute forward kinematics based on current joint states.
# #         Raises RuntimeError if retrieval fails.
# #         Returns:
# #         - state: np.ndarray of end-effector positions
# #         """
# #         joint = self._get_joint_state()
# #         pose = self.arm.rm_algo_forward_kinematics(joint[:-1], flag=1)
# #         return np.array(pose + [joint[-1]])

# # import importlib
# # import numpy as np
# # import time
# # from ..base_robot import BaseRobot
# # from .configuration_realman import RealmanConfig


# # class Realman(BaseRobot):
# #     """
# #     Realman robot implementation.
# #     Params:
# #     - config: RealmanConfig
# #     """

# #     config_class = RealmanConfig
# #     name = "realman"

# #     def __init__(self, config: RealmanConfig) -> None:
# #         super().__init__(config)
# #         self.config = config
# #         self._last_gripper_target = 0.0  # 用于缓存上一次下发的夹爪目标值

# #     def _check_dependency(self) -> None:
# #         """
# #         Check for dependencies required by the Realman robot.
# #         Raises ImportError if the required package is not found.
# #         """
# #         if importlib.util.find_spec("Robotic_Arm") is None:
# #             raise ImportError(
# #                 "Realman robot requires the Robotic_Arm package. "
# #                 "Please install it using 'pip install Robotic_Arm'."
# #             )
    
# #     def _connect_arm(self) -> None:
# #         """
# #         Connect to the Realman robot arm.
# #         Initializes the RoboticArm interface and creates a robot arm handle.
# #         If use_zhixing_gripper is True, it also initializes the Modbus mode and enables the gripper.
# #         """
# #         from Robotic_Arm.rm_robot_interface import (
# #             RoboticArm, 
# #             rm_thread_mode_e,
# #             rm_peripheral_read_write_params_t  # 必须导入
# #         )
# #         self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# #         self.handle = self.arm.rm_create_robot_arm(self.config.ip, self.config.port)
        
# #         # 设置透传模式，降低通信延迟
# #         # self.arm.rm_set_arm_run_mode(1)
# #         # print("[RealMan] Forcing Run Mode to 0 (Trajectory Mode)...")
# #         # ret = self.arm.rm_set_arm_run_mode(0)
# #         # if ret != 0:
# #         #     print(f"Warning: Failed to set Run Mode 0, ret={ret}")
            
# #         # time.sleep(1)

# #         if self.handle.id == -1:
# #             raise RuntimeError(f"Failed to connect to RealMan arm at {self.config.ip}")

# #         # === [核心配置: 拖拽示教模式] ===
        
# #         # 1. 必须先切回标准轨迹模式(0)，否则后续指令可能被拒
# #         print("[RealMan] Setting Run Mode to 0 (Trajectory)...")
# #         self.arm.rm_set_arm_run_mode(1)
# #         time.sleep(0.5)
        
# #         # 2. 开启零力拖动示教 (Drag Teach)
# #         # 开启后，机械臂变软，可以手动拖动
# #         # print("[RealMan] Enabling Drag Teach Mode (Zero Force)...")
# #         # try:
# #         #     ret = self.arm.rm_start_multi_drag_teach(1, 0)
# #         # except AttributeError:
# #         #     # 如果还是报错，尝试备用函数 rm_start_drag_teach(1)
# #         #     print("Warning: rm_start_multi_drag_teach not found, trying rm_start_drag_teach...")
# #         #     ret = self.arm.rm_start_drag_teach(1)
# #         # if ret != 0:
# #         #     print(f"Warning: Failed to enable Drag Teach, ret={ret}")
        
# #         # time.sleep(1.0)

# #         print("[RealMan] Enabling Drag Teach Mode (Zero Force)...")
# #         ret = 1
        
# #         # 尝试 A: 复合拖拽接口 (适用于力控版)
# #         if hasattr(self.arm, 'rm_start_multi_drag_teach'):
# #             ret = self.arm.rm_start_multi_drag_teach(1, 0)
# #             if ret == 0:
# #                 print("  > Multi-drag teach enabled.")
        
# #         # 尝试 B: 基础拖拽接口 (适用于标准版，或者 A 失败时回退)
# #         # 注意：这里的逻辑是 if ret != 0，即如果上面没成功，就试这个
# #         if ret != 0 and hasattr(self.arm, 'rm_start_drag_teach'):
# #             print("  > Multi-drag failed or not supported. Trying basic rm_start_drag_teach(1)...")
# #             ret = self.arm.rm_start_drag_teach(1)
# #             if ret == 0:
# #                 print("  > Basic drag teach enabled.")

# #         if ret != 0:
# #             print(f"Warning: Failed to enable Drag Teach, ret={ret}. Robot might be stiff!")
        
# #         time.sleep(1.0)
# #         # [新增] 知行夹爪初始化
# #         if getattr(self.config, 'use_zhixing_gripper', False):
# #             print(f"[RealMan] Initializing Zhixing Gripper on Port {self.config.gripper_port}...")
            
# #             # 1. 配置 Modbus 模式 (Port, Baud, Timeout)
# #             # 参考你的代码: 115200, timeout=2
# #             ret = self.arm.rm_set_modbus_mode(
# #                 self.config.gripper_port, 
# #                 self.config.gripper_baudrate, 
# #                 2
# #             )
# #             if ret != 0:
# #                 print(f"Warning: Failed to set Modbus mode, ret={ret}")
            
# #             time.sleep(1) # 等待生效

# #             # 2. 使能夹爪 (写寄存器 256 -> 1)
# #             enable_params = rm_peripheral_read_write_params_t(
# #                 self.config.gripper_port, 
# #                 self.config.gripper_enable_reg, 
# #                 self.config.gripper_device_id
# #             )
# #             ret = self.arm.rm_write_single_register(enable_params, 1)
# #             print(f"[RealMan] Enabling Zhixing Gripper... ret={ret}")
            
# #             time.sleep(3) # 等待使能完成
# #             print("[RealMan] Zhixing Gripper Ready.")
    
# #     def _disconnect_arm(self) -> None:
# #         """
# #         Disconnect from the Realman robot arm.
# #         Destroys the robot arm handle.
# #         """
# #         # ret_code = self.arm.rm_destroy()
# #         # if ret_code != 0:
# #         #     raise RuntimeError(f'Failed to disconnect: {ret_code}')
# #         print("[RealMan] Disabling Drag Teach...")
# #         # === [修改点] 使用正确的停止函数 ===
# #         try:
# #             self.arm.rm_stop_drag_teach()
# #         except AttributeError:
# #             pass # 如果找不到函数，可能已经停止或不需要显式停止
            
# #         time.sleep(0.5)
# #         ret_code = self.arm.rm_destroy()
# #         if ret_code != 0:
# #             print(f'Failed to disconnect: {ret_code}')


# #     def _set_joint_state(self, state: np.ndarray) -> None:
# #         """
# #         Set the joint state of the Realman robot.
# #         Uses the RoboticArm interface to move the joints and set the gripper position.
# #         Raises RuntimeError if the movement fails.
# #         Params:
# #         - state: np.ndarray of joint positions
# #         """
# #         from Robotic_Arm.rm_robot_interface import rm_peripheral_read_write_params_t

# #         state = list(state)
        
# #         # 1. 机械臂运动控制 (前7轴)
# #         # 使用 connect=0 (非阻塞直接下发) 以支持高频控制
# #         # success = self.arm.rm_movej(state[:-1], v=self.config.velocity, r=0, connect=0, block=self.config.block)
# #         # if success != 0:
# #             # 这里的报错可能会频繁打断训练，如果不影响运行可以改为 print warning
# #             # raise RuntimeError(f'Failed movej: {success}')
# #         pass
# #         # 2. 夹爪控制 (第8轴 state[-1])
# #         if getattr(self.config, 'use_zhixing_gripper', False):
# #             # 获取目标值 (假设 LeRobot 传入的是 0-1000 的范围，如果是一样的数据集)
# #             target_pos = int(state[-1])
            
# #             # 简单的限幅保护
# #             target_pos = max(0, min(1000, target_pos))
            
# #             # 缓存目标值，供 get_observation 使用
# #             self._last_gripper_target = target_pos

# #             # 步骤 A: 写入位置 (寄存器 259)
# #             pos_params = rm_peripheral_read_write_params_t(
# #                 self.config.gripper_port, 
# #                 self.config.gripper_pos_reg, 
# #                 self.config.gripper_device_id
# #             )
# #             # 这里不判断返回值，以保证速度
# #             self.arm.rm_write_single_register(pos_params, target_pos)

# #             # 步骤 B: 写入触发 (寄存器 264 -> 1)
# #             trigger_params = rm_peripheral_read_write_params_t(
# #                 self.config.gripper_port, 
# #                 self.config.gripper_trigger_reg, 
# #                 self.config.gripper_device_id
# #             )
# #             self.arm.rm_write_single_register(trigger_params, 1)

# #         else:
# #             # 原有的标准夹爪控制
# #             success = self.arm.rm_set_gripper_position(int(state[-1]), block=self.config.block, timeout=3)
# #             if success != 0:
# #                 raise RuntimeError('Failed set gripper')

# #         if not self.config.block:
# #             time.sleep(self.config.wait_second)
    
# #     def _get_joint_state(self) -> np.ndarray:
# #         """
# #         Get the joint state of the Realman robot.
# #         Uses the RoboticArm interface to retrieve the current joint and gripper states.
# #         Raises RuntimeError if retrieval fails.
# #         Returns:
# #         - state: np.ndarray of joint positions
# #         """
# #         ret_code, joint = self.arm.rm_get_joint_degree()
# #         if ret_code != 0:
# #             raise RuntimeError(f'Failed to get joint state: {ret_code}')
        
# #         grip_pos = 0.0
        
# #         if getattr(self.config, 'use_zhixing_gripper', False):
# #             # 对于透传夹爪，标准 API rm_get_gripper_state 可能无法读取真实位置。
# #             # 为了避免报错并保证训练循环继续，我们这里直接返回“上一次设定的目标值”。
# #             # 如果需要真实反馈，需要实现 rm_read_holding_registers 读取夹爪的当前位置寄存器。
# #             grip_pos = float(self._last_gripper_target)
# #         else:
# #             ret_code, grip = self.arm.rm_get_gripper_state()
# #             if ret_code == 0:
# #                 grip_pos = grip['actpos']
# #             else:
# #                 # 即使读取失败，也尽量不要抛出异常导致整个程序崩溃，打印警告即可
# #                 print(f"Warning: Failed to get gripper state: {ret_code}")
# #                 grip_pos = 0.0
                
# #         return np.array(joint + [grip_pos])
    
# #     def _set_ee_state(self, state: np.ndarray) -> None:
# #         """
# #         Set the end-effector state of the Realman robot.
# #         """
# #         # from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t
# #         # state = list(state)
# #         # # 使用当前关节状态计算逆解
# #         # current_joints = self._get_joint_state()[:-1] # 不包含夹爪
        
# #         # ret_code, joint = self.arm.rm_algo_inverse_kinematics(rm_inverse_kinematics_params_t(
# #         #     q_in=list(current_joints),
# #         #     q_pose=state[:-1],
# #         #     flag=1
# #         # ))
        
# #         # if ret_code != 0:
# #         #     print(f'IK error: {ret_code}')
# #         #     # 如果逆解失败，可以选择保持不动或者抛出异常
# #         #     return

# #         # # 调用 _set_joint_state 执行运动
# #         # self._set_joint_state(joint + [state[-1]])
# #         pass

# #     def _get_ee_state(self) -> np.ndarray:
# #         """
# #         Get the end-effector state of the Realman robot.
# #         """
# #         joint = self._get_joint_state()
# #         pose = self.arm.rm_algo_forward_kinematics(joint[:-1], flag=1)
# #         # 返回 [x, y, z, rx, ry, rz, gripper]
# #         return np.array(pose + [joint[-1]])

# import importlib
# import numpy as np
# import time
# from ..base_robot import BaseRobot
# from .configuration_realman import RealmanConfig

# class Realman(BaseRobot):
#     config_class = RealmanConfig
#     name = "realman"

#     def __init__(self, config: RealmanConfig) -> None:
#         super().__init__(config)
#         self.config = config

#     def _check_dependency(self) -> None:
#         if importlib.util.find_spec("Robotic_Arm") is None:
#             raise ImportError("Please install 'Robotic_Arm' package.")
    
#     def _connect_arm(self) -> None:
#         from Robotic_Arm.rm_robot_interface import (
#             RoboticArm, 
#             rm_thread_mode_e,
#         )
#         print(f"[RealMan] Connecting to {self.config.ip}:{self.config.port}...")
#         self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#         self.handle = self.arm.rm_create_robot_arm(self.config.ip, self.config.port)
        
#         if self.handle.id <= 0:
#             raise ConnectionError(f"Failed to connect to robot at {self.config.ip}")
            
#         print("[RealMan] Connection successful.")

#     def _disconnect_arm(self) -> None:
#         if hasattr(self, 'arm'):
#             self.arm.rm_destroy()
    
#     def _set_joint_state(self, state: np.ndarray) -> None:
#         # state: [joint_1, ..., joint_7, gripper]
#         state = list(state)
#         arm_joints = state[:-1]
#         gripper_val = state[-1]

#         # 1. 机械臂运动
#         # r=0 (弧度), block=0 (非阻塞)
#         self.arm.rm_movej(arm_joints, v=self.config.velocity, r=0, connect=0, block=0)

#         # 2. 夹爪控制 (使用你验证过的标准接口)
#         # 这里的 1000/0 取决于模型输出。LeRobot模型输出通常适配了范围。
#         # 如果模型输出是 [0, 1]，这里可能需要 * 1000。
#         # 但既然你之前的Log显示模型输出已经是 0-1000，这里直接转 int 即可。
#         target_pos = int(gripper_val)
        
#         # 这里的参数完全照搬你的代码: (pos, block=False, timeout=1)
#         # 注意：你代码里写的是 5 或 1，这里给 1 应该够了
#         self.arm.rm_set_gripper_position(target_pos, False, 1)

#         if not self.config.block:
#             # 控制频率同步
#             time.sleep(self.config.wait_second)
    
#     def _get_joint_state(self) -> np.ndarray:
#         # 1. 读取机械臂关节 (6或7轴)
#         # ret_code, joint = self.arm.rm_get_joint_degree()
#         # if ret_code != 0:
#         #     # 容错返回
#         #     return np.zeros(8)

#         # # 2. 读取夹爪状态 (照搬你的代码 logic)
#         # # rm_get_rm_plus_state_info 返回 (flag, dict)
#         # ret_grip, gripper_dict = self.arm.rm_get_rm_plus_state_info()
        
#         # gripper_pos = 0.0
#         # if ret_grip == 0 and 'pos' in gripper_dict:
#         #     # 获取实际位置
#         #     gripper_pos = float(gripper_dict['pos'][0])
            
#         # # 拼接返回: [j1, j2, ..., j7, gripper]
#         # return np.array(joint + [gripper_pos])
#         # state: [joint_1, ..., joint_7, gripper]
#         ret_code, joint = self.arm.rm_get_joint_degree()
#         if ret_code != 0:
#             joint = [0.0] * 6 # 容错

#         # 2. 读取夹爪状态
#         ret_grip, gripper_dict = self.arm.rm_get_rm_plus_state_info()
#         gripper_pos = 0.0
#         if ret_grip == 0 and 'pos' in gripper_dict:
#             gripper_pos = float(gripper_dict['pos'][0])
            
#         # 3. ⚡️ 关键修复：补齐关节速度 (Velocity)
#         # 模型训练时用了速度 (6维)，但我们现在只读位置。
#         # 为了不报错，我们补 6 个 0。
#         # 如果你的机械臂是 7 轴，这里应该是 7 个 0。
#         # 现在的状态向量：[j1, j2, j3, j4, j5, j6, gripper, v1, v2, v3, v4, v5, v6]
#         # 总共 6+1+6 = 13 维。双臂就是 26 维，正好匹配！
#         velocity = [0.0] * 6  
            
#         return np.array(joint + [gripper_pos] + velocity)
        
    
#     # IK/FK 部分保持原样
#     def _set_ee_state(self, state: np.ndarray) -> None:
#         pass
#     def _get_ee_state(self) -> np.ndarray:
#         joint = self._get_joint_state()
#         pose = self.arm.rm_algo_forward_kinematics(joint[:-1], flag=1)
#         return np.array(pose + [joint[-1]])

# import importlib
# import numpy as np
# import time
# import torch
# from ..base_robot import BaseRobot
# from .configuration_realman import RealmanConfig

# class Realman(BaseRobot):
#     config_class = RealmanConfig
#     name = "realman"

#     def __init__(self, config: RealmanConfig) -> None:
#         super().__init__(config)
#         self.config = config

#     def _check_dependency(self) -> None:
#         if importlib.util.find_spec("Robotic_Arm") is None:
#             raise ImportError("Please install 'Robotic_Arm' package.")
    
#     def _connect_arm(self) -> None:
#         from Robotic_Arm.rm_robot_interface import (
#             RoboticArm, 
#             rm_thread_mode_e,
#         )
#         print(f"[RealMan] Connecting to {self.config.ip}:{self.config.port}...")
#         self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#         self.handle = self.arm.rm_create_robot_arm(self.config.ip, self.config.port)
        
#         if self.handle.id <= 0:
#             raise ConnectionError(f"Failed to connect to robot at {self.config.ip}")
            
#         print("[RealMan] Connection successful.")

#     def _disconnect_arm(self) -> None:
#         if hasattr(self, 'arm'):
#             self.arm.rm_destroy()
            
#     # =========================================================================
#     # 🔥 核心修改：重写 get_observation，完全绕过 BaseRobot 的逻辑
#     # =========================================================================

#     # def get_observation(self) -> torch.Tensor:
#     #     """
#     #     手动读取状态，转换单位，补齐速度，直接返回 Tensor。
#     #     完全绕过 BaseRobot 的 get_joint_state 和单位转换检查。
#     #     """
#     #     # 1. 读取硬件 (7维: 6关节 + 1夹爪)
#     #     ret_code, joint = self.arm.rm_get_joint_degree()
#     #     if ret_code != 0: joint = [0.0] * 6
        
#     #     ret_grip, gripper_dict = self.arm.rm_get_rm_plus_state_info()
#     #     gripper_pos = 0.0
#     #     if ret_grip == 0 and 'pos' in gripper_dict:
#     #         gripper_pos = float(gripper_dict['pos'][0])
            
#     #     raw_pos = np.array(joint + [gripper_pos]) # Shape: (7,)

#     #     # 2. 手动进行单位转换 (Degree -> Radian)
#     #     # 注意：使用 joint_transform 进行第一次转换 (robot -> standard)
#     #     # 然后再进行 output_transform (standard -> model，通常是 Radian)
#     #     # 这里为了简化，我们假设模型就是 Radian，直接调用 input_transform 转为标准单位即可
#     #     calibrated_pos = self.joint_transform.input_transform(raw_pos)
        
#     #     # 3. 补齐速度向量 (6维)，凑够 13维
#     #     velocity = np.zeros(6)
        
#     #     # 4. 拼接
#     #     final_state = np.concatenate([calibrated_pos, velocity])
        
#     #     # 5. 转为 Tensor 并返回 (Float32)
#     #     return torch.from_numpy(final_state).float()
    
#     def get_observation(self) -> dict:
#         """
#         手动读取状态，转换单位，补齐速度，返回字典。
#         🔥 关键修复：同时提供 "joint_1" 和 "joint_1_pos" 两种格式，防止 KeyError。
#         """
#         # 1. 读取硬件 (7维: 6关节 + 1夹爪)
#         ret_code, joint = self.arm.rm_get_joint_degree()
#         if ret_code != 0: joint = [0.0] * 6
        
#         ret_grip, gripper_dict = self.arm.rm_get_rm_plus_state_info()
#         gripper_pos = 0.0
#         if ret_grip == 0 and 'pos' in gripper_dict:
#             gripper_pos = float(gripper_dict['pos'][0])
            
#         raw_pos = np.array(joint + [gripper_pos])

#         # 2. 单位转换
#         calibrated_pos = self.joint_transform.input_transform(raw_pos)
        
#         # 3. 构造字典 (双保险策略)
#         state_dict = {}
#         for i, name in enumerate(self.config.joint_names):
#             # 原始名字 (e.g. "joint_1")
#             state_dict[name] = calibrated_pos[i]
#             # 带后缀的名字 (e.g. "joint_1_pos") <- 解决 KeyError 的关键
#             state_dict[f"{name}_pos"] = calibrated_pos[i]
            
#         # 4. 补齐速度 (同样双保险)
#         # 假设关节是 joint_1 ... joint_6 (gripper通常没有速度或默认为0)
#         # 我们只给前6个关节补速度，因为 gripper 可能不需要或者名字特殊
#         for i in range(6):
#             j_name = f"joint_{i+1}"
#             state_dict[f"{j_name}_vel"] = 0.0      # 常用命名
#             state_dict[f"{j_name}_velocity"] = 0.0 # 备用命名
            
#         return state_dict

#     # def set_joint_state(self, state: np.ndarray) -> None:
#     #     """执行动作：接收13维模型输出 -> 截取前7维 -> 转换单位 -> 发送硬件"""
        
#     #     # 1. 截取前7维 (位置信息)
#     #     if len(state) > 7:
#     #         target_pos = state[:7]
#     #     else:
#     #         target_pos = state
            
#     #     # 2. 单位转换 (Radian -> Degree)
#     #     target_pos = self.joint_transform.output_transform(target_pos)
        
#     #     # 3. 发送给底层
#     #     self._set_joint_state(target_pos)
    
#     def set_joint_state(self, state: np.ndarray) -> np.ndarray:
#         """
#         执行动作：接收13维模型输出 -> 截取前7维 -> 转换单位 -> 发送硬件
#         ⚡️ 修复：必须返回当前的关节状态 (np.ndarray)，否则 BaseRobot 会报错。
#         """
        
#         # 1. 截取前7维 (位置信息)
#         if len(state) > 7:
#             target_pos = state[:7]
#         else:
#             target_pos = state
            
#         # 2. 单位转换 (Radian -> Degree)
#         target_pos_deg = self.joint_transform.output_transform(target_pos)
        
#         # 3. 发送给底层
#         self._set_joint_state(target_pos_deg)
        
#         # 4. 🔥 关键修复：返回实际的关节状态 (BaseRobot 需要这个返回值)
#         # 我们可以直接返回目标位置 (开环)，或者重新读取一次硬件 (闭环)
#         # 为了速度，通常返回目标位置即可 (但在 BaseRobot 逻辑里，它期望的是 calibrated_pos)
        
#         # 注意：BaseRobot 期望返回的是 calibrated (标准单位) 的数据
#         return target_pos

#     # =========================================================================
#     # 底层实现
#     # =========================================================================
    
#     def _set_joint_state(self, state: np.ndarray) -> None:
#         state = list(state)
#         arm_joints = state[:-1]
#         gripper_val = state[-1]

#         # 1. 机械臂运动
#         self.arm.rm_movej(arm_joints, v=self.config.velocity, r=0, connect=0, block=0)

#         # 2. 夹爪控制
#         target_pos = int(gripper_val)
#         target_pos = max(0, min(1000, target_pos))
#         self.arm.rm_set_gripper_position(target_pos, False, 1)

#         if not self.config.block:
#             time.sleep(self.config.wait_second)
            
#     # 下面这些必须实现，但已经被 get_observation 架空
#     def _get_joint_state(self): pass
#     def _set_ee_state(self, state): pass
#     def _get_ee_state(self): pass
#     def get_joint_state(self): pass # 覆盖基类方法以防万一

import importlib
import numpy as np
import time
from ..base_robot import BaseRobot
from .configuration_realman import RealmanConfig

class Realman(BaseRobot):
    config_class = RealmanConfig
    name = "realman"

    def __init__(self, config: RealmanConfig) -> None:
        super().__init__(config)
        self.config = config

    def _check_dependency(self) -> None:
        if importlib.util.find_spec("Robotic_Arm") is None:
            raise ImportError("Please install 'Robotic_Arm' package.")
    
    # def _connect_arm(self) -> None:
    #     from Robotic_Arm.rm_robot_interface import (
    #         RoboticArm, 
    #         rm_thread_mode_e,
    #     )
    #     print(f"[RealMan] Connecting to {self.config.ip}:{self.config.port}...")
    #     self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    #     self.handle = self.arm.rm_create_robot_arm(self.config.ip, self.config.port)
        
    #     if self.handle.id <= 0:
    #         raise ConnectionError(f"Failed to connect to robot at {self.config.ip}")
            
    #     print("[RealMan] Connection successful.")
    
    def _connect_arm(self) -> None:
        from Robotic_Arm.rm_robot_interface import (
            RoboticArm, 
            rm_thread_mode_e,
        )
        print(f"[RealMan] Connecting to {self.config.ip}:{self.config.port}...")
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(self.config.ip, self.config.port)
        
        if self.handle.id <= 0:
            raise ConnectionError(f"Failed to connect to robot at {self.config.ip}")
            
        print("[RealMan] Connection successful.")

        # === 🚀 核心添加：开启透传/高频模式 ===
        # 1 代表透传模式 (Transparent Mode)，适用于 AI 推理这种连续指令流
        # 0 代表轨迹模式 (Trajectory Mode)，适用于单次点对点移动
        ret = self.arm.rm_set_arm_run_mode(1) 
        if ret == 0:
            print("[RealMan] Transparent Mode (Run Mode 1) enabled.")
        else:
            print(f"⚠️ [RealMan] Failed to set Run Mode 1, return code: {ret}")

    def _disconnect_arm(self) -> None:
        if hasattr(self, 'arm'):
            self.arm.rm_destroy()

    # =========================================================================
    # 1. 关节状态读取 (必须实现)
    # =========================================================================
    def _get_joint_state(self) -> np.ndarray:
        """读取底层硬件数据 (7维)"""
        ret_code, joint = self.arm.rm_get_joint_degree()
        if ret_code != 0: 
            joint = [0.0] * 6
        
        ret_grip, gripper_dict = self.arm.rm_get_rm_plus_state_info()
        gripper_pos = 0.0
        if ret_grip == 0 and 'pos' in gripper_dict:
            gripper_pos = float(gripper_dict['pos'][0])
            
        return np.array(joint + [gripper_pos])

    # =========================================================================
    # 2. 动作执行 (核心控制)
    # =========================================================================
    def set_joint_state(self, state: np.ndarray):
        """
        这是 LeRobot 调用机器人的标准公有接口。
        输入：标准单位（弧度）
        输出：机器人单位（角度）
        """
        # 1. 确保输入是 numpy 数组
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # 2. 单位转换：标准单位(弧度) -> 机器人单位(角度)
        state = self.joint_transform.output_transform(state)
            
        # 3. 执行底层动作
        return self._set_joint_state(state)
    
    
    def _set_joint_state(self, state: np.ndarray) -> None:
        """发送命令 (7维) - 使用透传模式 rm_movej_canfd"""
        state_list = list(state)
        arm_joints = state_list[:-1]  # 前6个是关节角度
        gripper_val = state_list[-1]   # 最后一个是夹爪

        # 🔥 透传模式下必须使用 rm_movej_canfd，不是 rm_movej！
        # 参数: (joints, follow=False, expand=0, retry_count=0, retry_interval=0)
        self.arm.rm_movej_canfd(arm_joints, False, 0, 0, 0)

        # 夹爪控制
        target_pos = int(gripper_val)
        target_pos = max(0, min(1000, target_pos))
        self.arm.rm_set_gripper_position(target_pos, False, 1)

        # ⚡ 透传模式不需要等待

    # =========================================================================
    # 3. 🔥 核心修复: 跳过末端位姿读取 (解决 visualize 报错)
    # =========================================================================
    def get_ee_state(self):
        """
        覆盖基类的公有方法，直接返回 None。
        这会告诉 BiBaseRobot: '我不支持末端读取，请跳过可视化'。
        从而彻底避开 BaseRobot 里的单位转换报错。
        """
        return None

    def _get_ee_state(self): pass
    def _set_ee_state(self, state): pass
    # 注意: 不要覆盖 _set_joint_state，上面已经正确实现了！