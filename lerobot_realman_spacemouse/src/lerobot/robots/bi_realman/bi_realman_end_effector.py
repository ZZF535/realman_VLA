# from functools import cached_property
# from typing import Any

# from lerobot.cameras import make_cameras_from_configs
# from lerobot.errors import DeviceNotConnectedError
# from lerobot.robots.robot import Robot

# from .configuration_bi_realman import BiRealmanEndEffectorConfig
# from ..realman import RealmanEndEffector, RealmanEndEffectorConfig
# from ..misc import get_visualizer


# class BiRealmanEndEffector(Robot):
#     """
#     BiRealmanEndEffector is a robot class for controlling the end effector of the BiRealman robot using end-effector control.
#     """
    
#     config_class =  BiRealmanEndEffectorConfig
#     name = "bi_realman_end_effector"

#     def __init__(self, config: BiRealmanEndEffectorConfig):
#         self.config = config
#         super().__init__(config)

#         if len(config.init_state) < 10:
#             init_state_left = init_state_right = config.init_state
#         else:
#             init_state_left = config.init_state[:len(config.init_state) // 2]
#             init_state_right = config.init_state[len(config.init_state) // 2:]

#         left_arm_config = RealmanEndEffectorConfig(
#             id=f"{config.id}_left" if config.id else None,
#             port=config.port_left,
#             ip=config.ip_left,
#             cameras={},
#             init_state=init_state_left,
#             control_mode=config.control_mode,
#             delta_with_previous=config.delta_with_previous,
#             base_euler=config.base_euler,
#             visualize=False,
#         )
#         right_arm_config = RealmanEndEffectorConfig(
#             id=f"{config.id}_right" if config.id else None,
#             port=config.port_right,
#             ip=config.ip_right,
#             cameras={},
#             init_state=init_state_right,
#             control_mode=config.control_mode,
#             delta_with_previous=config.delta_with_previous,
#             base_euler=config.base_euler,
#             visualize=False,
#         )

#         self.left_arm = RealmanEndEffector(left_arm_config)
#         self.right_arm = RealmanEndEffector(right_arm_config)
#         self.cameras = make_cameras_from_configs(config.cameras)
        
#         # 修复: 使用 init_state 而不是 init_ee_state
#         self.visualizer = get_visualizer(list(self._cameras_ft.keys()), 
#                                          ['arm_left', 'arm_right'], 
#                                          [self.left_arm.standardization.input_transform(config.init_state), 
#                                           self.right_arm.standardization.input_transform(config.init_state)], 
#                                          'ee_absolute') \
#                                    if config.visualize else None
      
#     @property
#     def _motors_ft(self) -> dict[str, type]:
#         # 1. 获取基础关节特征 (14维)
#         left_ft = {f"left_{each}": float for each in self.left_arm._motors_ft.keys()}
#         right_ft = {f"right_{each}": float for each in self.right_arm._motors_ft.keys()}
        
#         # 2. 补全末端位姿特征 (12维: 2个手臂 x 6个自由度) -> 总共 26维
#         ee_keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
#         for k in ee_keys:
#             left_ft[f"left_{k}"] = float
#             right_ft[f"right_{k}"] = float
            
#         return {**left_ft, **right_ft}
      
#     @property
#     def _cameras_ft(self) -> dict[str, tuple]:
#         return {
#             cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
#         }
      
#     @cached_property
#     def observation_features(self) -> dict:
#         return {**self._motors_ft, **self._cameras_ft}
      
#     @property
#     def action_features(self) -> dict[str, Any]:
#         return {
#             each: float for each in [
#                 'left_x', 'left_y', 'left_z', 'left_roll', 'left_pitch', 'left_yaw', 'left_gripper',
#                 'right_x', 'right_y', 'right_z', 'right_roll', 'right_pitch', 'right_yaw', 'right_gripper',
#             ]
#         }
      
#     @property
#     def is_connected(self) -> bool:
#         return self.left_arm.is_connected and self.right_arm.is_connected and all(cam.is_connected for cam in self.cameras.values())
      
#     def connect(self):
#         self.left_arm.connect()
#         self.right_arm.connect()
#         for cam in self.cameras.values():
#             cam.connect()
      
#     def is_calibrated(self) -> bool:
#         return self.left_arm.is_calibrated and self.right_arm.is_calibrated
      
#     def calibrate(self) -> None:
#         self.left_arm.calibrate()
#         self.right_arm.calibrate()
      
#     def configure(self) -> None:
#         self.left_arm.configure()
#         self.right_arm.configure()
      
#     def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
#         if not self.is_connected:
#             raise DeviceNotConnectedError(f"{self} is not connected.")
          
#         left_action = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
#         right_action = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}

#         send_action_left = self.left_arm.send_action(left_action)
#         send_action_right = self.right_arm.send_action(right_action)

#         if self.visualizer:
#             left_state = self.left_arm.standardization.input_transform(self.left_arm._get_ee_state())
#             right_state = self.right_arm.standardization.input_transform(self.right_arm._get_ee_state())
#             observation = self.get_observation()
#             images = [observation[cam_key] for cam_key in self._cameras_ft.keys()]
#             self.visualizer.add(images, [left_state, right_state])
#             self.visualizer.plot()

#         send_action_left = {f"left_{k}": v for k, v in send_action_left.items()}
#         send_action_right = {f"right_{k}": v for k, v in send_action_right.items()}
#         return {**send_action_left, **send_action_right}
      
#     def get_observation(self) -> dict[str, Any]:
#         if not self.is_connected:
#             raise DeviceNotConnectedError(f"{self} is not connected.")
          
#         obs_dict = {}

#         # 1. 获取关节状态
#         left_obs = self.left_arm.get_observation()
#         obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

#         right_obs = self.right_arm.get_observation()
#         obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

#         # 2. 补全末端状态 (EE Pose)
#         # _get_ee_state 返回 [x, y, z, roll, pitch, yaw, gripper]
#         left_ee = self.left_arm._get_ee_state()
#         right_ee = self.right_arm._get_ee_state()
        
#         ee_keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
#         for i, k in enumerate(ee_keys):
#             obs_dict[f"left_{k}"] = left_ee[i]
#             obs_dict[f"right_{k}"] = right_ee[i]

#         # 3. 获取相机图像
#         for cam_key, cam in self.cameras.items():
#             outputs = cam.async_read()
#             if isinstance(outputs, dict):
#                 for key, value in outputs.items():
#                     obs_dict[f"{cam_key}_{key}"] = value
#             else:
#                 obs_dict[cam_key] = outputs
#         return obs_dict
      
#     def disconnect(self):
#         self.left_arm.disconnect()
#         self.right_arm.disconnect()
#         for cam in self.cameras.values():
#             cam.disconnect()
#         print("BiRealman robot disconnected.")


from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .configuration_bi_realman import BiRealmanEndEffectorConfig
from ..realman import RealmanEndEffector, RealmanEndEffectorConfig
from ..misc import get_visualizer


class BiRealmanEndEffector(Robot):
    """
    BiRealmanEndEffector is a robot class for controlling the end effector of the BiRealman robot using end-effector control.
    """
    
    config_class =  BiRealmanEndEffectorConfig
    name = "bi_realman_end_effector"

    def __init__(self, config: BiRealmanEndEffectorConfig):
        # 修复1: 添加 self.config 初始化
        self.config = config
        super().__init__(config)

        if len(config.init_state) < 10:
            init_state_left = init_state_right = config.init_state
        else:
            init_state_left = config.init_state[:len(config.init_state) // 2]
            init_state_right = config.init_state[len(config.init_state) // 2:]

        left_arm_config = RealmanEndEffectorConfig(
            id=f"{config.id}_left" if config.id else None,
            port=config.port_left,
            ip=config.ip_left,
            cameras={},
            init_state=init_state_left,
            control_mode=config.control_mode,
            delta_with_previous=config.delta_with_previous,
            base_euler=config.base_euler,
            visualize=False,
        )
        right_arm_config = RealmanEndEffectorConfig(
            id=f"{config.id}_right" if config.id else None,
            port=config.port_right,
            ip=config.ip_right,
            cameras={},
            init_state=init_state_right,
            control_mode=config.control_mode,
            delta_with_previous=config.delta_with_previous,
            base_euler=config.base_euler,
            visualize=False,
        )

        self.left_arm = RealmanEndEffector(left_arm_config)
        self.right_arm = RealmanEndEffector(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # 修复2: 使用 config.init_state 替代不存在的 config.init_ee_state
        self.visualizer = get_visualizer(list(self._cameras_ft.keys()), 
                                         ['arm_left', 'arm_right'], 
                                         [self.left_arm.standardization.input_transform(config.init_state), 
                                          self.right_arm.standardization.input_transform(config.init_state)], 
                                         'ee_absolute') \
                                   if config.visualize else None
      
    @property
    def _motors_ft(self) -> dict[str, type]:
        # 修复3(核心): 补全维度，让 observation 包含末端位姿 (14 + 12 = 26维)
        left_ft = {f"left_{each}": float for each in self.left_arm._motors_ft.keys()}
        right_ft = {f"right_{each}": float for each in self.right_arm._motors_ft.keys()}
        
        # 强制加上 xyz 和 euler 角度
        ee_keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        for k in ee_keys:
            left_ft[f"left_{k}"] = float
            right_ft[f"right_{k}"] = float
            
        return {**left_ft, **right_ft}
      
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
      
    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}
      
    @property
    def action_features(self) -> dict[str, Any]:
        # 修复4: 直接复用 _motors_ft，确保 Action 也是 26 维，与 Model 匹配！
        return self._motors_ft
      
    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected and all(cam.is_connected for cam in self.cameras.values())
      
    def connect(self):
        self.left_arm.connect()
        self.right_arm.connect()
        for cam in self.cameras.values():
            cam.connect()
      
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated
      
    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()
      
    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()
      
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
          
        left_action = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
        right_action = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        if self.visualizer:
            left_state = self.left_arm.standardization.input_transform(self.left_arm._get_ee_state())
            right_state = self.right_arm.standardization.input_transform(self.right_arm._get_ee_state())
            observation = self.get_observation()
            images = [observation[cam_key] for cam_key in self._cameras_ft.keys()]
            self.visualizer.add(images, [left_state, right_state])
            self.visualizer.plot()

        send_action_left = {f"left_{k}": v for k, v in send_action_left.items()}
        send_action_right = {f"right_{k}": v for k, v in send_action_right.items()}
        return {**send_action_left, **send_action_right}
      
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
          
        obs_dict = {}

        # 1. 获取关节状态
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # 2. 获取并填入末端位姿 (核心修复点)
        # _get_ee_state 通常返回 [x, y, z, roll, pitch, yaw, gripper]
        left_ee = self.left_arm._get_ee_state()
        right_ee = self.right_arm._get_ee_state()
        
        ee_keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        for i, k in enumerate(ee_keys):
            obs_dict[f"left_{k}"] = left_ee[i]
            obs_dict[f"right_{k}"] = right_ee[i]

        # 3. 获取相机图像
        for cam_key, cam in self.cameras.items():
            outputs = cam.async_read()
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    obs_dict[f"{cam_key}_{key}"] = value
            else:
                obs_dict[cam_key] = outputs
        return obs_dict
      
    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        print("BiRealman robot disconnected.")