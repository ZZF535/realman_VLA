# config_realman_dual_arm.py
from dataclasses import dataclass, field
import numpy as np
import dataclasses
# from lerobot.cameras.realsense import IntelRealSenseCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from ..config import RobotConfig
# from lerobot.utils import get_logger
import logging
from lerobot.motors import Motor, MotorCalibration, MotorNormMode, RMMotorConfig

@RobotConfig.register_subclass("realman_dual_arm")
@dataclass
@dataclasses.dataclass

class RealManDualArmConfig(RobotConfig):
    """
    睿尔曼双臂机器人配置类
    包含：2x RM65-B-V 机械臂, 2x 知行夹爪, 3x Realsense D435
    """
    # === 机械臂通讯配置 ===
    left_arm_ip: str = "192.168.1.18"   # 左臂 IP
    right_arm_ip: str = "192.168.1.19"  # 右臂 IP
    arm_port: int = 8080                # SDK 默认端口

    # === 夹爪配置 ===
    # 知行夹爪通常是 0(松开)-1000(闭合) 或者反之，需根据实际情况调整 min/max
    gripper_open_pos: int = 0
    gripper_closed_pos: int = 1000

    # === 安全限制 ===
    # 限制单步最大关节变动幅度 (弧度)，防止模型输出剧烈抖动损坏硬件
    max_relative_target: float | None = 0.05  

    # === 相机配置 ===
    # 这里定义了你的三个 Realsense 相机
    # 如果你想直接使用官方数据集训练，可能需要把 "head" 改名为 "phone"
    cameras: dict[str, RealSenseCameraConfig] = field(
        default_factory=lambda: {
            # 头部相机 (主视角)
            # "head": IntelRealSenseCameraConfig(
            #     width=640, height=480, fps=30, 
            #     serial_number="YOUR_HEAD_SERIAL_NUMBER" # <--- 请填入实际序列号
            # ),
            # 左手腕相机
            "wrist_left": RealSenseCameraConfig(
                width=640, height=480, fps=30, 
                serial_number_or_name ="243722073715"  # <--- 请填入实际序列号
            ),
            # 右手腕相机
            "wrist_right": RealSenseCameraConfig(
                width=640, height=480, fps=30, 
                serial_number_or_name ="346522074543" # <--- 请填入实际序列号
            ),
        }
    )


# === 电机安全限位 (MotorConfig) ===
    # 基于 RM65-B 官方手册的物理极限 (单位：弧度)
    # 作用：防止模型输出离谱角度导致撞机MotorConfig
    motors: dict[str, RMMotorConfig] = dataclasses.field(
        default_factory=lambda: {
            # --- 左臂 (6轴) ---
            "left_joint1": RMMotorConfig(name="left_joint1", type="rm65", min_pos=-3.10, max_pos=3.10),
            "left_joint2": RMMotorConfig(name="left_joint2", type="rm65", min_pos=-2.26, max_pos=2.26),
            "left_joint3": RMMotorConfig(name="left_joint3", type="rm65", min_pos=-2.35, max_pos=2.35),
            "left_joint4": RMMotorConfig(name="left_joint4", type="rm65", min_pos=-3.10, max_pos=3.10),
            "left_joint5": RMMotorConfig(name="left_joint5", type="rm65", min_pos=-2.23, max_pos=2.23),
            "left_joint6": RMMotorConfig(name="left_joint6", type="rm65", min_pos=-6.28, max_pos=6.28),
            "left_gripper": RMMotorConfig(name="left_gripper", type="ctag", min_pos=0.0, max_pos=1.0),
            
            # --- 右臂 (6轴) ---
            "right_joint1": RMMotorConfig(name="right_joint1", type="rm65", min_pos=-3.10, max_pos=3.10),
            "right_joint2": RMMotorConfig(name="right_joint2", type="rm65", min_pos=-2.26, max_pos=2.26),
            "right_joint3": RMMotorConfig(name="right_joint3", type="rm65", min_pos=-2.35, max_pos=2.35),
            "right_joint4": RMMotorConfig(name="right_joint4", type="rm65", min_pos=-3.10, max_pos=3.10),
            "right_joint5": RMMotorConfig(name="right_joint5", type="rm65", min_pos=-2.23, max_pos=2.23),
            "right_joint6": RMMotorConfig(name="right_joint6", type="rm65", min_pos=-6.28, max_pos=6.28),
            "right_gripper": RMMotorConfig(name="right_gripper", type="ctag", min_pos=0.0, max_pos=1.0),
        }
    )
    # === 数据特征定义 (Features) ===
    # 对应 info.json 中的 shape: (14,)
    # 顺序约定：[左臂6轴, 左夹爪, 右臂6轴, 右夹爪]
    features: dict = field(
        default_factory=lambda: {
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": [
                    # 左臂
                    "left_joint1", 
                    "left_joint2", 
                    "left_joint3", 
                    "left_joint4", 
                    "left_joint5", 
                    "left_joint6",
                    "left_gripper",
                    # 右臂
                    "right_joint1", 
                    "right_joint2", 
                    "right_joint3", 
                    "right_joint4", 
                    "right_joint5", 
                    "right_joint6",
                    "right_gripper",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": [
                    "left_joint1", 
                    "left_joint2", 
                    "left_joint3", 
                    "left_joint4", 
                    "left_joint5", 
                    "left_joint6",
                    "left_gripper",
                    "right_joint1", 
                    "right_joint2", 
                    "right_joint3", 
                    "right_joint4", 
                    "right_joint5", 
                    "right_joint6",
                    "right_gripper",
                ],
            },
        }
    )

    # 归一化参数 (建议运行 compute_stats.py 自动生成，这里给个默认值防止报错)
    default_calibration: dict = field(
        default_factory=lambda: {
            "action": {"mean": [0.0]*14, "std": [1.0]*14},
            "observation.state": {"mean": [0.0]*14, "std": [1.0]*14}
        }
    )