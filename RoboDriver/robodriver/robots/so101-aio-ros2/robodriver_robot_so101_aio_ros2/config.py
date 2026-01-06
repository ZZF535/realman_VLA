from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("so101_aio_ros2")
@dataclass
class So101AioRos2RobotConfig(RobotConfig):
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    # 按组件分组：{ comp_id: { joint_name: Motor, ... }, ... }
    leader_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "leader_arm": {
                "joint_0": Motor(1, "robot_motor", norm_mode_body),
                "joint_1": Motor(2, "robot_motor", norm_mode_body),
                "joint_2": Motor(3, "robot_motor", norm_mode_body),
                "joint_3": Motor(4, "robot_motor", norm_mode_body),
                "joint_4": Motor(5, "robot_motor", norm_mode_body),
                "joint_5": Motor(6, "robot_motor", norm_mode_body),
            },
        }
    )

    follower_motors: Dict[str, Dict[str, Motor]] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "follower_arm": {
                "joint_0": Motor(1, "robot_motor", norm_mode_body),
                "joint_1": Motor(2, "robot_motor", norm_mode_body),
                "joint_2": Motor(3, "robot_motor", norm_mode_body),
                "joint_3": Motor(4, "robot_motor", norm_mode_body),
                "joint_4": Motor(5, "robot_motor", norm_mode_body),
                "joint_5": Motor(6, "robot_motor", norm_mode_body),
            },
        }
    )

    cameras: Dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image_top": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
        }
    )

    use_videos: bool = False

    microphones: Dict[str, int] = field(default_factory=lambda: {}
    )
