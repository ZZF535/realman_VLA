from dataclasses import dataclass, field
from typing import Dict

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.motors import Motor, MotorNormMode


@TeleoperatorConfig.register_subclass("so101_leader_dora")
@dataclass
class SO101LeaderDoraTeleoperatorConfig(TeleoperatorConfig):
    use_degrees = True
    norm_mode_body = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
 
    motors: Dict[str, Motor] = field(
        default_factory=lambda norm_mode_body=norm_mode_body: {
            "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
            "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
            "elbow_flex": Motor(3, "sts3215", norm_mode_body),
            "wrist_flex": Motor(4, "sts3215", norm_mode_body),
            "wrist_roll": Motor(5, "sts3215", norm_mode_body),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
    )
