from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import abc
import draccus

  
@dataclass
class ArmInfo:
    name: str = ""
    type: str = ""
    start_pose: List[float] = field(default_factory=list)
    joint_p_limit: List[float] = field(default_factory=list)
    joint_n_limit: List[float] = field(default_factory=list)
    is_connect: bool = False
 
@dataclass
class ArmStatus:
    number: int = 0
    information: List[ArmInfo] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.information:  # 如果 information 为空，则 number 设为 0
            self.number = 0
        else:
            self.number = len(self.information)
 
@dataclass
class Specifications:
    end_type: str = "Default"
    fps: int = 30
    arm: Optional[ArmStatus] = None

@dataclass
class TeleoperatorStatus(draccus.ChoiceRegistry, abc.ABC):
    device_name: str = "Default"
    device_body: str = "Default"
    specifications: Specifications = field(default_factory=Specifications)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    
    def to_dict(self) -> dict:
        return asdict(self)
 
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

@TeleoperatorStatus.register_subclass("so101_leader_dora")
@dataclass
class SO101LeaderDoraTeleoperatorStatus(TeleoperatorStatus):
    device_name: str = "SO101_V1"
    device_body: str = "SO101"

    def __post_init__(self):
        self.specifications.end_type = "二指夹爪"
        self.specifications.fps = 30

        self.specifications.arm = ArmStatus(
            information=[
                ArmInfo(
                    name="leader",
                    type="so101 主臂 5DOF",
                    start_pose=[],
                    joint_p_limit=[90.0, 90.0, 90.0, 90.0, 90.0],
                    joint_n_limit=[-90.0, -90.0, -90.0, -90.0, -90.0],
                    is_connect=False
                ),
                ArmInfo(
                    name="follower",
                    type="so101 从臂 5DOF",
                    start_pose=[],
                    joint_p_limit=[90.0, 90.0, 90.0, 90.0, 90.0],
                    joint_n_limit=[-90.0, -90.0, -90.0, -90.0, -90.0],
                    is_connect=False
                ),
            ]
        )
