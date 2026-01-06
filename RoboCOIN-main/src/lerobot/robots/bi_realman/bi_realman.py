# """
# BiRealman robot implementation.
# """

# from ..bi_base_robot import BiBaseRobot
# from .configuration_bi_realman import BiRealmanConfig

# from ..realman import Realman, RealmanConfig


# class BiRealman(BiBaseRobot):
#     """
#     BiRealman robot implementation.
#     Params:
#     - config: BiRealmanConfig
#     """

#     config_class = BiRealmanConfig
#     name = "bi_realman"

#     def __init__(self, config: BiRealmanConfig):
#         super().__init__(config)
#         self.config = config
    
#     def _prepare_robots(self):
#         """
#         Prepare the left and right Realman robots.
#         Initializes two Realman instances with respective configurations.
#         """
#         left_config = RealmanConfig(
#             ip=self.config.ip_left,
#             port=self.config.port_left,
#             block=self.config.block,
#             wait_second=self.config.wait_second,
#             velocity=self.config.velocity,
#             joint_names=self.config.joint_names,
#             init_type=self.config.init_type,
#             init_state=self.config.init_state_left,
#             joint_units=self.config.joint_units,
#             pose_units=self.config.pose_units,
#             model_joint_units=self.config.model_joint_units,
#             delta_with=self.config.delta_with,
#             visualize=False,
#             id=f"{self.config.id}_left" if self.config.id else None,
#             cameras={},
#         )
#         right_config = RealmanConfig(
#             ip=self.config.ip_right,
#             port=self.config.port_right,
#             block=self.config.block,
#             wait_second=self.config.wait_second,
#             velocity=self.config.velocity,
#             joint_names=self.config.joint_names,
#             init_type=self.config.init_type,
#             init_state=self.config.init_state_right,
#             joint_units=self.config.joint_units,
#             pose_units=self.config.pose_units,
#             model_joint_units=self.config.model_joint_units,
#             delta_with=self.config.delta_with,
#             visualize=False,
#             id=f"{self.config.id}_right" if self.config.id else None,
#             cameras={},
#         )
#         self.left_robot = Realman(left_config)
#         self.right_robot = Realman(right_config)

from ..bi_base_robot import BiBaseRobot
from .configuration_bi_realman import BiRealmanConfig

from ..realman import Realman, RealmanConfig


class BiRealman(BiBaseRobot):
    """
    BiRealman robot implementation.
    Params:
    - config: BiRealmanConfig
    """

    config_class = BiRealmanConfig
    name = "bi_realman"

    def __init__(self, config: BiRealmanConfig):
        super().__init__(config)
        self.config = config
    
    def _prepare_robots(self):
        """
        Prepare the left and right Realman robots.
        Initializes two Realman instances with respective configurations,
        including specific settings for Zhixing grippers.
        """
        # === 配置左臂 (Left Arm) ===
        left_config = RealmanConfig(
            ip=self.config.ip_left,
            port=self.config.port_left,
            block=self.config.block,
            wait_second=self.config.wait_second,
            velocity=self.config.velocity,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_left,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            delta_with=self.config.delta_with,
            visualize=False,
            id=f"{self.config.id}_left" if self.config.id else None,
            cameras={},
            
            # 传递左臂夹爪参数到单臂实例
            use_zhixing_gripper=self.config.use_zhixing_gripper_left,
            gripper_port=self.config.gripper_port_left,
            gripper_baudrate=self.config.gripper_baudrate_left,
            gripper_device_id=self.config.gripper_device_id_left,
            gripper_enable_reg=self.config.gripper_enable_reg_left,
            gripper_pos_reg=self.config.gripper_pos_reg_left,
            gripper_trigger_reg=self.config.gripper_trigger_reg_left,
        )

        # === 配置右臂 (Right Arm) ===
        right_config = RealmanConfig(
            ip=self.config.ip_right,
            port=self.config.port_right,
            block=self.config.block,
            wait_second=self.config.wait_second,
            velocity=self.config.velocity,
            joint_names=self.config.joint_names,
            init_type=self.config.init_type,
            init_state=self.config.init_state_right,
            joint_units=self.config.joint_units,
            pose_units=self.config.pose_units,
            model_joint_units=self.config.model_joint_units,
            delta_with=self.config.delta_with,
            visualize=False,
            id=f"{self.config.id}_right" if self.config.id else None,
            cameras={},

            # 传递右臂夹爪参数到单臂实例
            use_zhixing_gripper=self.config.use_zhixing_gripper_right,
            gripper_port=self.config.gripper_port_right,
            gripper_baudrate=self.config.gripper_baudrate_right,
            gripper_device_id=self.config.gripper_device_id_right,
            gripper_enable_reg=self.config.gripper_enable_reg_right,
            gripper_pos_reg=self.config.gripper_pos_reg_right,
            gripper_trigger_reg=self.config.gripper_trigger_reg_right,
        )

        self.left_robot = Realman(left_config)
        self.right_robot = Realman(right_config)