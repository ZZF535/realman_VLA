from Robotic_Arm.rm_robot_interface import *

# 初始化为三线程模式
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("169.254.128.19", 8080)
print(handle.id)

print(arm.rm_get_current_arm_state())
# 结束机械臂控制，删除指定机械臂对象
arm.rm_delete_robot_arm()

