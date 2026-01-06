# import time
# import sys

# try:
#     from Robotic_Arm.rm_robot_interface import *
# except ImportError:
#     print("Error: 找不到 Robotic_Arm 库")
#     sys.exit(1)

# def test_drag_pro():
#     IP = "169.254.128.18"  # <--- 确认你的 IP
#     PORT = 8080

#     print(f"连接机械臂 {IP}...")
#     arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#     handle = arm.rm_create_robot_arm(IP, PORT)
    
#     if handle.id == -1:
#         print("连接失败！")
#         return

#     # 1. 清除错误 (重要)
#     print("清除错误...")
#     if hasattr(arm, 'rm_clear_err'):
#         arm.rm_clear_err()
    
#     # 2. 设置为轨迹模式
#     print("设置运行模式为 0 (Trajectory)...")
#     arm.rm_set_arm_run_mode(0)
#     time.sleep(1.0)

#     # 3. [关键] 设置拖动灵敏度 (仅对标准版有效)
#     # 范围 1-100，默认可能是 50 或更低，导致感觉硬
#     print("设置拖动灵敏度为 90%...")
#     if hasattr(arm, 'rm_set_drag_teach_sensitivity'):
#         ret = arm.rm_set_drag_teach_sensitivity(90)
#         print(f"灵敏度设置结果: {ret}")
#     else:
#         print("警告: 当前 SDK 版本不支持设置灵敏度")

#     # 4. 开启拖拽 (不录制轨迹)
#     print("启动拖拽示教 (rm_start_drag_teach)...")
#     ret = arm.rm_start_drag_teach(0)
#     print(f"启动结果: {ret}")

#     if ret == 0:
#         print("\n=== 拖拽模式已开启！请尝试拖动 ===")
#         print("如果依然很硬，请检查示教器设置或机械臂负载配置。")
        
#         for i in range(15):
#             print(f"测试中... {15-i}s")
#             time.sleep(1)
            
#         print("停止拖拽...")
#         arm.rm_stop_drag_teach()
#     else:
#         print(f"开启失败，错误码: {ret}")
#         print("常见原因：机械臂处于报错状态（红灯）、急停被按下、或不在 Mode 0。")

#     arm.rm_destroy()

# if __name__ == "__main__":
#     test_drag_pro()

from Robotic_Arm.rm_robot_interface import *

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("169.254.128.18", 8080)
print(handle.id)

# 关节阻塞运动到[0, 20, 70, 0, 90, 0]
print(arm.rm_movej([171.568,119.904,12.745,-92.657,3.754,1.045], 10, 0, 0, 1))

arm.rm_delete_robot_arm()