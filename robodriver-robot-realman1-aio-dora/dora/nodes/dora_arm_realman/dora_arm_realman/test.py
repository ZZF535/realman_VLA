from Robotic_Arm.rm_robot_interface import *
import numpy as np
import time 
import sys
import os
import time
from itertools import count

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm("169.254.128.19", 8080)
flag, gripper_dict = arm.rm_get_rm_plus_state_info()
print(gripper_dict)
gripper_actpos = np.array([gripper_dict['pos'][0]]).astype(np.float64)
print(gripper_actpos)
i = arm.rm_set_gripper_position(500, False ,0)
print(i)
time.sleep(2)
flag2, gripper_dict2 = arm.rm_get_rm_plus_state_info()

gripper_actpos2 = np.array([gripper_dict2['pos'][0]]).astype(np.float64)
print(gripper_actpos2)
# j = arm.rm_set_gripper_position(300, False, 0)
# print(j)

# import pandas as pd

# # 1. 读取文件
# df = pd.read_parquet('/home/robot/DoRobot/dataset/20251208/dev/打开洗衣机取出衣服_洗衣服Copy_890/打开洗衣机取出衣服_洗衣服Copy_890_85551/data/chunk-000/episode_000000.parquet')

# # 2. 查看所有列名，确认顺序
# print("所有列名:", df.columns.tolist())

# # 3. 读取第 7 列数据 (索引为 6)
# col7_data = df.iloc[:, 6]

# # 打印第 7 列的名称和前 5 行数据
# print(f"\n第七列的名称是: {df.columns[6]}")
# print("前 5 行数据:")
# print(col7_data.head())


# from Robotic_Arm.rm_robot_interface import *
# import numpy as np
# import time 
# import sys
# import os

# # 1. 连接机械臂
# print("🔌 正在连接...")
# arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
# handle = arm.rm_create_robot_arm("169.254.128.18", 8080)

# if handle.id <= 0:
#     print("❌ 连接失败，请检查 IP (169.254.128.18) 是否正确？还是应该连 .19？")
#     sys.exit()

# print(f"✅ 连接成功 (Handle ID: {handle.id})")

# # ==========================================
# # ⚡️ 核心修复：重启后必须重新给末端上电！
# # ==========================================
# print("⚡️ 正在开启末端 24V 电源...")
# # 注意：根据你之前的报错，你的SDK版本 set_tool_voltage 只接受一个参数
# ret = arm.rm_set_tool_voltage(3) 
# print(f"   电源指令返回值: {ret} (0=成功)")

# print("⏳ 等待 3 秒让夹爪启动...")
# time.sleep(3)
# # ==========================================

# # 2. 读取当前状态
# print("\n🔍 读取当前状态...")
# flag, gripper_dict = arm.rm_get_rm_plus_state_info()
# if flag == 0:
#     print(f"   原始数据: {gripper_dict}")
#     if 'pos' in gripper_dict:
#         gripper_actpos = np.array([gripper_dict['pos'][0]]).astype(np.float64)
#         print(f"   当前位置: {gripper_actpos}")
# else:
#     print(f"❌ 读取状态失败 (Flag: {flag}) - 可能还是没通电或没接好")

# # 3. 动作测试 1
# print("\n👉 尝试移动到 600...")
# i = arm.rm_set_gripper_position(600, False ,5)
# print(f"   指令返回值: {i}")
# time.sleep(2)

# # 4. 再次读取
# flag2, gripper_dict2 = arm.rm_get_rm_plus_state_info()
# if flag2 == 0 and 'pos' in gripper_dict2:
#     gripper_actpos2 = np.array([gripper_dict2['pos'][0]]).astype(np.float64)
#     print(f"   移动后位置: {gripper_actpos2}")

# # 5. 动作测试 2
# print("\n👉 尝试移动到 300...")
# j = arm.rm_set_gripper_position(300, False, 1)
# print(f"   指令返回值: {j}")

# # 断开连接
# arm.rm_delete_robot_arm()