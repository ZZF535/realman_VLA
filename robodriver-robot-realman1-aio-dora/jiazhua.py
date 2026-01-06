import sys
import os
import time

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from Robotic_Arm.rm_robot_interface import *

# def log(msg):
#     print(msg, flush=True)

# def main():
#     arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
#     handle = arm.rm_create_robot_arm("169.254.128.18", 8080)
#     log(f"current c api version: {arm.rm_get_c_api_version()[1] if hasattr(arm,'rm_get_c_api_version') else 'unknown'}")
#     log(f"机械臂 handle.id = {handle.id}")

#     try:
#         # 1. 切到真实模式（有些 IO / 外设在仿真模式下会直接拒绝）
#         if hasattr(arm, "rm_set_arm_run_mode"):
#             ret_mode = arm.rm_set_arm_run_mode(1)
#             log(f"rm_set_arm_run_mode(1) 返回: {ret_mode}")

#         # 2. 配置末端 RS485 为 Modbus RTU 主站
#         ret_modbus = arm.rm_set_modbus_mode(1, 115200, 2)
#         log(f"rm_set_modbus_mode(port=1,115200,2) 返回: {ret_modbus}")
#         time.sleep(0.5)

#         # 3. 使能夹爪：写 256 = 1
#         #    这里明确给 num=1
#         enable_params = rm_peripheral_read_write_params_t(
#             1,      # port: 末端 RS485
#             256,    # address
#             1,      # device: 假设从站地址为 1
#             1       # num: 写 1 个寄存器
#         )
#         ret_en = arm.rm_write_single_register(enable_params, 1)
#         log(f"rm_write_single_register(256,1) 返回: {ret_en}")

#         time.sleep(0.5)

#         # 4. 尝试读回 256 看看值是不是 1（确认总线/设备有没有回数据）
#         if hasattr(arm, "rm_read_holding_registers"):
#             ret_r, data_256 = arm.rm_read_holding_registers(enable_params)
#             log(f"rm_read_holding_registers(256) 返回: code={ret_r}, data={data_256}")
#         else:
#             log("rm_read_holding_registers 在当前版本中不存在")

#         # 5. 测试张开
#         log("\n>>> 测试：张开夹爪")
#         open_params = rm_peripheral_read_write_params_t(
#             1,      # port
#             258,    # address
#             1,      # device
#             2       # num: 写两个寄存器 [0,1000]
#         )
#         datas_open = [0, 1000]
#         ret_open = arm.rm_write_registers(open_params, datas_open)
#         log(f"rm_write_registers(258,[0,1000]) 返回: {ret_open}")

#         # 触发 264: 0 -> 1
#         trig_params = rm_peripheral_read_write_params_t(
#             1,      # port
#             264,    # address
#             1,      # device
#             1       # num
#         )
#         ret_264_0 = arm.rm_write_single_register(trig_params, 0)
#         log(f"写 264 = 0 返回: {ret_264_0}")
#         time.sleep(0.05)
#         ret_264_1 = arm.rm_write_single_register(trig_params, 1)
#         log(f"写 264 = 1 返回: {ret_264_1}")

#         time.sleep(1.0)
#         log("如一切正常，此时夹爪应当张开。")

#     finally:
#         log(">>> 断开机械臂连接")
#         arm.rm_delete_robot_arm()

# if __name__ == "__main__":
#     main()

import sys
import os
import time
from itertools import count

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Robotic_Arm.rm_robot_interface import *

# 实例化RoboticArm类
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

# 创建机械臂连接，打印连接id
handle = arm.rm_create_robot_arm("169.254.128.18", 8080)
print(handle.id)

print(arm.rm_set_modbus_mode(1,115200,2))
# 通过机械臂末端RS485接口写数据1，起始地址为256， 外设设备地址为1
time.sleep(3)


write_params = rm_peripheral_read_write_params_t(1, 256, 1)
print(arm.rm_write_single_register(write_params, 1))
print("知行90D夹爪使能")
time.sleep(3)
# #张开，需先设置夹爪行走位置，再通过264寄存器触发运动
write_params = rm_peripheral_read_write_params_t(1, 258, 1)
print(arm.rm_write_single_register(write_params, 0))
# datas=[0,500]
write_params = rm_peripheral_read_write_params_t(1, 259, 1)
print(arm.rm_write_single_register(write_params, 1000))
print("夹爪张开位置已设置")
#接上述张开指令，寄存器地址264用于运动触发
write_params = rm_peripheral_read_write_params_t(1, 264, 1)
print(arm.rm_write_single_register(write_params, 1))
time.sleep(3)
print("夹爪已张开")

# # 闭合，需先设置夹爪行走位置，再通过264寄存器触发运动
# write_params = rm_peripheral_read_write_params_t(1, 258, 1, 2)
# datas=[0,0]
# print(arm.rm_write_registers(write_params, datas))
# print("夹爪闭合位置已设置")
# # #接上述闭合指令，寄存器地址264用于运动触发
# write_params = rm_peripheral_read_write_params_t(1, 264, 1)
# print(arm.rm_write_single_register(write_params, 1))
# time.sleep(1)
# print("夹爪已闭合")

#夹爪功能循环测试10次
# count=1
# for i in range(10):
#     print(f"第 {count} 次循环")
#     # 张开，需先设置夹爪行走位置，再通过264寄存器触发运动
#     write_params = rm_peripheral_read_write_params_t(1, 258, 1, 2)
#     datas = [0, 500]
#     print(arm.rm_write_registers(write_params, datas))
#     print("夹爪张开位置已设置")
#     # 接上述张开指令，寄存器地址264用于运动触发
#     write_params = rm_peripheral_read_write_params_t(1, 264, 1)
#     print(arm.rm_write_single_register(write_params, 1))
#     time.sleep(1)
#     print("夹爪已张开")

#     # 闭合，需先设置夹爪行走位置，再通过264寄存器触发运动
#     write_params = rm_peripheral_read_write_params_t(1, 258, 1, 2)
#     datas = [0, 0]
#     print(arm.rm_write_registers(write_params, datas))
#     print("夹爪闭合位置已设置")
#     # 接上述闭合指令，寄存器地址264用于运动触发
#     write_params = rm_peripheral_read_write_params_t(1, 264, 1)
#     print(arm.rm_write_single_register(write_params, 1))
#     time.sleep(1)
#     print("夹爪已闭合")
#     print(f"夹爪张开闭合功能已完成测试 {count} 次")

#     count +=1
#     if i ==10:
#         break

arm.rm_delete_robot_arm()
