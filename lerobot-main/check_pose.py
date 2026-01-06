import time
import sys

# 右臂 IP (改成你正在调试的那个臂)
ROBOT_IP = "169.254.128.19" 

try:
    from Robotic_Arm.rm_robot_interface import *
except ImportError:
    print("❌ 找不到 SDK")
    sys.exit()

def main():
    print(f"🔌 连接机械臂 {ROBOT_IP} ...")
    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = robot.rm_create_robot_arm(ROBOT_IP, 8080)
    
    if handle.id <= 0:
        print("❌ 连接失败，请检查网线")
        return

    # 1. 检查当前电压状态
    print("\n🔍 检查末端电源状态...")
    ret, voltage = robot.rm_get_tool_voltage()
    if ret == 0:
        print(f"   当前档位: {voltage} (0=0V, 1=5V, 2=12V, 3=24V)")
        if voltage == 3:
            print("   ✅ 电源似乎已经是开着的。")
        else:
            print("   ⚠️ 电源未开启 (0V) 或电压不对！")
    
    # 2. 强制开启 24V 电源
    print("\n⚡️ 正在强制开启 24V 电源...")
    # 参数: (3, True) -> 3代表24V, True代表阻塞等待
    ret = robot.rm_set_tool_voltage(3, True)
    
    if ret == 0:
        print("✅ 电源开启指令发送成功！")
    else:
        print(f"❌ 电源开启失败 (Code: {ret})")

    # 3. 等待夹爪启动 (关键步骤！)
    # 夹爪上电后需要几秒钟自检，这时候发指令也没用
    print("⏳ 等待 5 秒让夹爪启动...")
    for i in range(5, 0, -1):
        print(f"   {i}...", end="\r")
        time.sleep(1)
    print("\n")

    # 4. 动一下试试
    print("🧪 测试动作：闭合 (位置10)...")
    # 使用你数据采集时的那个函数逻辑
    ret = robot.rm_set_gripper_position(10, True, 5) # 稍微给长一点超时时间
    if ret == 0:
        print("🎉 成功！夹爪动了！(闭合)")
    else:
        print(f"❌ 依然不动 (Code: {ret})")
        print("👉 请肉眼检查：夹爪侧面的【蓝灯/绿灯】亮了吗？")

    print("🧪 测试动作：张开 (位置1000)...")
    robot.rm_set_gripper_position(1000, True, 5)
    
    robot.rm_delete_robot_arm()

if __name__ == "__main__":
    main()