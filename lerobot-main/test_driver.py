import sys
import time
import os
import torch
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

# --- 1. 路径修正 ---
current_dir = os.getcwd()
src_path = os.path.join(current_dir, "src")
if os.path.exists(src_path):
    sys.path.append(src_path)

# --- 2. 导入 Robot 类 ---
try:
    from lerobot.robots.realman_dual_arm import RealManDualArmRobot
    print(f"✅ 成功导入 Robot 类: {RealManDualArmRobot.name}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
            observation = robot.get_observation()
            # --------------------

            state = observation["observation.sta
            observation = robot.get_observation()
            # --------------------

            state = observation["observation.state"] 
            
            # 简单校验te"] 
            
            # 简单校验
    # 为了调试路径问题，打印一下sys.path
    # print(sys.path)
    exit(1)

# --- 3. [关键修改] 模拟配置类 (MockRobotConfig) ---
@dataclass
class MockRobotConfig:
    # === LeRobot 基类必须字段 ===
    type: str = "realman_dual_arm"
    id: str = "realman_test_bot"
    
    # [新增] 修复 calibration_dir 报错
    # 设为 None，让 Robot 类自动使用默认路径
    calibration_dir: str | None = None  
    
    # === 你的自定义字段 ===
    # 请务必确认这里的 IP 和你机械臂真实 IP 一致
    left_arm_ip: str = "169.254.128.18"   
    right_arm_ip: str = "169.254.128.19"
    
    # === 相机配置 ===
    cameras: dict = field(default_factory=lambda: {}) 

def test_robot_read():
    print("========================================")
    print("🚀 开始 RealMan 双臂接入冒烟测试")
    print("========================================")

    # 1. 初始化
    print("\n[1/4] 初始化 Robot 实例...")
    config = MockRobotConfig()
    
    try:
        robot = RealManDualArmRobot(config) 
        print("✅ 实例创建成功")
    except Exception as e:
        print(f"❌ 实例创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 连接
    print("\n[2/4] 尝试连接机械臂 (Connect)...")
    try:
        robot.connect()
        # 注意：如果 IP 填错或者网线没插，这里会卡住或者报错
        print("✅ 连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("💡 提示: 请检查网线连接、IP地址设置以及是否关闭了防火墙。")
        return

    # 3. 循环读取测试
    print("\n[3/4] 开始读取数据 loop (测试 100 帧)...")
    print("⚠️  请尝试【手动】轻轻移动机械臂或按压夹爪，观察数值是否变化")
    
    cnt = 0
    max_steps = 100
    start_time = time.time()
    
    try:
        while cnt < max_steps:
            loop_start = time.time()
            
            # --- 调用核心函数 ---
            observation = robot.get_observation()
            # --------------------

            state = observation["observation.state"] 
            
            # 简单校验
            if state.shape != (14,):
                print(f"❌ 维度错误! 期望 (14,), 实际 {state.shape}")
                break

            # 解析数据用于显示
            state_np = state.numpy() if isinstance(state, torch.Tensor) else state.cpu().numpy()
            
            # 数据解包: [左臂6, 左爪1, 右臂6, 右爪1]
            l_joints = np.degrees(state_np[:6])            if state.shape != (14,):

            l_grip = state_np[6]
            r_joints = np.degrees(state_np[7:13])
            r_grip = state_np[13]

            # 打印日志
            if cnt % 10 == 0:
                
                print(f"Frame {cnt:03d} | L: {l_joints} | R: {r_joints}")
            
            cnt += 1
            
            # 频率控制 (模拟 30Hz)
            elapsed = time.time() - loop_start
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)

    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 4. 断开连接
        total_time = time.time() - start_time
        avg_freq = cnt / total_time if total_time > 0 else 0
        
        print(f"\n[4/4] 测试结束。平均频率: {avg_freq:.2f} Hz")
        print("正在断开连接...")
        try:
            robot.disconnect()
            print("✅ 已断开连接")
        except:
            pass

if __name__ == "__main__":
    test_robot_read()