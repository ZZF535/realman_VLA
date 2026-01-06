import time
import torch
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path  # <--- [新增] 必须导入这个

# 导入你的机器人驱动
from lerobot.robots.realman_dual_arm.realman_dual_arm import RealManDualArmRobot

@dataclass
class MockConfig:
    # LeRobot 基类必须字段
    type: str = "realman_dual_arm"
    id: str = "realman_test_bot"
    
    # 你的自定义字段 (IP请确认正确)
    left_arm_ip: str = "169.254.128.18"
    right_arm_ip: str = "169.254.128.19"
    
    # 其他必须字段
    cameras: dict = field(default_factory=dict)
    
    # === [关键修复] ===
    # 必须是 Path 对象，不能是 str！
    calibration_dir: Path = Path(".cache/calibration") 

def test_gripper_motion():
    print("🚀 开始 send_action 动作测试...")
    
    # 1. 初始化
    print("-> 初始化 Robot...")
    config = MockConfig()
    try:
        robot = RealManDualArmRobot(config)
        robot.connect()
        print("✅ 连接成功，准备测试...")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        # 2. 读取当前状态作为基准
        obs = robot.get_observation()
        current_state = obs["observation.state"] # Tensor[14]
        print(f"-> 当前状态读取成功。")
        
        # 3. 构造动作
        # 动作 A: 夹爪完全张开 (值为 0.0)
        action_open = current_state.clone()
        action_open[6] = 1   # 左爪张开
        action_open[13] = 1  # 右爪张开
        
        # 动作 B: 夹爪闭合一半 (值为 0.5)
        # 注意：不要给 1.0，防止没拿东西空捏导致电机过热
        action_close = current_state.clone()
        action_close[6] = 0.5
        action_close[13] = 0.5

        # 4. 执行测试循环
        print("\n👉 测试 1: 张开夹爪 (持续 2 秒)...")
        for _ in range(60): # 30Hz * 2s
            robot.send_action(action_open)
            time.sleep(0.033)
            
        print("👉 测试 2: 闭合夹爪 (持续 2 秒)...")
        for _ in range(60):
            robot.send_action(action_close)
            time.sleep(0.033)
            
        print("👉 测试 3: 再次张开 (复位)...")
        for _ in range(30):
            robot.send_action(action_open)
            time.sleep(0.033)

        print("\n✅ 测试完成！请确认夹爪是否动作。")

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 安全断开
        try:
            robot.disconnect()
            print("已断开连接")
        except:
            pass

if __name__ == "__main__":
    test_gripper_motion()