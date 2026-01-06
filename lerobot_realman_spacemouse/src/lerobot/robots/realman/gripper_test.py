#!/usr/bin/env python3
import serial
import time

class GripperController:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        
    def connect(self):
        """连接串口设备"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.5
            )
            print(f"成功连接到串口设备: {self.port}")
            time.sleep(2)  # 给设备初始化时间
            return True
        except serial.SerialException as e:
            print(f"连接失败: {e}")
            return False
        except Exception as e:
            print(f"发生错误: {e}")
            return False

    def disconnect(self):
        """断开串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口连接已关闭")
            
    def calculate_checksum(self, data):
        """计算BCC校验位 (异或校验)"""
        checksum = 0
        for byte in data:
            checksum ^= byte
        return checksum

    def create_command(self, direction, angle_deg, speed_radps):
        """
        创建控制命令帧
        
        参数:
        - direction: 0 开爪(逆时针), 1 闭爪(顺时针)
        - angle_deg: 目标角度 (度)
        - speed_radps: 旋转速度 (弧度/秒)
        """
        # 基本帧参数 (根据协议)
        frame = [
            0x7B,  # 帧头
            0x01,  # 设备地址ID
            0x02,  # 控制模式 (0x02=位置控制)
            direction,  # 转向  0闭合，1张开
            0x20   # 细分值 (0x20=32细分)
        ]
        
        # 角度处理 (放大10倍，转换为16位整数)
        angle_scaled = int(angle_deg * 10)
        # 分解为高低字节
        frame.append((angle_scaled >> 8) & 0xFF)  # POS_H
        frame.append(angle_scaled & 0xFF)         # POS_L
        
        # 速度处理 (放大10倍，转换为16位整数)
        speed_scaled = int(speed_radps * 10)
        frame.append((speed_scaled >> 8) & 0xFF)  # SPEED_H
        frame.append(speed_scaled & 0xFF)         # SPEED_L
        
        # 计算校验位
        checksum = self.calculate_checksum(frame)
        frame.append(checksum)
        
        # 添加帧尾
        frame.append(0x7D)
        
        return bytes(frame)

    def send_command(self, command):
        """发送命令到串口设备"""
        try:
            if self.ser and self.ser.is_open:
                print(f"发送命令: {[f'0x{x:02X}' for x in command]}")
                self.ser.write(command)
                
                # 读取并打印设备响应 (如果有)
                time.sleep(0.1)
                response = self.ser.read_all()
                if response:
                    print(f"设备响应: {response.hex()}")
                return True
            else:
                print("串口未连接!")
                return False
        except serial.SerialException as e:
            print(f"串口错误: {e}")
            return False
        except Exception as e:
            print(f"发送命令错误: {e}")
            return False

    def close_gripper(self, close_angle=1572.0, speed=20.0):
        """闭爪命令"""
        print("执行闭爪操作...")
        # 方向1:顺时针闭爪，1872度位置 (对应5.2圈)
        command = self.create_command(direction=1, angle_deg=close_angle, speed_radps=speed)
        return self.send_command(command)

    def open_gripper(self, open_angle=1000.0, speed=20.0):
        """开爪命令"""
        print("执行开爪操作...")
        # 方向0:逆时针开爪，0度位置
        command = self.create_command(direction=0, angle_deg=open_angle, speed_radps=speed)
        return self.send_command(command)


if __name__ == "__main__":
    gripper = GripperController()
    
    if gripper.connect():
        try:
            # 示例操作
            time.sleep(1)
            
            # 闭爪 (参数可选)
            gripper.close_gripper()
            time.sleep(3)  # 等待动作完成
            
            # 开爪 (参数可选)
            # gripper.open_gripper()
            # time.sleep(3)
            
        finally:
            gripper.disconnect()
    else:
        print("无法建立串口连接，请检查设备")