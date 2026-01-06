import cv2
import time
import os

def scan_cameras():
    print("==================================================")
    print("📸 正在拍照存档 (Index 0 - 14)...")
    print("==================================================")
    
    # 获取当前脚本所在目录
    save_dir = os.getcwd()
    print(f"📂 照片将保存到: {save_dir}")

    # 根据你的日志，只扫这几个成功的
    indices_to_check = [0, 2, 6, 8, 12, 14] 

    for index in indices_to_check:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # 读几帧让白平衡稳定一下
            for _ in range(10):
                cap.read()
            
            ret, frame = cap.read()
            if ret:
                filename = f"camera_index_{index}.jpg"
                
                # 在图片上写个大大的数字
                cv2.putText(frame, f"Index: {index}", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
                
                cv2.imwrite(filename, frame)
                print(f"✅ 已保存: {filename}")
            else:
                print(f"❌ Index {index} 打开了但读不到画面")
            cap.release()
        else:
            print(f"⚠️ Index {index} 无法打开")

    print("\n🎉 完成！请现在打开文件管理器，查看生成的 .jpg 图片。")

if __name__ == "__main__":
    scan_cameras()