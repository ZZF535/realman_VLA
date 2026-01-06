import cv2
import glob
import os
import time

# 从你的 ls -l 结果里提取出来的三个序列号
SERIALS = [
    # "218423024458",
    "218423026633",
    "219423020362"
]

def get_video_id(serial):
    # RealSense 通常有很多个节点 (index0, index1...)
    # index0 通常是 RGB 图像，最适合 OpenCV 读取
    pattern = f"/dev/v4l/by-id/*{serial}*index0"
    paths = glob.glob(pattern)
    
    if not paths:
        return None
        
    # 解析软链接，比如 -> ../../video8
    real_path = os.path.realpath(paths[0])
    dev_name = os.path.basename(real_path) # "video8"
    try:
        return int(dev_name.replace("video", ""))
    except:
        return None

def main():
    print("🕵️‍♂️ 开始相机身份鉴定...")
    print("按 'q' 键切换到下一个相机，或者退出。")

    for serial in SERIALS:
        vid_idx = get_video_id(serial)
        if vid_idx is None:
            print(f"❌ 序列号 {serial} 未找到 (可能没插好?)")
            continue
            
        print(f"\n📸 正在打开序列号: {serial} (设备ID: {vid_idx})")
        print("👉 请用手遮挡各个相机，看看是哪一个变黑了！")
        
        cap = cv2.VideoCapture(vid_idx)
        if not cap.isOpened():
            print(f"⚠️ 无法打开设备 {vid_idx}")
            continue
            
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取图像")
                break
                
            # 在画面上用显眼的绿字写上序列号
            text = f"Serial: {serial}"
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)
            
            cv2.imshow(f"Identify: {serial}", frame)
            
            # 按 q 退出当前相机，看下一个
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ 序列号 {serial} 测试结束。")
        time.sleep(1)

    print("\n🎉 所有相机测试完毕！请去修改 run_realman_safe.py")

if __name__ == "__main__":
    main()