import shutil
from pathlib import Path
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

# 兼容导入
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ================= 🔧 配置区域 (保持不变) =================

RAW_DATA_ROOT = Path("/home/robot/data_896_test") 
TARGET_DATASET_DIR = Path("converted_dataset_final")
REPO_ID = "local/washing_machine_task" 
FPS = 30           
ROBOT_TYPE = "bi_realman" 

# 相机映射
CAMERA_MAPPING = {
    "image_left": "observation.images.image_left_wrist",
    "image_right": "observation.images.image_right_wrist",
    "image_top": "observation.images.image_top"
}

# 任务描述
TASK_DESCRIPTION = "put clothes into washing machine"

# ==========================================================

def main():
    # 如果目标数据集目录已存在，清理旧目录
    if TARGET_DATASET_DIR.exists():
        print(f"⚠️  正在清理旧目录 {TARGET_DATASET_DIR} ...")
        shutil.rmtree(TARGET_DATASET_DIR)

    # 扫描所有 episode 文件夹
    episode_folders = sorted([p for p in RAW_DATA_ROOT.glob("put_clothes_*") if p.is_dir()])
    print(f"👀 扫描到 {len(episode_folders)} 个 Episode")

    if not episode_folders:
        print(f"❌ 错误: 没找到文件夹")
        return

    # --- 1. 自动探测维度 ---
    first_parquet = episode_folders[0] / "data" / "chunk-000" / "episode_000000.parquet"
    if not first_parquet.exists():
        print(f"❌ 错误: 找不到 {first_parquet}")
        return

    print(f"📖 读取示例以检测维度...")
    df_sample = pd.read_parquet(first_parquet)
    
    # 自动检测 Action 维度
    if 'action' in df_sample.columns:
        action_dim = len(df_sample['action'].iloc[0])
    else:
        action_cols = [c for c in df_sample.columns if "action" in c]
        action_dim = len(action_cols)

    # 自动检测 State 维度
    if 'observation.state' in df_sample.columns:
        state_dim = len(df_sample['observation.state'].iloc[0])
    else:
        state_cols = [c for c in df_sample.columns if "state" in c or "qpos" in c]
        state_dim = len(state_cols)

    print(f"✅ 维度确认: Action={action_dim}, State={state_dim}")

    # --- 2. 定义 Features ---
    features = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"motor_{i}" for i in range(action_dim)]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"motor_{i}" for i in range(state_dim)]
        },
        # ⚠️ [修复点1] 你的版本要求这里的 key 必须叫 'task'，而不是 'task_index'
        "task": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["index"]
        }
    }
    
    # 为每个相机添加图像特征
    for cam_key in CAMERA_MAPPING.keys():
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": (480, 640, 3), 
            "names": ["height", "width", "channel"]
        }

    # --- 3. 初始化数据集 ---
    print("🚀 初始化 LeRobot 数据集...")
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=TARGET_DATASET_DIR,
        robot_type=ROBOT_TYPE,
        features=features,
        use_videos=True,
    )

    # --- 4. 转换循环 ---
    for ep_folder in tqdm(episode_folders, desc="Converting"):
        try:
            # 读取 Parquet 文件
            parquet_path = ep_folder / "data" / "chunk-000" / "episode_000000.parquet"
            df = pd.read_parquet(parquet_path)
            num_frames = len(df)
            
            for i in range(num_frames):
                # 提取 Action
                if 'action' in df.columns:
                    action = torch.tensor(df.iloc[i]['action'])
                else:
                    act_cols = sorted([c for c in df.columns if "action" in c])
                    action = torch.tensor(df.iloc[i][act_cols].values, dtype=torch.float32)

                # 提取 State
                if 'observation.state' in df.columns:
                    state = torch.tensor(df.iloc[i]['observation.state'])
                else:
                    state_cols = sorted([c for c in df.columns if "state" in c or "qpos" in c])
                    if state_cols:
                        state = torch.tensor(df.iloc[i][state_cols].values, dtype=torch.float32)
                    else:
                        state = torch.zeros(state_dim)

                # 准备帧数据
                frame_data = {
                    "action": action,
                    "observation.state": state,
                    "task": torch.tensor(0, dtype=torch.int64)  # 为每帧添加 task 特征
                }

                # 读取图像并添加到帧数据
                for cam_key, folder_name in CAMERA_MAPPING.items():
                    img_dir = ep_folder / "images" / folder_name / "episode_000000"
                    
                    # 优先找 frame_000000.jpg
                    img_path = img_dir / f"frame_{i:06d}.jpg"
                    if not img_path.exists():
                        img_path = img_dir / f"{i}.jpg"
                    
                    if not img_path.exists():
                        raise FileNotFoundError(f"Missing image: {img_path}")

                    frame_data[f"observation.images.{cam_key}"] = Image.open(img_path)

                dataset.add_frame(frame_data)

            # 保存 Episode
            dataset.save_episode(task=TASK_DESCRIPTION)

        except Exception as e:
            print(f"\n❌ Error in {ep_folder.name}: {e}")
            dataset.clear_episode_buffer()
            continue

    # --- 5. 收尾 ---
    print("\n📦 Finalizing Dataset...")
    
    # 尝试调用 consolidate，如果版本不支持则跳过
    try:
        dataset.consolidate()
        print("✅ Metadata consolidated successfully.")
    except AttributeError:
        print("⚠️  Skip: 'consolidate' method not found (OK for older versions).")
        print("    数据已保存完成。如果需要统计信息，请尝试运行 lerobot 自带的 compute_stats 脚本。")
    except Exception as e:
        print(f"⚠️  Consolidation warning: {e}")
    
    print("="*50)
    print(f"🎉 转换完成！")
    print(f"📂 新数据集位置: {TARGET_DATASET_DIR}/{REPO_ID}")
    print("="*50)

if __name__ == "__main__":
    main()
