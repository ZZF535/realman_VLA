import shutil
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


from pathlib import Path

RAW_DATA_ROOT = Path("/home/xyg/DoRobot/dataset/20251211/dev/把...放进..._打开洗衣机取出衣服_把衣服放到洗衣机里_896")
TARGET_DATASET_DIR = Path("/home/xyg/DoRobot/dataset/converted_dataset_v21")
REPO_ID = "local/washing_machine_task_896"

# print("RAW_DATA_ROOT", RAW_DATA_ROOT)
# print("exists", RAW_DATA_ROOT.exists())
# print("children", len(list(RAW_DATA_ROOT.iterdir())) if RAW_DATA_ROOT.exists() else -1)


FPS = 30
ROBOT_TYPE = "bi_realman"

TASK_DESCRIPTION = "put clothes into washing machine"

# 原始文件夹名 以及 你希望写入 v2.1 数据集的相机键
# 原始结构是 images/observation.images.image_left_wrist/episode_000000/frame_000000.jpg 这一类
CAMERAS = {
    "observation.images.image_left_wrist": "observation.images.image_left_wrist",
    "observation.images.image_right_wrist": "observation.images.image_right_wrist",
    "observation.images.image_top": "observation.images.image_top",
}

ACTION_NAMES_26 = [
    "left_arm_joint_1.pos","left_arm_joint_2.pos","left_arm_joint_3.pos",
    "left_arm_joint_4.pos","left_arm_joint_5.pos","left_arm_joint_6.pos",
    "left_arm_gripper.pos",
    "left_arm_pose_x.pos","left_arm_pose_y.pos","left_arm_pose_z.pos",
    "left_arm_pose_rx.pos","left_arm_pose_ry.pos","left_arm_pose_rz.pos",
    "right_arm_joint_1.pos","right_arm_joint_2.pos","right_arm_joint_3.pos",
    "right_arm_joint_4.pos","right_arm_joint_5.pos","right_arm_joint_6.pos",
    "right_arm_gripper.pos",
    "right_arm_pose_x.pos","right_arm_pose_y.pos","right_arm_pose_z.pos",
    "right_arm_pose_rx.pos","right_arm_pose_ry.pos","right_arm_pose_rz.pos",
]
STATE_NAMES_26 = ACTION_NAMES_26  # 如果 state 也是同顺序

def detect_action_state_dims(df_sample: pd.DataFrame):
    if "action" in df_sample.columns:
        action_dim = len(df_sample["action"].iloc[0])
        action_mode = "packed"
    else:
        act_cols = sorted([c for c in df_sample.columns if "action" in c])
        action_dim = len(act_cols)
        action_mode = "columns"

    if "observation.state" in df_sample.columns:
        state_dim = len(df_sample["observation.state"].iloc[0])
        state_mode = "packed"
    else:
        state_cols = sorted([c for c in df_sample.columns if ("state" in c) or ("qpos" in c)])
        state_dim = len(state_cols)
        state_mode = "columns"

    return action_dim, action_mode, state_dim, state_mode


def read_action(df: pd.DataFrame, i: int, mode: str):
    if mode == "packed":
        return torch.tensor(df.iloc[i]["action"], dtype=torch.float32)
    act_cols = sorted([c for c in df.columns if "action" in c])
    return torch.tensor(df.iloc[i][act_cols].values, dtype=torch.float32)


def read_state(df: pd.DataFrame, i: int, mode: str, state_dim: int):
    if mode == "packed":
        return torch.tensor(df.iloc[i]["observation.state"], dtype=torch.float32)
    state_cols = sorted([c for c in df.columns if ("state" in c) or ("qpos" in c)])
    if not state_cols:
        return torch.zeros(state_dim, dtype=torch.float32)
    return torch.tensor(df.iloc[i][state_cols].values, dtype=torch.float32)


def open_image_as_rgb(img_path: Path):
    # 防止句柄泄露
    with Image.open(img_path) as im:
        return im.convert("RGB")


def main():
    if TARGET_DATASET_DIR.exists():
        shutil.rmtree(TARGET_DATASET_DIR)

    print("RAW_DATA_ROOT", RAW_DATA_ROOT)
    if not RAW_DATA_ROOT.exists():
        raise RuntimeError(f"RAW_DATA_ROOT not exists {RAW_DATA_ROOT}")

    episode_folders = sorted([
        p for p in RAW_DATA_ROOT.iterdir()
        if p.is_dir() and (p / "data" / "chunk-000" / "episode_000000.parquet").exists()
    ])

    print("scan episodes", len(episode_folders))
    if not episode_folders:
        # 额外打印前 20 个子目录帮你定位到底在扫到什么
        children = [p.name for p in list(RAW_DATA_ROOT.iterdir())[:20]]
        print("example children", children)
        raise RuntimeError("no episode folders found")


    first_parquet = episode_folders[0] / "data" / "chunk-000" / "episode_000000.parquet"
    df_sample = pd.read_parquet(first_parquet)

    action_dim, action_mode, state_dim, state_mode = detect_action_state_dims(df_sample)
    print(f"detected action_dim {action_dim} state_dim {state_dim}")

    # 自动探测一张图的分辨率
    any_ep = episode_folders[0]
    any_cam_key, any_raw_dir = next(iter(CAMERAS.items()))
    test_img = any_ep / "images" / any_raw_dir / "episode_000000" / "frame_000000.jpg"
    if not test_img.exists():
        raise FileNotFoundError(f"cannot find sample image {test_img}")
    w, h = open_image_as_rgb(test_img).size

    # v2.1 features 不要写 task
    features = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ACTION_NAMES_26 if action_dim == 26 else [f"motor_{k}" for k in range(action_dim)],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": STATE_NAMES_26 if state_dim == 26 else [f"motor_{k}" for k in range(state_dim)],
        },
    }

    for target_key in CAMERAS.keys():
        features[target_key] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=TARGET_DATASET_DIR,
        robot_type=ROBOT_TYPE,
        features=features,
        use_videos=True,
    )

    for ep_folder in tqdm(episode_folders, desc="Converting"):
        parquet_path = ep_folder / "data" / "chunk-000" / "episode_000000.parquet"
        if not parquet_path.exists():
            print(f"skip missing parquet {ep_folder.name}")
            continue

        try:
            df = pd.read_parquet(parquet_path)
            n = len(df)
            if n <= 0:
                print(f"skip empty episode {ep_folder.name}")
                continue

            for i in range(n):
                frame = {
                    "action": read_action(df, i, action_mode),
                    "observation.state": read_state(df, i, state_mode, state_dim),
                }

                for target_key, raw_dir in CAMERAS.items():
                    img_dir = ep_folder / "images" / raw_dir / "episode_000000"
                    img_path = img_dir / f"frame_{i:06d}.jpg"
                    if not img_path.exists():
                        img_path = img_dir / f"{i}.jpg"
                    if not img_path.exists():
                        raise FileNotFoundError(f"missing image {img_path}")
                    frame[target_key] = open_image_as_rgb(img_path)

                # 兼容 2 种 add_frame 形式
                try:
                    dataset.add_frame(frame, task=TASK_DESCRIPTION)
                except TypeError:
                    frame["task"] = TASK_DESCRIPTION
                    dataset.add_frame(frame)

            # dataset.save_episode(task=TASK_DESCRIPTION)
            dataset.save_episode()

        except Exception as e:
            print(f"error episode {ep_folder.name} {e}")
            try:
                dataset.clear_episode_buffer()
            except Exception:
                pass
            continue

    try:
        dataset.consolidate()
    except Exception:
        pass

    print("done")
    print(f"output {TARGET_DATASET_DIR}")


if __name__ == "__main__":
    main()
