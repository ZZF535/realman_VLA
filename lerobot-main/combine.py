import os
import shutil
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets

# ================= 配置区域 =================
# 1. 你的源数据父目录 (包含那50个文件夹的地方)
SOURCE_ROOT = Path("/home/robot/lerobot-main/src/lerobot/datasets/data_896")

# 2. 合并后的输出目录 (我们希望它存在 dataset 目录下)
OUTPUT_BASE_DIR = Path("/home/robot/lerobot-main/src/lerobot/datasets")

# 3. 新数据集的名字
NEW_DATASET_NAME = "put_clothes_aggregated_v3"
# ===========================================

def main():
    if not SOURCE_ROOT.exists():
        print(f"❌ 错误: 找不到源目录 {SOURCE_ROOT}")
        return

    # 扫描所有有效的 v3.0 子数据集
    print(f"🔍 正在扫描 {SOURCE_ROOT} ...")
    dataset_dirs = [
        d for d in SOURCE_ROOT.iterdir() 
        if d.is_dir() 
        and (d / "meta/info.json").exists()
        and not d.name.endswith("_old") # 排除旧备份
    ]
    
    repo_ids = [d.name for d in dataset_dirs]
    
    if not repo_ids:
        print("❌ 未找到任何有效的数据集。请检查路径或是否已完成转换。")
        return
        
    print(f"✅ 找到了 {len(repo_ids)} 个数据集，准备合并...")

    # [关键修复 1] 构建精确的源路径列表
    # 之前错误地只传了父目录，现在我们把每个子文件夹的完整路径传进去
    source_roots = [d for d in dataset_dirs]

    # [关键修复 2] 构建精确的目标路径
    # LeRobot 甚至需要我们手动指定合并后的完整文件夹路径
    aggr_path = OUTPUT_BASE_DIR / NEW_DATASET_NAME
    
    if aggr_path.exists():
        print(f"⚠️ 警告: 目标文件夹 {aggr_path} 已存在。正在删除以重新合并...")
        shutil.rmtree(aggr_path)

    print(f"📦 正在合并到: {aggr_path}")

    try:
        aggregate_datasets(
            repo_ids=repo_ids,
            aggr_repo_id=NEW_DATASET_NAME,
            roots=source_roots,  # 修复：传入具体的路径列表
            aggr_root=aggr_path, # 修复：传入完整的目标路径
        )
        print(f"\n🎉🎉🎉 合并成功！")
        print(f"📂 新数据集位置: {aggr_path}")
        print("-" * 30)
        print("🚀 训练命令参考:")
        print(f"dataset.repo_id={NEW_DATASET_NAME}")
        print(f"dataset.root={OUTPUT_BASE_DIR}")
        
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()