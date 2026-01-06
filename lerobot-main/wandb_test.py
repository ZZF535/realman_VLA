# 创建测试脚本：test_wandb.py
import random

import wandb

# 初始化wandb项目
run = wandb.init(
    # 设置wandb实体（通常为你的团队名称），用于记录你的项目。

    # 设置wandb项目名称，用于记录本次实验。
    project="lerobot-experiments",
    # 跟踪超参数和运行元数据。
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# 模仿训练过程
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# 完成实验
run.finish()
print("Wandb test completed successfully!")