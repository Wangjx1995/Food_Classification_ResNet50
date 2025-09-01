# FoodClassification-ResNet50

> 使用 PyTorch 与 ResNet50
> 的食品图像分类（多类分类）项目。支持数据增强、可选层冻结（迁移学习）、混合精度训练、训练日志与结果导出。

## ✨ 特性

-   以 **ResNet50** 为主干的分类模型（可做迁移学习/微调）
-   训练/验证数据加载与 **数据增强**（`data_augmentation.py`）
-   **可冻结部分层**，快速收敛小数据集
-   **AMP 混合精度**（可选），更快更省显存
-   训练日志与 **可视化**（`logs/`），**结果导出**（`result_export.py`）
-   结构清晰，便于扩展（自定义数据集/类别/网络）

## 🗂️ 目录结构

    FoodClassification-ResNet50/
    ├─ data/                   # 数据集根目录（建议：train/、val/ 子目录）
    ├─ logs/                   # 训练日志与可视化输出
    ├─ net/                    # 模型定义/封装
    ├─ result/                 # 模型权重、预测结果等导出文件
    ├─ data_augmentation.py    # 数据增强策略
    ├─ data_loader.py          # 数据集与 DataLoader 定义
    ├─ train_val.py            # 训练/验证循环与度量
    ├─ result_export.py        # 结果导出脚本（如CSV/图表等）
    └─ main.py                 # 入口脚本：训练/验证/测试

## 📦 环境依赖

-   Python 3.8+
-   PyTorch \>= 1.12
-   torchvision
-   numpy, pandas
-   tqdm
-   pillow, opencv-python
-   matplotlib（可选）
-   tensorboard（可选）

安装：

``` bash
pip install torch torchvision torchaudio
pip install numpy pandas tqdm pillow opencv-python matplotlib tensorboard
```

## 📁 数据准备

    data/
    ├─ train/
    │  ├─ class_a/
    │  └─ ...
    └─ val/
       ├─ class_a/
       └─ ...

## 🚀 快速开始

训练：

``` bash
python main.py   --data_dir ./data   --train_dir train   --val_dir val   --num_classes <类别数>   --epochs 30   --batch_size 32   --lr 3e-4   --img_size 224   --freeze_upto 0   --use_amp   --workers 4   --output_dir ./result
```

验证：

``` bash
python main.py   --data_dir ./data   --val_dir val   --num_classes <类别数>   --eval   --weights ./result/best.ckpt
```

导出结果：

``` bash
python result_export.py --input ./result --out_csv ./result/metrics.csv
```

## 🧠 迁移学习与层冻结

-   支持从 ImageNet 预训练的 ResNet50 开始训练
-   使用 `--freeze_upto` 控制冻结深度

## 📊 监控与可视化

``` bash
tensorboard --logdir ./logs
```

## 📌 复现实验（示例基线）

-   AdamW, lr=3e-4, batch=32, epochs=30, CosineAnnealingLR, AMP

## 📄 许可证

MIT License
