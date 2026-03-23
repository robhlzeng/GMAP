# GMAP 论文复现设计文档

**论文**: GMAP: Generalized Manipulation of Articulated Objects in Robotic Using Pre-trained Model (AAAI-25)

**日期**: 2026-03-23

**目标**: 严格复现论文完整 pipeline，包含预训练、三个下游网络、轨迹规划和 SAPIEN 仿真

---

## 1. 项目结构

```
gmap/
├── configs/                    # 配置文件 (YAML)
│   ├── pretrain.yaml          # VQ-VAE预训练配置
│   ├── segnet.yaml            # Seg-Net配置
│   ├── paranet.yaml           # Para-Net配置
│   ├── affordnet.yaml         # Afford-Net配置
│   └── simulation.yaml        # SAPIEN仿真配置
├── data/                       # 数据处理
│   ├── download.py            # 数据集下载脚本
│   ├── shapenet_dataset.py    # ShapeNet数据集（预训练）
│   ├── partnet_dataset.py     # PartNet-Mobility数据集
│   ├── shape2motion_dataset.py # Shape2Motion数据集
│   └── transforms.py          # 点云数据增强
├── models/                     # 模型定义
│   ├── msfe.py                # 多尺度特征提取器(MSFE) + dVAE
│   ├── pfe.py                 # 点级特征传播(PFE)
│   ├── segnet.py              # 部件分割+可动性预测网络
│   ├── paranet.py             # 关节参数估计网络
│   ├── affordnet.py           # 可操作点预测网络
│   ├── pointnet2_utils.py     # PointNet++工具(FPS, KNN等)
│   └── transformer.py         # ViT/Transformer模块
├── train/                      # 训练脚本
│   ├── train_pretrain.py      # VQ-VAE预训练
│   ├── train_segnet.py        # Seg-Net训练
│   ├── train_paranet.py       # Para-Net训练
│   └── train_affordnet.py     # Afford-Net训练
├── eval/                       # 评估脚本
│   ├── eval_segnet.py         # Seg-Net评估
│   ├── eval_paranet.py        # Para-Net评估
│   ├── eval_affordnet.py      # Afford-Net评估
│   └── metrics.py             # 评估指标计算
├── planner/                    # 轨迹规划
│   ├── trajectory.py          # 轨迹生成（旋转/平移）
│   └── motion_planning.py     # 运动规划算法
├── simulation/                 # SAPIEN仿真环境
│   ├── env.py                 # 仿真环境封装
│   ├── robot.py               # Panda机器人控制
│   └── evaluate_sim.py        # 仿真评估
├── utils/                      # 工具函数
│   ├── logger.py              # 日志工具
│   ├── checkpoint.py          # 模型保存/加载
│   └── pc_utils.py            # 点云处理工具
├── paper/                      # 论文PDF
├── docs/plans/                 # 设计文档
├── requirements.txt
├── setup.py
└── README.md
```

## 2. 模型架构

### 2.1 MSFE (多尺度特征提取器)

输入点云 P ∈ R^(N×3), N=8192。

**多尺度分组**:
- Scale 1: FPS(512点) + KNN(K=32) → 512个patch, 每个32点
- Scale 2: FPS(256点) + KNN(K=8) → 256个patch, 每个8点
- Scale 3: FPS(64点) + KNN(K=8) → 64个patch, 每个8点

每个Scale: Mini-PointNet → patch embedding → ViT Encoder (6层, 384维, 6头)

### 2.2 VQ-VAE 预训练 (Point-MGE 风格)

1. **dVAE Tokenizer**: 将点云patch离散化为token (codebook大小8192)
2. 随机遮蔽 60% 的 patch
3. **目标**: 重建被遮蔽patch的离散token
4. **损失**: 交叉熵 (token预测) + 重建损失

**预训练超参数**:
- 300 epochs, AdamW, lr=1e-3, weight_decay=0.05
- Cosine lr schedule with warmup
- 数据: ShapeNet 55类

### 2.3 PFE (点级特征传播)

类似 PointNet++ 的 Feature Propagation：
- 三个尺度特征通过反距离加权插值逐层上采样
- 最终得到逐点特征 N×C

### 2.4 Seg-Net (部件分割 + 可动性预测)

- 输入: PFE特征 (N×C)
- 部件分割头: MLP → N×K (K个部件类别)
- 可动性预测头: MLP → N×1 (二分类：可动/不可动)
- 损失: L_seg = CE_seg + CE_mov
- 微调: 100 epochs, lr=5e-4

### 2.5 Para-Net (关节参数估计)

- 输入: PFE特征 + Seg-Net分割结果(按part聚合)
- 关节类型: MLP → {revolute, prismatic} (二分类)
- 关节轴方向: MLP → R^3 (单位向量, 余弦相似度损失)
- 关节位置: MLP → R^3 (仅旋转关节需要, L2损失)
- 关节状态: MLP → R^1 (L1损失)

### 2.6 Afford-Net (可操作性预测)

**Stage 1 - Action Proposal**:
- PFE特征 → MLP → N×1 逐点可操作性评分
- Top-K采样候选操作点

**Stage 2 - Action Scoring**:
- 候选点特征 + 方向编码 → MLP → 评分
- 选择最优操作点和方向

## 3. 数据集

| 数据集 | 用途 | 来源 | 处理 |
|--------|------|------|------|
| ShapeNet (55类) | MSFE预训练 | 官方/Point-BERT处理版 | 采样8192点, 归一化 |
| PartNet-Mobility | Seg-Net, Para-Net训练 | SAPIEN官网 | 按论文9:1:1划分 |
| Shape2Motion | 额外评估 | 官方下载 | 统一格式处理 |

## 4. 训练流程

```
阶段1: VQ-VAE预训练 (ShapeNet, 300 epochs)
  → 产出: MSFE预训练权重

阶段2: 下游任务微调 (PartNet-Mobility, 各100 epochs)
  ├─ Seg-Net: 加载MSFE权重
  ├─ Para-Net: 加载MSFE权重, 使用Seg-Net预测
  └─ Afford-Net: 加载MSFE权重

阶段3: SAPIEN仿真评估
  → 加载所有训练好的模型
  → 在7个物体类别上评估操控成功率
```

## 5. 评估指标

- **Seg-Net**: mIoU (部件分割), Accuracy (可动性预测)
- **Para-Net**: 关节类型准确率, 方向误差(角度°), 位置误差(cm), 状态误差
- **Afford-Net**: 操作成功率
- **仿真**: 7个类别(Laptop, Box, Drawer, Door, Faucet, Kettle, Switch)操控成功率

## 6. 轨迹规划

根据 Para-Net 预测的关节参数：
- **旋转关节**: 绕关节轴旋转，计算末端执行器弧形轨迹
- **平移关节**: 沿关节轴平移，计算直线轨迹
- 使用动态步长控制，每步重新感知和调整

## 7. 技术栈

```
pytorch >= 1.12
timm                    # ViT实现
pointnet2_ops           # PointNet++ CUDA算子
sapien >= 2.0           # SAPIEN仿真器
open3d                  # 点云可视化
h5py                    # 数据读取
einops                  # Tensor操作
tensorboard             # 训练日志
pyyaml                  # 配置文件
```

## 8. 设计原则

- 模块化：每个网络独立可训练、可测试
- 配置驱动：所有超参数通过YAML配置文件管理
- 权重共享：预训练MSFE权重被所有下游网络复用
- 仿真解耦：仿真环境与网络通过planner桥接
- 单卡优先：所有代码先确保单GPU可运行
