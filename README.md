# CNN-ViT混合模型实验
本项目实现了CNN(ResNet-18)与Vision Transformer的多种组合模型,在CIFAR-10和CIFAR-100数据集上进行了实验对比。
## 文件结构
.
├── CNNVITP.py          // CNN-ViT并行融合基础版
├── CNNVITP_PRO.py      // CNN-ViT并行融合改进版
├── CNNVITS.py          // CNN-ViT串行融合
├── RESNETONLY.py       // 仅CNN基线模型  
├── VITONLY.py          // 仅ViT基线模型
├── draw.py             // 结果可视化
└── saved_models/       // 保存模型和结果



# 实验环境
Python 3.8+
PyTorch
torchvision
timm
matplotlib
tqdm
建议显卡使用3090或以上,以获得更好的训练速度和效果。

# 数据集下载
实验使用CIFAR-10和CIFAR-100数据集,代码会自动下载到./data目录。
CIFAR-10: 60000张32x32彩色图片,10个类别,每类6000张
https://www.kaggle.com/datasets/petitbonney/cifar10-image-recognition
CIFAR-100: 60000张32x32彩色图片,100个类别,每类600张
https://www.kaggle.com/datasets/melikechan/cifar100
模型结构
项目包含以下几种模型实现:
CNN Only - 仅使用ResNet-18 (RESNETONLY.py)
ViT Only - 仅使用Vision Transformer (VITONLY.py)
CNN-ViT并行融合 (CNNVITP.py)
CNN-ViT串行融合 (CNNVITS.py)
改进的CNN-ViT并行融合 (CNNVITP_PRO.py)

# 运行方式
训练单个模型,例如:
比较不同模型结果:
## 模型保存
训练过程中的最佳模型和指标会保存在saved_models目录下:
模型权重: best_model_*.pth
训练曲线: *_Loss.png, *_Accuracy.png
训练指标: *_accuracies.json
## 实验改进
改进版本(CNNVITP_PRO.py)主要包含以下优化:
添加Dropout层(p=0.3)减少过拟合
使用SGD优化器替代Adam
添加权重衰减(weight decay)
优化学习率策略
# 实验结果
实验结果可以通过运行draw.py生成对比图表,保存在saved_models/accuracy_comparison_all_models.png。
图表将展示:
不同模型在验证集上的准确率对比
训练过程中的loss变化
准确率随epochs的变化趋势
## 表3-1 对比实验信息
| Model       | Accuracy | Loss   |
|-------------|----------|--------|
| ResNet-18   | 92.81%   | 0.3622 |
| ViT         | 71.16%   | 0.8253 |
| RVSerial    | 89.11%   | 0.3167 |
| RVParallel  | 93.20%   | 0.3514 |

## 表3-2 消融实验信息
| Model       | Accuracy | Loss   |
|-------------|----------|--------|
| **Cifar-10 DataSet** |          |        |
| RVP_SGD     | 93.17%   | 0.2545 |
| RVParallel  | 93.20%   | 0.3514 |
| **Cifar-100 DataSet** |          |        |
| RVP_SGD     | 73.68%   | 1.2145 |
| RVParallel  | 71.37%   | 1.5021 |