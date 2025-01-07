import json
import matplotlib.pyplot as plt
import os

# 设置Matplotlib使用SimHei字体（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 定义四个模型的文件路径
accuracy_files = {
    'ResNet-18 + ViT并行': 'saved_models\Fusion_CIFAR100_metrics.json',
    'ResNet-18 + ViT并行PRO': 'saved_models\Fusion_CIFAR100_SGD_accuracies.json'
}

# 检查所有文件是否存在
for model_name, file_path in accuracy_files.items():
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please ensure the model has been trained and the file exists.")
        exit(1)

# 颜色和样式的选择（可根据需要调整）
colors = {
    'ResNet-18 + ViT并行': 'blue',
    'ResNet-18 + ViT并行PRO': 'red',
}

linestyles = {
    'ResNet-18 + ViT并行': '-',
    'ResNet-18 + ViT并行PRO': ':'
}

markers = {
    'ResNet-18 + ViT并行': 'o',
    'ResNet-18 + ViT并行PRO': 'd'
}

plt.figure(figsize=(14, 8))

for model_name, file_path in accuracy_files.items():
    # 读取准确率数据
    with open(file_path, 'r') as f:
        data = json.load(f)
        epochs = data['epochs']
        val_accuracies = data['val_accuracies']
    
    # 绘制验证准确率曲线
    plt.plot(epochs, val_accuracies, label=model_name,
             color=colors.get(model_name, 'black'),
             linestyle=linestyles.get(model_name, '-'),
             marker=markers.get(model_name, ''))

# 添加标题和标签
plt.title('消融实验在CIFAR-100上的验证集准确率对比', fontsize=18)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('验证集准确率', fontsize=16)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(fontsize=14)

# 调整x轴的刻度
max_epoch = max([len(json.load(open(f))['epochs']) for f in accuracy_files.values()])
plt.xticks(range(1, max_epoch + 1, 2))  # 每隔2个epoch显示一个刻度

# 保存图像
plt.tight_layout()
plt.savefig('saved_models/accuracy_comparison_all_models.png', dpi=300)
print("Accuracy comparison plot saved to 'saved_models/accuracy_comparison_all_models.png'")

# 显示图像
plt.show()
