# CNN_ViT_Fusion_CIFAR100_with_Visualization.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import os
import timm  # 确保安装了timm库: pip install timm
import matplotlib.pyplot as plt  # 新增：用于绘图
import json  # 新增：用于保存JSON文件

# 1. 数据加载与预处理
# CIFAR-100 的常用均值和标准差
mean_cifar100 = [0.5071, 0.4867, 0.4408]
std_cifar100 = [0.2675, 0.2565, 0.2761]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_cifar100,
                         std=std_cifar100),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_cifar100,
                         std=std_cifar100),
])

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                             download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

# 2. 定义ResNet18适用于CIFAR-100
from torchvision.models.resnet import ResNet, BasicBlock

class ResNet18CIFAR(ResNet):
    def __init__(self, num_classes=100):
        super(ResNet18CIFAR, self).__init__(block=BasicBlock, layers=[2, 2, 2, 2])
        # 修改初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 移除最大池化层
        self.maxpool = nn.Identity()
        # 修改全连接层
        self.fc = nn.Linear(512, num_classes)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 3. 定义ViT适用于32x32图像
def get_vit_for_cifar():
    # 使用timm库创建ViT模型
    vit = timm.models.vision_transformer.VisionTransformer(
        img_size=32,
        patch_size=4,  # 32 / 4 = 8 patches per dimension, total 64 patches
        in_chans=3,
        num_classes=0,  # 移除分类头
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        representation_size=None,
    )
    # 初始化权重
    vit.apply(vit._init_weights)
    return vit

# 4. 定义融合模型
class ResNetViTFusion(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNetViTFusion, self).__init__()
        
        # ResNet部分
        self.cnn = ResNet18CIFAR(num_classes=num_classes)
        self.cnn._initialize_weights()
        
        # ViT部分
        self.vit = get_vit_for_cifar()
        
        # 融合后的全连接层
        self.fc = nn.Linear(512 + self.vit.embed_dim, num_classes)
        
    def forward(self, x):
        # CNN部分
        cnn_feat = self.cnn.conv1(x)  # [batch_size, 64, 32, 32]
        cnn_feat = self.cnn.bn1(cnn_feat)
        cnn_feat = self.cnn.relu(cnn_feat)
        # self.cnn.maxpool(x) 已经是Identity()
        cnn_feat = self.cnn.layer1(cnn_feat)  # [batch_size, 64, 32, 32]
        cnn_feat = self.cnn.layer2(cnn_feat)  # [batch_size, 128, 16, 16]
        cnn_feat = self.cnn.layer3(cnn_feat)  # [batch_size, 256, 8, 8]
        cnn_feat = self.cnn.layer4(cnn_feat)  # [batch_size, 512, 4, 4]
        cnn_feat = self.cnn.avgpool(cnn_feat)  # [batch_size, 512, 1, 1]
        cnn_feat = torch.flatten(cnn_feat, 1)  # [batch_size, 512]
        
        # ViT部分
        vit_feat = self.vit(x)  # [batch_size, 256]
        
        # 特征融合
        combined = torch.cat((cnn_feat, vit_feat), dim=1)  # [batch_size, 512 + 256]
        
        # 分类
        out = self.fc(combined)  # [batch_size, num_classes]
        return out

# 5. 实例化模型、定义损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetViTFusion(num_classes=100).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 6. 训练与验证函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 7. 训练模型
num_epochs = 40  # 根据需要调整
best_acc = 0.0

# 新增：初始化记录列表
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # 新增：记录每个epoch的损失和准确率
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    scheduler.step()
    
    # 保存最好的模型
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs('saved_models', exist_ok=True)
        torch.save(model.state_dict(), 'saved_models/best_model_fusion_CIFAR100.pth')
        print('Best model saved.')
    
    print('-' * 30)

# 8. 测试模型
model.load_state_dict(torch.load('saved_models/best_model_fusion_CIFAR100.pth'))
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# 新增：可视化损失和准确率
epochs_range = range(1, num_epochs + 1)

# 绘制损失曲线
plt.figure(figsize=(10,5))
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('saved_models/loss_curve_CIFAR100_Fusion.png')  # 保存损失曲线
plt.show()

# 绘制准确率曲线
plt.figure(figsize=(10,5))
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('saved_models/accuracy_curve_CIFAR100_Fusion.png')  # 保存准确率曲线
plt.show()

# 保存准确率和损失数据为JSON
metrics_data = {
    'epochs': list(range(1, num_epochs + 1)),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies
}

with open('saved_models/Fusion_CIFAR100_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)

print("训练和验证的损失与准确率已保存到 'saved_models/Fusion_CIFAR100_metrics.json'")
print("损失曲线和准确率曲线已保存到 'saved_models/' 目录下。")
