# ViT_only.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import os
import matplotlib.pyplot as plt
import json

# 1. 数据加载与预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

# 2. 定义ViT模型
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

# 3. 定义仅使用ViT的模型
class ViTOnly(nn.Module):
    def __init__(self, num_classes=10):
        super(ViTOnly, self).__init__()
        self.vit = get_vit_for_cifar()
        self.fc = nn.Linear(self.vit.embed_dim, num_classes)
        
    def forward(self, x):
        vit_feat = self.vit(x)  # [batch_size, embed_dim]
        out = self.fc(vit_feat)  # [batch_size, num_classes]
        return out

# 4. 训练与评估函数
def train_model(model, loader, criterion, optimizer, device):
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

def evaluate_model(model, loader, criterion, device):
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

# 5. 主程序
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 实例化模型
    model = ViTOnly(num_classes=10).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    num_epochs = 40  # 根据需要调整
    best_acc = 0.0
    
    # 记录训练过程
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        scheduler.step()
        
        # 保存最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/best_model_ViTOnly.pth')
            print('Best model saved.')
        
        print('-' * 30)
    
    # 测试模型
    model.load_state_dict(torch.load('saved_models/best_model_ViTOnly.pth'))
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    # 可视化
    epochs = range(1, num_epochs + 1)
    
    # 绘制损失曲线
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss - ViT Only')
    plt.legend()
    plt.grid(True)
    plt.savefig('saved_models/ViTOnly_Loss.png')
    plt.show()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy - ViT Only')
    plt.legend()
    plt.grid(True)
    plt.savefig('saved_models/ViTOnly_Accuracy.png')
    plt.show()

    # 保存准确率数据
    accuracy_data = {
        'epochs': list(range(1, num_epochs + 1)),
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    
    with open('saved_models/ViTOnly_accuracies.json', 'w') as f:
        json.dump(accuracy_data, f)
    
    print("Validation accuracies saved to 'saved_models/ViTOnly_accuracies.json'")


if __name__ == '__main__':
    main()
