# CNN_only_CIFAR100.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
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

# 3. 定义仅使用CNN的模型
class CNNOnly(nn.Module):
    def __init__(self, num_classes=100):
        super(CNNOnly, self).__init__()
        self.cnn = ResNet18CIFAR(num_classes=num_classes)
        self.cnn._initialize_weights()
        
    def forward(self, x):
        cnn_feat = self.cnn(x)  # [batch_size, num_classes]
        return cnn_feat

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
    model = CNNOnly(num_classes=100).to(device)
    
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
            torch.save(model.state_dict(), 'saved_models/best_model_CNNOnly_CIFAR100.pth')
            print('Best model saved.')
        
        print('-' * 30)
    
    # 测试模型
    model.load_state_dict(torch.load('saved_models/best_model_CNNOnly_CIFAR100.pth'))
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
    plt.title('Training and Validation Loss - CNN Only (CIFAR-100)')
    plt.legend()
    plt.grid(True)
    plt.savefig('saved_models/CNNOnly_CIFAR100_Loss.png')
    plt.show()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy - CNN Only (CIFAR-100)')
    plt.legend()
    plt.grid(True)
    plt.savefig('saved_models/CNNOnly_CIFAR100_Accuracy.png')
    plt.show()

if __name__ == '__main__':
    main()
