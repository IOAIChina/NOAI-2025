# 导入需要的包
# Import the required packages.
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import random
import csv
from tqdm import tqdm
import zipfile
import pandas as pd

from data_download.dataset import CustomDataset

# Create model 
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# 训练函数 Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    max_accuracy = 0 # 打印最高准确率 Print the highest accuracy.
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step,(inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            print(step, loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = eval_model(model,train_loader, device)    
        log_message = f'Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_accuracy:.4f}'
        print(log_message)
    #print("max_accuracy:", max_accuracy)

# 评估函数 Evaluation function.
def eval_model(model, data_loader, device):
    model.eval()
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            preds = outputs >= 0.5
            corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
    accuracy = corrects / total
    return accuracy

def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc = 'Test'):
            x = batch.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            preds.extend(pred.cpu().numpy())
    return preds

# Save to CSV
def save_submission_csv(preds, save_name):
    df = pd.DataFrame(preds)
    df.to_csv(save_name, index=False, header=False)


def main():

    # data loading
    train_dir = 'data_download/train'  # 数据地址 #Address of dataset
    train_file = 'data/train.csv'  # 训练集标注地址 #Address of training data annotations


    # 数据预处理 Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 创建数据加载器
    # 真实数据label为0，生成图像label为1
    # Create a data loader
    # Real data label is 0, generated image label is 1

    # Train
    train_dataset = CustomDataset(train_dir, train_file, mode="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化参数
    # initialization params
    # 设置设备 Set up device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 创建模型、损失函数和优化器 Create model, loss function, and optimizer.
    model = MyModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 10
    #Train the model and log the process.
    train_model(model, train_loader, criterion, optimizer, device, num_epochs= epochs)

    # 测试阶段
    val_dir = 'data_download/val'
    val_file = 'data/val.csv'
    test_dir = 'data_download/test'
    test_file = 'data/test.csv'

    # Val (val: public score, test: private score)
    val_dataset = CustomDataset(val_dir, val_file, mode="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Test
    test_dataset = CustomDataset(test_dir, test_file, mode="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    val_preds = predict(model, val_loader, device)
    test_preds = predict(model, test_loader, device)
    # Submission Process
    save_submission_csv(val_preds, 'submissionA.csv')
    save_submission_csv(test_preds, 'submissionB.csv')
    with zipfile.ZipFile('submission.zip', 'w') as zipf:
        zipf.write('submissionA.csv')
        zipf.write('submissionB.csv')
    os.remove('submissionA.csv')
    os.remove('submissionB.csv')



if __name__ == "__main__":
    main()
