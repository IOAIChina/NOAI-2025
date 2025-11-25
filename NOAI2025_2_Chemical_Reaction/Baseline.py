import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import math

file_name = 'data/data_train/training_data.dat'
data = pd.read_csv(file_name)

train, val = train_test_split(data, test_size=0.05, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, initial_c, t12):
        self.input = torch.tensor(initial_c, dtype=torch.float32)
        self.label = torch.tensor(t12, dtype=torch.float32)

    def __len__(self):
        return len(self.label);

    def __getitem__(self, i):
        return self.input[i], self.label[i];


# ------训练阶段 training phase--------
input_columns = [1, 2, 3]
initial_c = train.iloc[:, input_columns].values
output_columns = 5
t12_all = train.iloc[:, output_columns].values    

dataset = CustomDataset(initial_c, t12_all);
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(3, 12)
        self.fc2 = nn.Linear(12, 24)
        self.fc3 = nn.Linear(24, 36)
        self.fc4 = nn.Linear(36, 1)
    
    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.nn.functional.leaky_relu(self.fc3(x2))
        x4 = self.fc4(x3)

        return x4

num_epochs = 20
LR = 0.01
model = BaselineModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(num_epochs):
    tot_loss = 0
    batch_num = 0
    for input, label in dataloader:
        #print(input)
        #print(label)
        optimizer.zero_grad()
        output = model(input)
        output = torch.squeeze(output)
        loss = criterion(output, label)
        tot_loss += loss
        batch_num += 1
        loss.backward()
        optimizer.step()

    avg_loss = tot_loss / batch_num
    print(f'epoch {epoch}; loss {avg_loss: .4f}')


#----------验证阶段 validation phase-----------
input_columns = [1, 2, 3]
initial_c = val.iloc[:, input_columns].values
output_columns = 5
t12_all = val.iloc[:, output_columns].values

num = t12_all.size
input_ = torch.tensor(initial_c, dtype=torch.float32)
t12_act = torch.tensor(t12_all, dtype=torch.float32)

model.eval()
tot_score = 0

with torch.no_grad():
    for i in range(num):
        output = model(input_[i])
        pred = output.item()
        act = t12_act[i].item()
        score = max(0, 1 - math.log(1+0.1*abs(pred-act))/5)
        tot_score += score
        
        print(f't1/2 predicted: {pred: .4f}; t1/2 actual: {act: .4f}; score: {score}')

avg_score = tot_score / num
print(f'final score: {avg_score}')


#---------测试阶段 testing phase------------

#------------
#验证集 validation set
#参赛者可在提交平台看到public score
data_file_name = 'data/data_val/val_data_question.dat'
data = pd.read_csv(data_file_name)
input_columns = [1, 2, 3]
initial_c = data.iloc[:, input_columns].values
input_val = torch.tensor(initial_c, dtype=torch.float32)


model.eval()
pd_pred_val = pd.DataFrame(columns = ['Exp #', 't12_simulated'])

with torch.no_grad():
    for i in range(len(initial_c)):
        t12_pred = model(input_val[i])
        pred = t12_pred.item()
        pd_pred_val.loc[i, 'Exp #']= i
        pd_pred_val.loc[i, 't12_simulated'] = pred


pd_pred_val['t12_simulated'] = pd_pred_val['t12_simulated'].apply(lambda x: f"{x:.4e}")
pd_pred_val.to_csv('submission_val.csv', index=False)


#----------------
#测试集 test set
#参赛者无法获取测试集或在提交后得到测试集得分
#The testing sets and its results are made not accessible for contestants.
data_file_name = 'data/data_test/test_data_question.dat'
data = pd.read_csv(data_file_name)
input_columns = [1, 2, 3]
initial_c = data.iloc[:, input_columns].values
input_test = torch.tensor(initial_c, dtype=torch.float32)


model.eval()
pd_pred_test = pd.DataFrame(columns = ['Exp #', 't12_simulated'])

with torch.no_grad():
    for i in range(num):
        t12_pred = model(input_test[i])
        pred = t12_pred.item()
        pd_pred_test.loc[i, 'Exp #']= i
        pd_pred_test.loc[i, 't12_simulated'] = pred




pd_pred_test['t12_simulated'] = pd_pred_test['t12_simulated'].apply(lambda x: f"{x:.4e}")
pd_pred_test.to_csv('submission_test.csv', index=False)