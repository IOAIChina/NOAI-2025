import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import math

#------------Method 1: XGBoost--------------
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# 读取训练数据
train_df = pd.read_csv("data/data_train/training_data.dat")  # 根据实际分隔符调整

L2M = train_df.iloc[:, 1].values
D = train_df.iloc[:, 2].values
L = train_df.iloc[:, 3].values

train_df['c(L2M)/c(D)'] = L2M / D
train_df['c(L)/c(D)'] = L / D
train_df['c(L)/c(L2M)'] = L / L2M
train_df['c(L2M)*c(D)'] = L2M * D
train_df['c(L)*c(D)'] = L * D
train_df['c(L)*c(L2M)'] = L * L2M
train_df['L2M_mole'] = L2M / (L2M + D + L)
train_df['D_mole'] = D / (L2M + D + L)
train_df['L_mole'] = L / (L2M + D + L)
train_df['logc(L2M)'] = np.log(1+L2M)

# 定义输入特征和标签
features = [1,2,3,9,10,11,12,13,14,15,16,17,18]
target = 5

X = train_df.iloc[:, features].values
y = train_df.iloc[:, target].values

# 数据归一化（可选，XGBoost对特征尺度不敏感，但可加速收敛）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化XGBoost回归模型
model = xgb.XGBRegressor(
    n_estimators=200,     # 树的数量
    learning_rate=0.5,    # 学习率
    max_depth=12,           # 树的最大深度
    subsample=0.8,         # 样本采样比例
    colsample_bytree=0.8,  # 特征采样比例
    objective='reg:squarederror',  # 回归任务
    random_state=42
)

# 训练模型
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    #early_stopping_rounds=50,  # 早停法防止过拟合
    verbose=10                 # 每10轮打印一次日志
)

# 预测验证集
y_pred = model.predict(X_val)

# 计算误差
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MAE: {mae:.4f}, MSE: {mse:.4f}")

# 自定义得分（根据题目公式）
def custom_score(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - np.log(1+0.1*abs(y_pred-y_true))/5))

score = custom_score(y_val, y_pred)
print(f"Custom Score: {score:.4f}")

# 读取测试数据
test_df = pd.read_csv("data/data_test/test_data_question.dat")

L2M = test_df.iloc[:, 1].values
D = test_df.iloc[:, 2].values
L = test_df.iloc[:, 3].values

test_df['c(L2M)/c(D)'] = L2M / D
test_df['c(L)/c(D)'] = L / D
test_df['c(L)/c(L2M)'] = L / L2M
test_df['c(L2M)*c(D)'] = L2M * D
test_df['c(L)*c(D)'] = L * D
test_df['c(L)*c(L2M)'] = L * L2M
test_df['L2M_mole'] = L2M / (L2M + D + L)
test_df['D_mole'] = D / (L2M + D + L)
test_df['L_mole'] = L / (L2M + D + L)
test_df['logc(L2M)'] = np.log(1+L2M)

X_test = test_df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13]].values
test_y = pd.read_csv("data/data_test/test_t12_target.dat")
y_test = test_y.iloc[:, 1].values

# 归一化（与训练集使用相同的scaler）
X_test_scaled = scaler.transform(X_test)

# 预测半衰期
t_half_pred = model.predict(X_test_scaled)

score = np.maximum(0, 1 - np.log(1+0.1*abs(t_half_pred-y_test))/5)

# 生成提交文件
submission_df = pd.DataFrame({
    "Experiment #": test_df["Experiment_#"],
    "t~1/2~": ["{:.4e}".format(x) for x in t_half_pred],  # 科学计数法保留4位小数
    #"ans": ["{:.4e}".format(x) for x in y_test],
    #"score": ["{:.4f}".format(x) for x in score]
})

print(np.mean(score))

submission_df.to_csv("submission.csv", index=False)


#----------Method 2: Neural Network-----------
file_root = "data/data_train"
file_root_p = os.listdir(file_root)
file_root_p.sort()

column_name = ['L2M_i','D_i','L_i','L2MD_i','L2Ms_i','L2M','D','L','L2MD','L2Ms']
data = pd.DataFrame(columns = column_name)

for file_name in file_root_p:
    if file_name == ".DS_Store" or file_name == ".ipynb_checkpoints" or file_name == "training_data.dat":
        continue
    
    #print(file_name)
    df = pd.read_csv(os.path.join(file_root, file_name))
    exp = df.iloc[:, 1:6].values
    features = exp[:-1, :]
    targets = exp[1:, :]
    if data.size == 0:
        for i in range(5):
            data[column_name[i]] = features[:,i]
        for i in range(5):
            data[column_name[i+5]] = targets[:,i]
    else:
        tmp = pd.DataFrame(columns = ['L2M_i','D_i','L_i','L2MD_i','L2Ms_i','L2M','D','L','L2MD','L2Ms'])
        for i in range(5):
            tmp[column_name[i]] = features[:,i]
        for i in range(5):
            tmp[column_name[i+5]] = targets[:,i]
        data = pd.concat([data, tmp], axis=0, ignore_index=True)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.input = torch.tensor(data.iloc[:, 0:5].values, dtype=torch.float32)
        self.target = torch.tensor(data.iloc[:, 5:10].values, dtype=torch.float32)

    def __len__(self):
        return self.target.shape[0];

    def __getitem__(self, i):
        return self.input[i], self.target[i];

dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#train, val = train_test_split(data, test_size=0.05, random_state=42)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(5, 12)
        self.fc2 = nn.Linear(12, 24)
        self.fc3 = nn.Linear(24, 48)
        self.fc4 = nn.Linear(48, 16)

        self.heads = nn.ModuleList([nn.Linear(16, 1) for _ in range(5)])
    
    def forward(self, x):
        
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2))
        x4 = torch.relu(self.fc4(x3))

        outputs = [head(x4) for head in self.heads]
        if outputs[0].shape[0] > 1:
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs = torch.tensor(outputs, dtype=torch.float32, requires_grad=True)

        return x+outputs


#setup training config
num_epochs = 30
LR = 0.00005
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#pd = pd.DataFrame(columns = ['ind', 'actual', 'predicted'])
#ind = 0

model.train()
for epoch in range(num_epochs):
    tot_loss = 0
    num = 0
    
    for inputs, target in dataloader:
        optimizer.zero_grad()
        output = model(inputs)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        tot_loss += loss
        num += 1
    
    avg_loss = tot_loss / num
    print(f'epoch {epoch}; loss {avg_loss}')

def t21_simulate(model, c_initial, dt, num_steps):
    input_tensor = torch.tensor(c_initial, dtype=torch.float32)
    #print(input_tensor)
    #print(input_tensor.size())
    predictions = []
    step_num = 0
    predictions.append((step_num * dt, c_initial))
    t21_found = 0
    with torch.no_grad():
        for step in range(num_steps):
            step_num += 1
            output = model(input_tensor)
            predictions.append((step_num * dt, output.squeeze(0).numpy()))
            c_L2M_t = output.squeeze(0).numpy()[0]
            if c_L2M_t < c_initial[0] / 2.0:
                #print(f"Finished @ t12 = {step_num * dt}\n")
                t21_found = step_num * dt
                break
            
            input_tensor = output  # Use the output as the input for the next step
    predictions_array = np.array([np.hstack((time, conc)) for time, conc in predictions])
    predictions_df = pd.DataFrame(predictions_array, columns=['Time', 'c_L2M', 'c_D', 'c_L', 'c_L2MD', 'c_L2Ms'])
    return predictions_df, t21_found


#Initial concentrations
c_L2MD_0 = 0.0
c_L2Ms_0 = 0.0
dt = 10.0
num_steps = 14400

#Read data file
data_file_name = 'data/data_test/test_data_question.dat'
data = pd.read_csv(data_file_name)
input_columns = [1, 2, 3]
initial_c = data.iloc[:, input_columns].values

target_file_name = 'data/data_test/test_t12_target.dat'
target = pd.read_csv(target_file_name)
output_columns = 1
t12_all = target.iloc[:, output_columns].values

model.eval()
num_exp = initial_c.shape[0]
pd_pred = pd.DataFrame(columns = ['Exp #', 't12_simulated', 't12_actual', 'score'])
pd_subm = pd.DataFrame(columns = ['Exp #', 't12_simulated'])
total_score = 0
with torch.no_grad():
    for experi in range(num_exp):
        c0 = np.concatenate((initial_c[experi], [c_L2MD_0, c_L2Ms_0]))
        #print(c0)
        _, t21_simulated = t21_simulate(model, c0, dt, num_steps)
        
        pd_pred.loc[experi, 'Exp #']= experi
        pd_pred.loc[experi, 't12_simulated'] = t21_simulated
        pd_subm.loc[experi, 'Exp #']= experi
        pd_subm.loc[experi, 't12_simulated'] = t21_simulated
        pd_pred.loc[experi, 't12_actual'] = t12_all[experi]
        score = max(0, 1 - math.log(1+0.1*abs(t21_simulated-t12_all[experi]))/5)
        total_score += score
        pd_pred.loc[experi, 'score'] = score
        print(f'Exp{experi}: t21_sim = {t21_simulated}, t21_act = {t12_all[experi]}')

total_score /= num_exp

print("total score: ", total_score)

#pd_pred.to_csv('predictions_newscore.csv', index=False)
pd_subm['t12_simulated'] = pd_pred['t12_simulated'].apply(lambda x: f"{x:.4e}")
pd_subm.to_csv('submission.csv', index=False)

# Plot the predictions
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# First subplot: t12_simulated vs. t12_actual
axs[0].plot(pd_pred['Exp #'], pd_pred['t12_simulated'], label='t12_simulated')
axs[0].plot(pd_pred['Exp #'], pd_pred['t12_actual'], label='t12_actual')
axs[0].set_xlabel('Experiment Number')
axs[0].set_ylabel('t1/2')
axs[0].legend()
axs[0].set_title('Simulation results: t12_simulated vs. t12_actual')

# Second subplot: %err vs. Experiment Number
axs[1].plot(pd_pred['Exp #'], pd_pred['score'], label='score', color='red')
axs[1].set_xlabel('Experiment Number')
axs[1].set_ylabel('Score')
axs[1].legend()
axs[1].set_title(f'Score vs. Experiment Number, total = {total_score:.2f}')

plt.tight_layout()
plt.show()

