#!/usr/bin/env python
# -*- coding: utf-8 -*-
# cnn_lstm_ts_forecast.py
"""
完整时间序列预测脚本：Excel → CNN‑LSTM → 结果 CSV
修复了“系统性高估”问题
"""

# ------------------ 0. 依赖 ------------------
import os, copy, math, datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)

# ------------------ 1. 配置参数 ------------------
FILE_PATH      = '2.1 training_data.xlsx'   # 原始数据
WINDOW_SIZE    = 20                         # 序列长度
TRAIN_RATIO    = 0.70                       # 70 % train
VAL_RATIO      = 0.15                       # 15 % val，剩余 test
BATCH_SIZE     = 64
MAX_EPOCHS     = 100
PATIENCE       = 15                         # Early stopping
LR             = 1e-3
DELTA_HUBER    = 1.0                        # Huber δ
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ 2. 数据读取 ------------------
df = pd.read_excel(FILE_PATH)

# 如果没有标题行，自行指定列名
# df = pd.read_excel(FILE_PATH, names=['X1','X2','X3','X4','X5','X6','Y'])

# ------------------ 3. 时间特征工程 ------------------
df['X1'] = pd.to_datetime(df['X1'])

# 基准时刻（最早时间）
t0 = df['X1'].min()

# 相对秒数
df['rel_sec'] = (df['X1'] - t0).dt.total_seconds()

# 日周期嵌入（24 h = 86400 s）
df['day_sin'] = np.sin(2 * np.pi * df['rel_sec'] / 86400.0)
df['day_cos'] = np.cos(2 * np.pi * df['rel_sec'] / 86400.0)

# 原始数值特征（去掉旧 X1，换成 rel_sec + sin/cos）
orig_feat = ['X2', 'X3', 'X4', 'X5', 'X6']
feature_cols = ['rel_sec', 'day_sin', 'day_cos'] + orig_feat

# 按时间排序
df = df.sort_values('X1').reset_index(drop=True)

# ------------------ 4. 数据切分 ------------------
N = len(df)
train_end = int(N * TRAIN_RATIO)
val_end   = int(N * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end]
val_df   = df.iloc[train_end:val_end]
test_df  = df.iloc[val_end:]

# ------------------ 5. 标准化 ------------------
# 特征 z-score
feat_mean = train_df[feature_cols].mean()
feat_std  = train_df[feature_cols].std()
train_df[feature_cols] = (train_df[feature_cols] - feat_mean) / feat_std
val_df[feature_cols]   = (val_df[feature_cols]   - feat_mean) / feat_std
test_df[feature_cols]  = (test_df[feature_cols]  - feat_mean) / feat_std

# 目标 z-score
Y_mean = train_df['Y'].mean()
Y_std  = train_df['Y'].std()
train_df['Y_std'] = (train_df['Y'] - Y_mean) / Y_std
val_df['Y_std']   = (val_df['Y']   - Y_mean) / Y_std
test_df['Y_std']  = (test_df['Y']  - Y_mean) / Y_std

# ------------------ 6. 滑动窗口 ------------------
def create_sequences(df_part: pd.DataFrame, window: int):
    X_seq, y_seq = [], []
    feat_arr = df_part[feature_cols].values.astype(np.float32)
    y_arr    = df_part['Y_std'].values.astype(np.float32)
    for i in range(window, len(df_part)):
        X_seq.append(feat_arr[i-window:i])  # shape: (window, n_feat)
        y_seq.append(y_arr[i])
    return np.asarray(X_seq), np.asarray(y_seq)

X_train, y_train = create_sequences(train_df, WINDOW_SIZE)
X_val,   y_val   = create_sequences(val_df,   WINDOW_SIZE)
X_test,  y_test  = create_sequences(test_df,  WINDOW_SIZE)

# ------------------ 7. DataLoader ------------------
def to_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)

train_loader = to_loader(X_train, y_train, shuffle=False)  # 不打乱
val_loader   = to_loader(X_val,   y_val,   shuffle=False)
test_loader  = to_loader(X_test,  y_test,  shuffle=False)

# ------------------ 8. 模型定义 ------------------
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, conv_channels=16, kernel=3,
                 lstm_hidden=64, lstm_layers=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel, padding=1)  # same padding
        self.lstm  = nn.LSTM(conv_channels, lstm_hidden,
                              num_layers=lstm_layers, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(lstm_hidden, 1)

    def forward(self, x):                       # x: [B, T, F]
        x = x.permute(0, 2, 1)                  # -> [B, F, T]
        x = F.relu(self.conv1(x))               # -> [B, C, T]
        x = x.permute(0, 2, 1)                  # -> [B, T, C]
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]                # last time step
        out = self.drop(out)
        return self.fc(out)                     # [B, 1]

input_dim = len(feature_cols)
model = CNNLSTM(input_dim).to(DEVICE)

# ------------------ 9. 损失函数 & 优化器 ------------------
criterion = nn.HuberLoss(delta=DELTA_HUBER)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # 监控值应该“越小越好”（这里用验证损失）
    factor=0.5,        # 当触发时，lr = lr * factor
    patience=3,        # 连续 3 个 epoch 验证损失无显著下降才触发
    threshold=1e-4,    # “无显著下降”的判定阈值；可留默认值
    min_lr=1e-5,       # 学习率下限
    verbose=True       # 触发时自动打印 lr 变化
)

# ------------------ 10. 训练循环 ------------------
best_val = math.inf
epochs_no_improve = 0
best_state = None
train_losses, val_losses = [], []

for epoch in range(1, MAX_EPOCHS + 1):
    # ---- train ----
    model.train()
    tr_loss, tr_cnt = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * xb.size(0)
        tr_cnt  += xb.size(0)
    train_losses.append(tr_loss / tr_cnt)
    

    # ---- val ----
    model.eval()
    vl_loss, vl_cnt = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            vl_loss += loss.item() * xb.size(0)
            vl_cnt  += xb.size(0)
    val_losses.append(vl_loss / vl_cnt)

    # ---- early stopping ----
    if val_losses[-1] < best_val:
        best_val = val_losses[-1]
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # ---- log ----
    print(f"Epoch {epoch:3d} | "
          f"Train {train_losses[-1]:.4f} | Val {val_losses[-1]:.4f}")

# 恢复最佳模型
model.load_state_dict(best_state)

# ------------------ 11. 测试评估 ------------------
model.eval()
y_true_std, y_pred_std = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb.to(DEVICE)).cpu()
        y_true_std.extend(yb.numpy().flatten())
        y_pred_std.extend(pred.numpy().flatten())

# 反标准化
y_true = np.array(y_true_std) * Y_std + Y_mean
y_pred = np.array(y_pred_std) * Y_std + Y_mean

rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
print(f"Test RMSE = {rmse:.4f},  MAE = {mae:.4f}")

# ------------------ 12. 结果保存 ------------------
result_df = pd.DataFrame({'Y_true': y_true, 'Y_pred': y_pred})
result_df.to_csv('prediction_results.csv', index=False)
print("Saved prediction_results.csv")

# ------------------ 13. (可选) 损失曲线 ------------------
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses,   label='Val')
plt.xlabel('Epoch'); plt.ylabel('Huber Loss'); plt.legend()
plt.tight_layout(); plt.savefig('loss_curve.png')
print("Saved loss_curve.png")
