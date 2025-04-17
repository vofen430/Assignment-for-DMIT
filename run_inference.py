#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载最佳模型，对 evaluating_data.xlsx 生成预测 CSV
"""

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EVAL_FILE = '2.2 evaluating data for students.xlsx'          # ← 评测集文件
CKPT_FILE = 'best_cnn_lstm.pth'            # ← 上一步保存的权重
OUT_FILE  = 'evaluating_predictions.csv'

# -------- 1. 重建模型结构 ----------
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, conv_channels=16, kernel=3,
                 lstm_hidden=64, lstm_layers=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel, padding=1)
        self.lstm  = nn.LSTM(conv_channels, lstm_hidden, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(lstm_hidden, 1)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = x.permute(0,2,1)
        out,_ = self.lstm(x)
        out = self.drop(out[:,-1,:])
        return self.fc(out)

# -------- 2. 读取 checkpoint ----------
ckpt = torch.load(CKPT_FILE, map_location=DEVICE)
feat_mean = ckpt['feat_mean'];  feat_std  = ckpt['feat_std']
Y_mean    = ckpt['Y_mean'];     Y_std     = ckpt['Y_std']
win       = ckpt['window'];     feat_cols = ckpt['feature_cols']

model = CNNLSTM(**ckpt['model_kwargs']).to(DEVICE)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# -------- 3. 载入评测数据并做同样的特征工程 ----------
df = pd.read_excel(EVAL_FILE)
df['X1'] = pd.to_datetime(df['X1'])
t0       = df['X1'].min()
df['rel_sec'] = (df['X1']-t0).dt.total_seconds()
df['day_sin'] = np.sin(2*np.pi*df['rel_sec']/86400.)
df['day_cos'] = np.cos(2*np.pi*df['rel_sec']/86400.)

# 同列顺序
df = df.sort_values('X1').reset_index(drop=True)

# 标准化
df[feat_cols] = (df[feat_cols] - feat_mean) / feat_std

# -------- 4. 生成滑动窗口并推理 ----------
X_evals, indices = [], []
feat_arr = df[feat_cols].values.astype(np.float32)
for i in range(win, len(df)):
    X_evals.append(feat_arr[i-win:i])
    indices.append(i)               # 预测对应的原始行索引

X_evals = torch.tensor(np.asarray(X_evals)).to(DEVICE)
with torch.no_grad():
    y_pred_std = model(X_evals).cpu().numpy().flatten()

# 反标准化
y_pred = y_pred_std * Y_std + Y_mean

# -------- 5. 保存结果 ----------
out_df = df.loc[indices, ['X1','X2','X3','X4','X5','X6']].copy()
# 把预测值当做 Y
out_df['Y'] = y_pred
# 按 Y, X1~X6 顺序重排
out_df = out_df[['Y','X1','X2','X3','X4','X5','X6']]
out_df.to_csv(OUT_FILE, index=False)
print(f"redictions saved to {OUT_FILE}")
