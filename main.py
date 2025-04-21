#!/usr/bin/env python3
# rolling_cnn_lstm_all_in_one.py
"""
Excel ⟶ CNN‑LSTM ⟶ Rolling‑Window 交叉验证
•  每个窗口:  Train(固定长度) ➜ Val(内部 20 %) ➜ Test(固定长度)
•  Early‑Stopping 用 Val Loss
•  结束后把所有窗口的 Train / Val Loss 和 LR 画到一张图
"""

# ---------------- 0. 依赖 ----------------
import copy, math, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging, sys, builtins

# 1. 配置 root logger，输出到文件和 stdout
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[
                        logging.FileHandler("run.log", encoding="utf-8"),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger()

# 2. 把内建 print 替换成 logger.info
builtins.print = logger.info

torch.manual_seed(42)
np.random.seed(42)

# ---------------- 1. 全局配置 ----------------
FILE_PATH   = '2.1 training data.xlsx'   # 原始数据
WINDOW_SIZE = 20                         # 滑窗长度 (过去 20 步)
TRAIN_LEN   = 4 * 288                    # 每窗口训练片段长度
TEST_LEN    = 1 * 288                    # 每窗口测试片段长度
STEP        = TEST_LEN                   # 窗口滑动步长
INNER_VAL_RATIO = 0.2                    # 训练片段再留 20 % 做内部 Val
BATCH_SIZE  = 64
MAX_EPOCHS  = 60
PATIENCE    = 10                         # Early‑Stopping
LR          = 1e-3
DELTA       = 1.0                        # Huber δ
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- 2. 读取 & 时间特征 ----------------
df = pd.read_excel(FILE_PATH)
# 如果无标题，取消注释:
# df = pd.read_excel(FILE_PATH, names=['X1','X2','X3','X4','X5','X6','Y'])

df['X1'] = pd.to_datetime(df['X1'])
t0 = df['X1'].min()
df['rel_sec'] = (df['X1'] - t0).dt.total_seconds()
df['day_sin'] = np.sin(2 * np.pi * df['rel_sec'] / 86400)
df['day_cos'] = np.cos(2 * np.pi * df['rel_sec'] / 86400)
orig_feat = ['X2', 'X3', 'X4', 'X5', 'X6']
feature_cols = ['rel_sec', 'day_sin', 'day_cos'] + orig_feat

df = df.sort_values('X1').reset_index(drop=True)
N = len(df)

# ---------------- 工具函数 ----------------
def create_seq(data_part: pd.DataFrame, win: int):
    Xs, ys = [], []
    X_arr = data_part[feature_cols].values.astype(np.float32)
    y_arr = data_part['Y_std'].values.astype(np.float32)
    for i in range(win, len(data_part)):
        Xs.append(X_arr[i-win:i])
        ys.append(y_arr[i])
    return np.asarray(Xs), np.asarray(ys)

def to_loader(X, y, shuffle):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)

class CNNLSTM(nn.Module):
    def __init__(self, inp_dim, conv_ch=16, kernel=3, lstm_hid=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(inp_dim, conv_ch, kernel, padding=kernel//2)
        self.lstm = nn.LSTM(conv_ch, lstm_hid, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(lstm_hid, 1)
    def forward(self, x):
        x = x.permute(0,2,1)          # [B,F,T]
        x = F.relu(self.conv(x))
        x = x.permute(0,2,1)          # [B,T,C]
        h,_ = self.lstm(x)
        out = self.drop(h[:,-1,:])
        return self.fc(out)

# ---------------- 3. 全局曲线缓冲区 ----------------
all_win_train, all_win_val, all_win_lr = [], [], []
max_epoch_len = 0
rmse_all, mae_all = [], []

# ---------------- 4. 滚动窗口循环 ----------------
start, win_id = 0, 0
while start + TRAIN_LEN + TEST_LEN <= N:
    # ---- 切片 ----
    tr_slice = slice(start, start + TRAIN_LEN)
    te_slice = slice(start + TRAIN_LEN, start + TRAIN_LEN + TEST_LEN)
    train_df_full = df.iloc[tr_slice].copy()
    test_df       = df.iloc[te_slice].copy()

    # ---- 内部 Train / Val 切分 ----
    val_len = int(len(train_df_full) * INNER_VAL_RATIO)
    inner_train_df = train_df_full.iloc[:-val_len].copy()
    inner_val_df   = train_df_full.iloc[-val_len:].copy()

    # ---- 标准化 (仅用 inner_train) ----
    f_mean, f_std = inner_train_df[feature_cols].mean(), inner_train_df[feature_cols].std()
    for part in [inner_train_df, inner_val_df, test_df]:
        part[feature_cols] = (part[feature_cols] - f_mean) / f_std
    y_mean, y_std = inner_train_df['Y'].mean(), inner_train_df['Y'].std()
    for part in [inner_train_df, inner_val_df, test_df]:
        part['Y_std'] = (part['Y'] - y_mean) / y_std

    # ---- 序列 & Dataloader ----
    X_tr, y_tr = create_seq(inner_train_df, WINDOW_SIZE)
    X_val, y_val = create_seq(inner_val_df, WINDOW_SIZE)
    X_te, y_te = create_seq(test_df, WINDOW_SIZE)
    train_loader = to_loader(X_tr, y_tr, shuffle=True)
    val_loader   = to_loader(X_val, y_val, shuffle=False)
    test_loader  = to_loader(X_te, y_te, shuffle=False)

    # ---- 模型 & 优化 ----
    model = CNNLSTM(len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False)
    criterion = nn.HuberLoss(delta=DELTA)

    # ---- 窗口级别曲线缓冲 ----
    tr_losses_win, val_losses_win, lr_win = [], [], []

    best_state, best_val, no_imp = None, math.inf, 0
    for epoch in range(1, MAX_EPOCHS+1):
        # 训练
        model.train(); tot, num = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            l = criterion(model(xb), yb)
            l.backward(); optimizer.step()
            tot += l.item() * xb.size(0); num += xb.size(0)
        avg_tr = tot/num

        # 验证
        model.eval(); tot, num = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                l = criterion(model(xb), yb)
                tot += l.item() * xb.size(0); num += xb.size(0)
        avg_val = tot/num

        # 记录曲线
        tr_losses_win.append(avg_tr)
        val_losses_win.append(avg_val)
        lr_win.append(optimizer.param_groups[0]['lr'])
        scheduler.step(avg_val)

        # Early‑Stopping
        if avg_val < best_val - 1e-4:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    # 恢复最佳
    model.load_state_dict(best_state)

    # ---- 评估 test ----
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(DEVICE)).cpu()
            preds.extend(pred.numpy().flatten())
            trues.extend(yb.numpy().flatten())
    preds = np.array(preds)*y_std + y_mean
    trues = np.array(trues)*y_std + y_mean
    rmse = math.sqrt(mean_squared_error(trues, preds))
    mae  = mean_absolute_error(trues, preds)
    rmse_all.append(rmse); mae_all.append(mae)
    print(f"Window {win_id:02d}:  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # ---- 收集曲线 ----
    cur_len = len(tr_losses_win)
    max_epoch_len = max(max_epoch_len, cur_len)
    all_win_train.append(tr_losses_win)
    all_win_val.append(val_losses_win)
    all_win_lr.append(lr_win)

    # ---- 下一窗口 ----
    start += STEP
    win_id += 1

# ---------------- 5. 曲线补齐 & 可视化 ----------------
def pad(lst, L): return lst + [np.nan]*(L-len(lst))
all_train = np.array([pad(l,max_epoch_len) for l in all_win_train])
all_val   = np.array([pad(l,max_epoch_len) for l in all_win_val])
all_lr    = np.array([pad(l,max_epoch_len) for l in all_win_lr])

epochs = np.arange(max_epoch_len)
cmap = plt.get_cmap('tab10')
fig, ax1 = plt.subplots(figsize=(8,5))
for i in range(len(all_train)):
    color = cmap(i % 10)
    ax1.plot(epochs, all_train[i], color=color, alpha=0.4, lw=1)
    ax1.plot(epochs, all_val[i], color=color, alpha=0.8, lw=1.5, ls='--')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Huber Loss')

ax2 = ax1.twinx()
for i in range(len(all_lr)):
    ax2.plot(epochs, all_lr[i], color=cmap(i%10), alpha=0.3, ls=':')
ax2.set_ylabel('Learning Rate', color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')
fig.tight_layout()
fig.savefig('rolling_loss_lr_all_windows.png', dpi=150)
print('Saved rolling_loss_lr_all_windows.png')

# ---------------- 6. 汇总指标 ----------------
print("\n====== Rolling Window 汇总 ======")
print(f"窗口数 = {len(rmse_all)}")
print(f"RMSE: mean={np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f}")
print(f" MAE: mean={np.mean(mae_all):.4f} ± {np.std(mae_all):.4f}")
