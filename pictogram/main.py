#!/usr/bin/env python
# -*- coding: utf-8 -*-
# compare_periodicity.py

import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from pathlib import Path

# ---------- 文件路径 ----------
TRAIN_FILE = '../2.1 training data.xlsx'      # 训练集
EVAL_FILE  = '../evaluating_predictions.csv'  # 评测集（含预测 Y）
OUT_DIR    = Path('period_compare')
OUT_DIR.mkdir(exist_ok=True)

# ---------- 1. 读取并仅保留 X1,Y ----------
train = pd.read_excel(TRAIN_FILE, usecols=['X1','Y'])
eval_  = pd.read_csv(EVAL_FILE,  usecols=['X1','Y'])

for df in (train, eval_):
    df['X1'] = pd.to_datetime(df['X1'])
    df.sort_values('X1', inplace=True)

# ---------- 2. 5 分钟*12 → 1 小时均值 ----------
train_hour = train.set_index('X1')['Y'].resample('1H').mean()
eval_hour  = eval_.set_index('X1')['Y'].resample('1H').mean()

# ---------- 3. ACF ------------
def acf(x, nlags):
    x = np.asarray(x) - np.mean(x)
    var = np.var(x)
    n   = len(x)
    return np.array([1. if k==0 else np.sum(x[:-k]*x[k:])/((n-k)*var)
                     for k in range(nlags+1)])

max_lag = 24*7                       # 7 天
acf_tr  = acf(train_hour.ffill(), max_lag)
acf_ev  = acf(eval_hour.ffill(),  max_lag)
acf_corr = np.corrcoef(acf_tr, acf_ev)[0,1]

plt.figure(); plt.plot(acf_tr,label='Train'); plt.plot(acf_ev,label='Eval')
plt.title(f'ACF  (corr={acf_corr:.3f})'); plt.xlabel('Lag(h)'); plt.ylabel('ACF')
plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR/'acf.png')

# ---------- 4. FFT -------------
fft_tr = np.abs(np.fft.rfft(train_hour.values))
fft_ev = np.abs(np.fft.rfft(eval_hour.values))

common_len = min(len(fft_tr), len(fft_ev))
fft_tr_c   = fft_tr[:common_len]
fft_ev_c   = fft_ev[:common_len]

cos_sim = np.dot(fft_tr_c, fft_ev_c) / (
          np.linalg.norm(fft_tr_c) * np.linalg.norm(fft_ev_c) )

freqs = np.fft.rfftfreq(len(train_hour), d=1)   # cycles/hour
plt.figure(); plt.plot(freqs[1:200],fft_tr[1:200],label='Train')
plt.plot(freqs[1:200],fft_ev[1:200],label='Eval')
plt.title(f'FFT  (cos={cos_sim:.3f})'); plt.xlabel('Freq(cph)'); plt.ylabel('|X(f)|')
plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR/'fft.png')

# ---------- 5. 7‑日滚动均值 ------
roll_tr = train_hour.rolling(24*7).mean()
roll_ev = eval_hour.rolling(24*7).mean()
plt.figure(figsize=(10,3)); plt.plot(roll_tr,label='Train'); plt.plot(roll_ev,label='Eval')
plt.title('7‑day Rolling Mean'); plt.tight_layout(); plt.savefig(OUT_DIR/'rolling.png')

# ---------- 6. 日内周期箱线 -------
train['hour'] = train['X1'].dt.hour
eval_['hour'] = eval_['X1'].dt.hour
med_tr = train.groupby('hour')['Y'].median()
med_ev = eval_.groupby('hour')['Y'].median()
plt.figure(); plt.bar(med_tr.index-0.2,med_tr,width=0.4,label='Train')
plt.bar(med_ev.index+0.2,med_ev,width=0.4,label='Eval')
plt.title('Hourly Median'); plt.xlabel('Hour'); plt.ylabel('Median Y')
plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR/'hourly_median.png')

# ---------- 7. 相似度摘要 -------
print(f"ACF 相关系数 : {acf_corr:.3f}")
print(f"FFT 余弦相似度: {cos_sim:.3f}")
