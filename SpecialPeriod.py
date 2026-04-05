# SpecialPeriod.py (Special Period Analysis Edition)
# 1. Interactive model selection: choose one, multiple, or all models to train.
# 2. Special period analysis: evaluate model performance during crisis/boom periods
#    (e.g., 2008 Financial Crisis, 2020 COVID Crash, 2023 AI Wave).
# 3. Compares special-period vs normal-period prediction accuracy.

import matplotlib
matplotlib.use('Agg')

import yfinance as yf
import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import math
import warnings
import gc
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA as _ARIMA

warnings.filterwarnings("ignore")

# ==========================================
# 0. Configuration
# ==========================================
class Config:
    SEED = random.randint(1000, 9999)
    RUN_NAME = "SpecialPeriod_Seed"

    MARKETS = {
        'SP500': '^GSPC',
        'HSI': '^HSI'
    }

    # Predefined special periods (name -> (start, end, description))
    SPECIAL_PERIODS = {
        '2008_Financial_Crisis': ('2007-10-01', '2009-03-31', '08 Financial Crisis: subprime mortgage collapse'),
        '2010_Euro_Debt_Crisis': ('2010-04-01', '2012-06-30', 'European Sovereign Debt Crisis'),
        '2015_China_Crash':      ('2015-06-01', '2016-02-29', 'China Stock Market Crash & Global Spillover'),
        '2018_Trade_War':        ('2018-01-01', '2018-12-31', 'US-China Trade War Escalation'),
        '2020_COVID_Crash':      ('2020-02-01', '2020-06-30', 'COVID-19 Market Crash & Recovery'),
        '2022_Rate_Hike':        ('2022-01-01', '2022-12-31', 'Fed Aggressive Rate Hikes & Bear Market'),
        '2023_AI_Wave':          ('2023-01-01', '2023-12-31', 'AI Boom driven by ChatGPT & Nvidia Rally'),
    }

    SEQ_LENS = [30, 60, 90, 120, 252]
    TEST_HORIZONS = [1, 5, 10]

    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    DROPOUT = 0.2

    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    PATIENCE = 15

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_MODEL_NAMES = ['ARIMA', 'Bi-RNN', 'Bi-GRU', 'Bi-LSTM', 'PatchTST', 'Mamba', 'SNN']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. Interactive Selection
# ==========================================
def select_models():
    print("\n========== Model Selection ==========")
    print("Available models:")
    for i, name in enumerate(ALL_MODEL_NAMES):
        print(f"  {i + 1}. {name}")
    print(f"  0. ALL models")
    print("=====================================")
    user_input = input("Enter model numbers separated by commas (e.g. 1,3,5) or 0 for all: ").strip()

    if user_input == '0' or user_input == '':
        print(f"[SELECT] All models selected: {ALL_MODEL_NAMES}")
        return ALL_MODEL_NAMES[:]

    selected = []
    for part in user_input.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(ALL_MODEL_NAMES):
                selected.append(ALL_MODEL_NAMES[idx - 1])
    if not selected:
        print("[WARN] No valid selection, defaulting to ALL models.")
        return ALL_MODEL_NAMES[:]

    print(f"[SELECT] Models selected: {selected}")
    return selected

def select_markets():
    print("\n========== Market Selection ==========")
    market_list = list(Config.MARKETS.keys())
    for i, name in enumerate(market_list):
        print(f"  {i + 1}. {name} ({Config.MARKETS[name]})")
    print(f"  0. ALL markets")
    print("======================================")
    user_input = input("Enter market numbers separated by commas (e.g. 1,2) or 0 for all: ").strip()

    if user_input == '0' or user_input == '':
        print(f"[SELECT] All markets selected: {market_list}")
        return {k: Config.MARKETS[k] for k in market_list}

    selected = {}
    for part in user_input.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(market_list):
                k = market_list[idx - 1]
                selected[k] = Config.MARKETS[k]
    if not selected:
        print("[WARN] No valid selection, defaulting to ALL markets.")
        return dict(Config.MARKETS)

    print(f"[SELECT] Markets selected: {list(selected.keys())}")
    return selected

def select_special_periods():
    print("\n========== Special Period Selection ==========")
    period_list = list(Config.SPECIAL_PERIODS.keys())
    for i, name in enumerate(period_list):
        start, end, desc = Config.SPECIAL_PERIODS[name]
        print(f"  {i + 1}. {name} ({start} ~ {end})")
        print(f"       {desc}")
    print(f"  0. ALL periods")
    print("===============================================")
    user_input = input("Enter period numbers separated by commas (e.g. 1,5,7) or 0 for all: ").strip()

    if user_input == '0' or user_input == '':
        print(f"[SELECT] All periods selected.")
        return dict(Config.SPECIAL_PERIODS)

    selected = {}
    for part in user_input.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(period_list):
                k = period_list[idx - 1]
                selected[k] = Config.SPECIAL_PERIODS[k]
    if not selected:
        print("[WARN] No valid selection, defaulting to ALL periods.")
        return dict(Config.SPECIAL_PERIODS)

    print(f"[SELECT] Periods selected: {list(selected.keys())}")
    return selected

def select_seq_lens():
    print("\n========== Sequence Length Selection ==========")
    for i, sl in enumerate(Config.SEQ_LENS):
        print(f"  {i + 1}. {sl} days")
    print(f"  0. ALL sequence lengths")
    print("================================================")
    user_input = input("Enter numbers separated by commas (e.g. 1,3) or 0 for all: ").strip()

    if user_input == '0' or user_input == '':
        print(f"[SELECT] All sequence lengths: {Config.SEQ_LENS}")
        return Config.SEQ_LENS[:]

    selected = []
    for part in user_input.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(Config.SEQ_LENS):
                selected.append(Config.SEQ_LENS[idx - 1])
    if not selected:
        print("[WARN] No valid selection, defaulting to ALL.")
        return Config.SEQ_LENS[:]

    print(f"[SELECT] Sequence lengths selected: {selected}")
    return selected

def select_horizons():
    print("\n========== Forecast Horizon Selection ==========")
    for i, h in enumerate(Config.TEST_HORIZONS):
        print(f"  {i + 1}. {h}-day ahead")
    print(f"  0. ALL horizons")
    print("=================================================")
    user_input = input("Enter numbers separated by commas (e.g. 1,2) or 0 for all: ").strip()

    if user_input == '0' or user_input == '':
        print(f"[SELECT] All horizons: {Config.TEST_HORIZONS}")
        return Config.TEST_HORIZONS[:]

    selected = []
    for part in user_input.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(Config.TEST_HORIZONS):
                selected.append(Config.TEST_HORIZONS[idx - 1])
    if not selected:
        print("[WARN] No valid selection, defaulting to ALL.")
        return Config.TEST_HORIZONS[:]

    print(f"[SELECT] Horizons selected: {selected}")
    return selected

# ==========================================
# 2. Data Loading (with date index preserved)
# ==========================================
def get_data(ticker, start='2000-01-01', end='2025-01-01'):
    """Returns (scaled_data, scaler, date_index)"""
    filename_map = {'^GSPC': 'GSPC.csv', '^HSI': 'HSI.csv'}
    filename = filename_map.get(ticker, f"{ticker.replace('^','')}.csv")

    df = None
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            df.columns = [c.capitalize() for c in df.columns]
            df.index.name = 'Date'
        except:
            pass

    if df is None:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        except:
            return None, None, None
    if df is None or df.empty:
        return None, None, None

    if 'Close' not in df.columns and 'Adj close' in df.columns:
        df['Close'] = df['Adj close']
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HL_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['OC_Pct'] = (df['Close'] - df['Open']) / df['Open']
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    df = df.dropna()
    feature_cols = ['Log_Ret', 'HL_Pct', 'OC_Pct', 'RSI', 'MACD', 'ATR', 'Close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(df[feature_cols].values)

    return data, scaler, df.index

# ==========================================
# 3. Model Definitions 
# ==========================================
class BaseRecurrent(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, num_layers, dropout):
        super(BaseRecurrent, self).__init__()
        if model_type == 'BiRNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        elif model_type == 'BiGRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        elif model_type == 'BiLSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, patch_len=8, stride=4, d_model=256, nhead=4, num_layers=4, dropout=0.2):
        super(PatchTST, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = int((seq_len - patch_len) / stride) + 1
        self.patch_embedding = nn.Linear(input_dim * patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model * self.num_patches, 1)

    def forward(self, x):
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :]
            patch = patch.reshape(x.size(0), -1)
            patches.append(patch)
        x_p = torch.stack(patches, dim=1)
        x_emb = self.patch_embedding(x_p) + self.pos_embedding
        x_out = self.transformer_encoder(self.dropout(x_emb))
        x_flat = x_out.reshape(x_out.size(0), -1)
        return self.head(x_flat)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, bias=True, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x_val, res_val) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x_val = self.act(self.conv1d(x_val.transpose(1, 2))[:, :, :l].transpose(1, 2))
        x_dbl = self.x_proj(x_val)
        (dt, B, C) = x_dbl.split(split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        y = []
        h = torch.zeros(b, self.d_inner, self.d_state).to(x.device)
        for t in range(l):
            dt_t = dt[:, t, :].unsqueeze(-1)
            dA = torch.exp(dt_t * A)
            dB = dt_t * B[:, t, :].unsqueeze(1)
            h = h * dA + dB * x_val[:, t, :].unsqueeze(-1)
            y.append((h * C[:, t, :].unsqueeze(1)).sum(dim=-1))
        y = torch.stack(y, dim=1)
        y = y + x_val * self.D
        return self.out_proj(y * self.act(res_val))

class MambaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_state, d_conv, expand):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.mamba = MambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.mamba(x)
        x = self.norm(x)
        return self.fc(x[:, -1, :])

class SNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(torch.mean(x, dim=1))

# ==========================================
# 4. Model Factory, Dataset & Training
# ==========================================
def get_fresh_model(name, input_dim, win_len):
    if name == 'Bi-RNN':
        return BaseRecurrent('BiRNN', input_dim, Config.HIDDEN_DIM, Config.NUM_LAYERS, Config.DROPOUT)
    if name == 'Bi-GRU':
        return BaseRecurrent('BiGRU', input_dim, Config.HIDDEN_DIM, Config.NUM_LAYERS, Config.DROPOUT)
    if name == 'Bi-LSTM':
        return BaseRecurrent('BiLSTM', input_dim, Config.HIDDEN_DIM, Config.NUM_LAYERS, Config.DROPOUT)
    if name == 'PatchTST':
        return PatchTST(input_dim, win_len, patch_len=8, stride=4, d_model=Config.HIDDEN_DIM,
                        num_layers=Config.NUM_LAYERS, dropout=Config.DROPOUT)
    if name == 'Mamba':
        return MambaModel(input_dim, Config.HIDDEN_DIM, d_state=32, d_conv=4, expand=2)
    if name == 'SNN':
        return SNN(input_dim, Config.HIDDEN_DIM)
    return None


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, data, seq_len, horizon):
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len + self.horizon - 1, -1]
        return x, torch.tensor([y], dtype=torch.float32)

class IndexedDataset(torch.utils.data.Dataset):
    """Dataset that also returns the global index for date mapping."""
    def __init__(self, data, seq_len, horizon, global_offset=0):
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon
        self.global_offset = global_offset

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len + self.horizon - 1, -1]
        # The prediction target date index in the full dataset
        target_idx = self.global_offset + idx + self.seq_len + self.horizon - 1
        return x, torch.tensor([y], dtype=torch.float32), target_idx

def train_model(model_name, model, train_loader, val_loader):
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    best_loss = float('inf')
    early_stop = 0
    best_state = None

    for epoch in range(Config.EPOCHS):
        model.train()
        for batch in train_loader:
            x, y = batch[0], batch[1]
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0], batch[1]
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                out = model(x)
                v_loss += criterion(out, y).item()

        avg_v = v_loss / len(val_loader)

        if avg_v < best_loss:
            best_loss = avg_v
            early_stop = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            early_stop += 1
            if early_stop >= Config.PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ==========================================
# 5. Period-based Evaluation
# ==========================================
def evaluate_on_period(model, indexed_loader, date_index, period_start, period_end):
    """Evaluate model only on samples whose target date falls within [period_start, period_end]."""
    model.eval()
    preds, actuals, dates = [], [], []
    p_start = pd.Timestamp(period_start)
    p_end = pd.Timestamp(period_end)

    with torch.no_grad():
        for x, y, target_indices in indexed_loader:
            x = x.to(Config.DEVICE)
            out = model(x).cpu().numpy().flatten()
            y_np = y.numpy().flatten()
            idx_np = target_indices.numpy().flatten()

            for i in range(len(idx_np)):
                global_idx = idx_np[i]
                if global_idx < len(date_index):
                    dt = date_index[global_idx]
                    if p_start <= dt <= p_end:
                        preds.append(out[i])
                        actuals.append(y_np[i])
                        dates.append(dt)

    if len(preds) < 5:
        return None  # Not enough data points in this period

    preds = np.array(preds)
    actuals = np.array(actuals)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return {'MAE': mae, 'R2': r2, 'RMSE': rmse, 'N_samples': len(preds),
            'preds': preds, 'actuals': actuals, 'dates': dates}

def evaluate_full_test(model, indexed_loader):
    """Evaluate model on the full test set."""
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y, _ in indexed_loader:
            x = x.to(Config.DEVICE)
            preds.extend(model(x).cpu().numpy().flatten())
            actuals.extend(y.numpy().flatten())
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    return {'MAE': mae, 'R2': r2}


def evaluate_arima_on_period(data_array, date_index, p_start_str, p_end_str, win_len, horizon, order=(2, 1, 2)):
    """
    Rolling ARIMA baseline evaluated on a special period.
    Uses the last column (normalized Close) as the univariate series.
    For each target date in the period, fits ARIMA on the preceding `win_len` observations
    and forecasts `horizon` steps ahead.
    """
    close_series = data_array[:, -1]  # normalized Close
    p_start = pd.Timestamp(p_start_str)
    p_end = pd.Timestamp(p_end_str)

    preds, actuals, dates = [], [], []

    for i in range(win_len + horizon - 1, len(close_series)):
        if i >= len(date_index):
            break
        dt = date_index[i]
        if dt < p_start or dt > p_end:
            continue

        # history window for fitting
        history = close_series[i - win_len - horizon + 1: i - horizon + 1]
        if len(history) < win_len:
            continue

        try:
            model = _ARIMA(history, order=order)
            fitted = model.fit(method_kwargs={'maxiter': 200})
            fc = fitted.forecast(steps=horizon)
            pred_val = float(fc.iloc[-1]) if hasattr(fc, 'iloc') else float(fc[-1])
        except Exception:
            pred_val = float(history[-1])  # fallback: persist last value

        preds.append(pred_val)
        actuals.append(float(close_series[i]))
        dates.append(dt)

    if len(preds) < 5:
        return None

    preds = np.array(preds)
    actuals = np.array(actuals)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    return {'MAE': mae, 'R2': r2, 'RMSE': rmse, 'N_samples': len(preds),
            'preds': preds, 'actuals': actuals, 'dates': dates}

# ==========================================
# 6. Visualization
# ==========================================
def plot_period_comparison(period_results, market_name, win_len, horizon, folder):
    """Bar chart: R2 and MAE for each model across special periods vs full test."""
    if not period_results:
        return

    df = pd.DataFrame(period_results)

    # R2 comparison
    plt.figure(figsize=(14, 7))
    sns.barplot(x='Period', y='R2', hue='Model', data=df)
    plt.title(f"{market_name} | Window={win_len} | Horizon={horizon} | R² by Period")
    plt.xticks(rotation=30, ha='right')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"W{win_len}_H{horizon}_R2_by_period.png"), dpi=150)
    plt.close()

    # MAE comparison
    plt.figure(figsize=(14, 7))
    sns.barplot(x='Period', y='MAE', hue='Model', data=df)
    plt.title(f"{market_name} | Window={win_len} | Horizon={horizon} | MAE by Period")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"W{win_len}_H{horizon}_MAE_by_period.png"), dpi=150)
    plt.close()

def plot_prediction_curve(result, model_name, period_name, market_name, win_len, horizon, folder):
    """Plot predicted vs actual values for a specific period."""
    if result is None or 'preds' not in result:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(result['dates'], result['actuals'], label='Actual', color='black', linewidth=1.5)
    plt.plot(result['dates'], result['preds'], label='Predicted', color='red', linewidth=1.2, alpha=0.8)
    plt.title(f"{model_name} | {market_name} | {period_name} | W{win_len} H{horizon}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Close")
    plt.legend()
    plt.tight_layout()
    safe_name = period_name.replace(' ', '_')
    plt.savefig(os.path.join(folder, f"{model_name}_W{win_len}_H{horizon}_{safe_name}_curve.png"), dpi=150)
    plt.close()

def plot_heatmap(all_results, market_name, folder):
    """Heatmap: models x periods, colored by R2."""
    if not all_results:
        return
    df = pd.DataFrame(all_results)
    for (wl, h), grp in df.groupby(['Window', 'Horizon']):
        pivot = grp.pivot_table(index='Model', columns='Period', values='R2', aggfunc='mean')
        if pivot.empty:
            continue
        plt.figure(figsize=(14, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, linewidths=0.5)
        plt.title(f"{market_name} | W{wl} H{h} | R² Heatmap (Models × Periods)")
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"Heatmap_W{wl}_H{h}_R2.png"), dpi=150)
        plt.close()

# ==========================================
# 7. Walk-Forward Helpers
# ==========================================
def find_date_index_range(date_index, start_str, end_str):
    """Find the integer index range [start_idx, end_idx) for a date range."""
    p_start = pd.Timestamp(start_str)
    p_end = pd.Timestamp(end_str)
    mask = (date_index >= p_start) & (date_index <= p_end)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None, None
    return int(indices[0]), int(indices[-1]) + 1  # [start, end)

def build_walk_forward_loaders(data_tensor, date_index, period_start_str, win_len, horizon, batch_size,
                               val_ratio=0.1):
    """
    Walk-Forward split:
      - Training data  = all data BEFORE the period start
      - Validation data = last val_ratio of training data (for early stopping)
      - Test data       = period data (needs seq_len lookback before period start)
    Returns: train_loader, val_loader, test_loader (IndexedDataset), period_start_idx, period_end_idx
             or None if not enough data.
    """
    p_start_idx, _ = find_date_index_range(date_index, period_start_str, period_start_str)
    if p_start_idx is None:
        # period_start is before data begins; find first date >= period_start
        p_start = pd.Timestamp(period_start_str)
        later = np.where(date_index >= p_start)[0]
        if len(later) == 0:
            return None
        p_start_idx = int(later[0])

    # Need at least seq_len + horizon samples before period for training
    min_train = win_len + horizon + 50  # at least 50 usable training samples
    if p_start_idx < min_train:
        return None

    # Training portion: everything before the period
    train_data = data_tensor[:p_start_idx]

    # Split training into train / val for early stopping
    val_size = max(int(len(train_data) * val_ratio), win_len + horizon + 10)
    train_split = len(train_data) - val_size
    if train_split < min_train:
        train_split = min_train
        val_size = len(train_data) - train_split

    train_ds = DatasetWrapper(train_data[:train_split], win_len, horizon)
    val_ds = DatasetWrapper(train_data[train_split:], win_len, horizon)

    if len(train_ds) < 10 or len(val_ds) < 5:
        return None

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    # Test portion: we need seq_len lookback before the period, then predict within the period
    # Start the test tensor from (p_start_idx - win_len) so the first sample's input window
    # ends right at period start, and the target falls within the period.
    test_start = max(0, p_start_idx - win_len)
    test_data = data_tensor[test_start:]
    test_ds = IndexedDataset(test_data, win_len, horizon, global_offset=test_start)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader, test_loader

# ==========================================
# 8. Smart Period Matching & Auto Model Selection
# ==========================================

def compute_period_features(data_array, date_index, start_str, end_str):
    """
    Compute a 6-dim feature vector for a given date range:
      [mean_return, volatility, max_drawdown, skewness, kurtosis, trend_strength]
    Uses the raw (scaled) data's Close column log-returns.
    """
    p_start = pd.Timestamp(start_str)
    p_end = pd.Timestamp(end_str)
    mask = (date_index >= p_start) & (date_index <= p_end)
    indices = np.where(mask)[0]

    if len(indices) < 10:
        return None

    # Use Log_Ret column (index 0) for return stats
    log_ret = data_array[indices, 0]
    # Use Close column (last column, index -1) for drawdown
    close = data_array[indices, -1]

    mean_ret = np.mean(log_ret)
    vol = np.std(log_ret)

    # Max drawdown from Close price series (peak-to-trough)
    running_max = np.maximum.accumulate(close)
    drawdowns = (running_max - close) / np.where(running_max > 1e-9, running_max, 1.0)
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    skew = float(pd.Series(log_ret).skew())
    kurt = float(pd.Series(log_ret).kurtosis())
    trend = (np.sum(log_ret) / vol) if vol > 1e-9 else 0.0

    return np.array([mean_ret, vol, max_dd, skew, kurt, trend])


def match_period(input_features, reference_features_dict):
    """
    Match input period to the closest predefined special period using
    z-score normalized Euclidean distance.
    Returns: sorted list of (period_name, distance).
    """
    names = list(reference_features_dict.keys())
    ref_matrix = np.array([reference_features_dict[n] for n in names])

    # Stack input with references for joint z-score normalization
    all_feats = np.vstack([ref_matrix, input_features.reshape(1, -1)])
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std[std < 1e-9] = 1.0
    normed = (all_feats - mean) / std

    input_normed = normed[-1]
    ref_normed = normed[:-1]

    distances = np.sqrt(np.sum((ref_normed - input_normed) ** 2, axis=1))

    ranked = sorted(zip(names, distances), key=lambda x: x[1])
    return ranked


# ---- Market-specific, horizon-aware heuristic model rules ----
# ARIMA is benchmark only (excluded from prediction), SNN always excluded.
# PatchTST: excluded on SP500; on HSI it's usable but still weaker than top-4.
#           Re-added for >3yr trending periods on both markets.
# Format: market -> category -> horizon_band -> {models, weights}
#   horizon_band: 'short' (H<=1), 'medium' (H=2~5), 'long' (H>=6)

# === SP500 Rules ===
# Mamba dominates H=1 (4/7 periods), Bi-RNN dominates H=10 (4/7 periods)
# Models spread wide (R² 0.48~0.75 for top DL), so weight differentiation matters
SP500_MODEL_RULES = {
    'high_vol_crash': {
        # 2008: Bi-GRU(0.99) H=1, Bi-LSTM(0.95) H=5, Mamba(0.90) H=10
        # 2020: Bi-RNN(0.76) H=1, Bi-RNN(0.51) H=5, all negative H=10
        'short':  {'models': ['Mamba', 'Bi-GRU', 'Bi-RNN', 'Bi-LSTM'],
                   'weights': {'Mamba': 0.30, 'Bi-GRU': 0.25, 'Bi-RNN': 0.25, 'Bi-LSTM': 0.20}},
        'medium': {'models': ['Bi-RNN', 'Mamba', 'Bi-LSTM', 'Bi-GRU'],
                   'weights': {'Bi-RNN': 0.30, 'Mamba': 0.25, 'Bi-LSTM': 0.25, 'Bi-GRU': 0.20}},
        'long':   {'models': ['Mamba', 'Bi-RNN', 'Bi-GRU'],
                   'weights': {'Mamba': 0.40, 'Bi-RNN': 0.30, 'Bi-GRU': 0.30}},
    },
    'moderate_decline': {
        # 2010: all ~0.96 H=1; Bi-RNN(0.83) H=10
        # 2018: Mamba(0.87) H=1; Bi-RNN(0.27) H=10
        # 2022: Mamba(0.90) H=1; Mamba(0.49) H=10
        'short':  {'models': ['Mamba', 'Bi-GRU', 'Bi-RNN', 'Bi-LSTM'],
                   'weights': {'Mamba': 0.30, 'Bi-GRU': 0.25, 'Bi-RNN': 0.25, 'Bi-LSTM': 0.20}},
        'medium': {'models': ['Mamba', 'Bi-RNN', 'Bi-LSTM', 'Bi-GRU'],
                   'weights': {'Mamba': 0.30, 'Bi-RNN': 0.25, 'Bi-LSTM': 0.25, 'Bi-GRU': 0.20}},
        'long':   {'models': ['Mamba', 'Bi-RNN', 'Bi-GRU'],
                   'weights': {'Mamba': 0.40, 'Bi-RNN': 0.30, 'Bi-GRU': 0.30}},
    },
    'v_shape_recovery': {
        # 2015: Mamba(0.90) H=1; Bi-GRU(0.57) H=5; Bi-RNN(0.39) H=10
        'short':  {'models': ['Mamba', 'Bi-GRU', 'Bi-RNN', 'Bi-LSTM'],
                   'weights': {'Mamba': 0.35, 'Bi-GRU': 0.25, 'Bi-RNN': 0.20, 'Bi-LSTM': 0.20}},
        'medium': {'models': ['Bi-GRU', 'Bi-LSTM', 'Mamba', 'Bi-RNN'],
                   'weights': {'Bi-GRU': 0.30, 'Bi-LSTM': 0.25, 'Mamba': 0.25, 'Bi-RNN': 0.20}},
        'long':   {'models': ['Mamba', 'Bi-RNN', 'Bi-GRU'],
                   'weights': {'Mamba': 0.40, 'Bi-RNN': 0.30, 'Bi-GRU': 0.30}},
    },
    'trending_bull': {
        # 2023: Mamba(0.96) H=1; Bi-GRU(0.83) H=5; Bi-GRU(0.67) H=10
        'short':  {'models': ['Mamba', 'Bi-GRU', 'Bi-RNN'],
                   'weights': {'Mamba': 0.40, 'Bi-GRU': 0.35, 'Bi-RNN': 0.25}},
        'medium': {'models': ['Bi-GRU', 'Mamba', 'Bi-RNN'],
                   'weights': {'Bi-GRU': 0.35, 'Mamba': 0.35, 'Bi-RNN': 0.30}},
        'long':   {'models': ['Mamba', 'Bi-GRU', 'Bi-RNN'],
                   'weights': {'Mamba': 0.40, 'Bi-GRU': 0.30, 'Bi-RNN': 0.30}},
    },
}

# === HSI Rules ===
# Top-4 DL models are extremely close (R² 0.80~0.81), weights more uniform.
# Bi-RNN & Bi-GRU dominate H=1 (not Mamba like SP500).
# Bi-GRU is the overall best DL model on HSI.
# Bi-LSTM is much more competitive on HSI (+0.32 vs SP500).
# PatchTST is positive (0.52) but still well behind the top-4.
HSI_MODEL_RULES = {
    'high_vol_crash': {
        # 2008: Bi-RNN(0.98) H=1, Bi-RNN(0.94) H=5, Bi-LSTM(0.89) H=10
        # 2020: Bi-LSTM(0.91) H=1, Mamba(0.57) H=5, all negative H=10
        'short':  {'models': ['Bi-RNN', 'Bi-GRU', 'Bi-LSTM', 'Mamba'],
                   'weights': {'Bi-RNN': 0.30, 'Bi-GRU': 0.25, 'Bi-LSTM': 0.25, 'Mamba': 0.20}},
        'medium': {'models': ['Bi-RNN', 'Mamba', 'Bi-LSTM', 'Bi-GRU'],
                   'weights': {'Bi-RNN': 0.30, 'Mamba': 0.25, 'Bi-LSTM': 0.25, 'Bi-GRU': 0.20}},
        'long':   {'models': ['Bi-LSTM', 'Bi-RNN', 'Bi-GRU'],
                   'weights': {'Bi-LSTM': 0.40, 'Bi-RNN': 0.30, 'Bi-GRU': 0.30}},
    },
    'moderate_decline': {
        # 2010: Bi-RNN leads all 3 horizons (0.97, 0.87, 0.77)
        # 2018: Bi-GRU leads all 3 horizons (0.97, 0.86, 0.78)
        # 2022: Bi-RNN(0.97) H=1, Bi-RNN(0.86) H=5, Mamba(0.71) H=10
        'short':  {'models': ['Bi-RNN', 'Bi-GRU', 'Mamba', 'Bi-LSTM'],
                   'weights': {'Bi-RNN': 0.30, 'Bi-GRU': 0.30, 'Mamba': 0.20, 'Bi-LSTM': 0.20}},
        'medium': {'models': ['Bi-RNN', 'Bi-GRU', 'Mamba', 'Bi-LSTM'],
                   'weights': {'Bi-RNN': 0.30, 'Bi-GRU': 0.28, 'Mamba': 0.22, 'Bi-LSTM': 0.20}},
        'long':   {'models': ['Bi-GRU', 'Bi-RNN', 'Mamba'],
                   'weights': {'Bi-GRU': 0.35, 'Bi-RNN': 0.35, 'Mamba': 0.30}},
    },
    'v_shape_recovery': {
        # 2015: Bi-GRU(0.978) H=1, Bi-LSTM(0.887) H=5, Bi-GRU(0.730) H=10
        'short':  {'models': ['Bi-GRU', 'Mamba', 'Bi-LSTM', 'Bi-RNN'],
                   'weights': {'Bi-GRU': 0.30, 'Mamba': 0.25, 'Bi-LSTM': 0.25, 'Bi-RNN': 0.20}},
        'medium': {'models': ['Bi-LSTM', 'Bi-GRU', 'Mamba', 'Bi-RNN'],
                   'weights': {'Bi-LSTM': 0.30, 'Bi-GRU': 0.25, 'Mamba': 0.25, 'Bi-RNN': 0.20}},
        'long':   {'models': ['Bi-GRU', 'Bi-RNN', 'Bi-LSTM'],
                   'weights': {'Bi-GRU': 0.40, 'Bi-RNN': 0.30, 'Bi-LSTM': 0.30}},
    },
    'trending_bull': {
        # 2023: Mamba(0.967) H=1, Bi-GRU(0.853) H=5, Mamba(0.738) H=10
        # Mamba is the rare HSI case where it leads — recent/complex dynamics
        'short':  {'models': ['Mamba', 'Bi-GRU', 'Bi-LSTM', 'Bi-RNN'],
                   'weights': {'Mamba': 0.30, 'Bi-GRU': 0.25, 'Bi-LSTM': 0.25, 'Bi-RNN': 0.20}},
        'medium': {'models': ['Bi-GRU', 'Mamba', 'Bi-LSTM', 'Bi-RNN'],
                   'weights': {'Bi-GRU': 0.28, 'Mamba': 0.28, 'Bi-LSTM': 0.22, 'Bi-RNN': 0.22}},
        'long':   {'models': ['Mamba', 'Bi-GRU', 'Bi-RNN'],
                   'weights': {'Mamba': 0.35, 'Bi-GRU': 0.35, 'Bi-RNN': 0.30}},
    },
}

# Dispatch table: market_name -> rules
MARKET_MODEL_RULES = {
    'SP500': SP500_MODEL_RULES,
    'HSI':   HSI_MODEL_RULES,
}

# Map each special period name to a category
PERIOD_TO_CATEGORY = {
    '2008_Financial_Crisis': 'high_vol_crash',
    '2010_Euro_Debt_Crisis': 'moderate_decline',
    '2015_China_Crash':      'v_shape_recovery',
    '2018_Trade_War':        'moderate_decline',
    '2020_COVID_Crash':      'high_vol_crash',
    '2022_Rate_Hike':        'moderate_decline',
    '2023_AI_Wave':          'trending_bull',
}

# Models excluded from smart prediction
# PatchTST re-added only for >3yr sustained trending periods
EXCLUDED_MODELS = {'ARIMA', 'SNN', 'PatchTST'}
# Categories considered "trending" (sustained directional movement)
TRENDING_CATEGORIES = {'trending_bull', 'high_vol_crash', 'moderate_decline'}


def _get_horizon_band(horizon):
    """Map numeric horizon to band key: short / medium / long."""
    if horizon <= 1:
        return 'short'
    elif horizon <= 5:
        return 'medium'
    else:
        return 'long'


def _period_duration_years(start_str, end_str):
    """Return approximate duration in years between two date strings."""
    d1 = pd.Timestamp(start_str)
    d2 = pd.Timestamp(end_str)
    return (d2 - d1).days / 365.25


def auto_select_models(matched_period_name, horizon=5, period_start=None, period_end=None,
                       summary_csv_path=None, top_k=3, market_name='SP500'):
    """
    Automatically select models for the matched period, aware of forecast horizon
    and market (SP500 vs HSI have different optimal model mixes).

    Strategy A: If summary CSV exists, pick top-K models by R2 for that period+horizon.
    Strategy B: Use market-specific, horizon-aware heuristic rules.

    PatchTST is included when period > 3 years AND matched to a trending category.
    ARIMA and SNN are always excluded.

    Returns: (model_list, weight_dict, strategy_used)
    """
    is_long_trending = False
    category = PERIOD_TO_CATEGORY.get(matched_period_name, 'moderate_decline')
    if period_start and period_end:
        duration = _period_duration_years(period_start, period_end)
        is_long_trending = (duration > 3.0) and (category in TRENDING_CATEGORIES)

    # --- Strategy A: historical data ---
    if summary_csv_path and os.path.exists(summary_csv_path):
        try:
            df = pd.read_csv(summary_csv_path)
            period_df = df[df['Period'] == matched_period_name]
            # Filter by horizon if available
            if 'Horizon' in period_df.columns:
                horizon_df = period_df[period_df['Horizon'] == horizon]
                if len(horizon_df) > 0:
                    period_df = horizon_df
            if len(period_df) > 0:
                model_perf = period_df.groupby('Model')['R2'].mean().sort_values(ascending=False)
                # Exclude benchmark/poor models
                excluded = EXCLUDED_MODELS if not is_long_trending else {'ARIMA', 'SNN'}
                model_perf = model_perf[~model_perf.index.isin(excluded | {'Ensemble'})]
                good_models = model_perf[model_perf > 0]
                if len(good_models) >= 1:
                    selected = good_models.head(top_k)
                    models = list(selected.index)
                    raw_weights = {}
                    for m, r2 in selected.items():
                        w = max(r2, 0.01)
                        if m == 'PatchTST':
                            w *= 0.3  # heavy penalty even when included
                        raw_weights[m] = w
                    total = sum(raw_weights.values())
                    weights = {m: round(w / total, 3) for m, w in raw_weights.items()}
                    return models, weights, 'historical'
        except Exception:
            pass

    # --- Strategy B: market-specific, horizon-aware heuristic rules ---
    market_rules = MARKET_MODEL_RULES.get(market_name, SP500_MODEL_RULES)
    category = PERIOD_TO_CATEGORY.get(matched_period_name, 'moderate_decline')
    band = _get_horizon_band(horizon)
    rule = market_rules[category][band]
    models = rule['models'][:]
    weights = dict(rule['weights'])

    # Add PatchTST for >3-year trending periods (sustained bull or bear)
    if is_long_trending and 'PatchTST' not in models:
        models.append('PatchTST')
        weights['PatchTST'] = 0.20
        # Re-normalize
        total = sum(weights.values())
        weights = {m: round(w / total, 3) for m, w in weights.items()}

    return models, weights, 'heuristic'


def smart_predict(market_name, ticker, start_str, end_str, output_folder,
                  seq_lens=None, horizons=None, summary_csv_path=None):
    """
    Full auto-predict pipeline:
      1. Compute features of the input period
      2. Match to closest special period
      3. Auto-select models
      4. Train & predict with weighted ensemble
    """
    if seq_lens is None:
        seq_lens = [60]
    if horizons is None:
        horizons = [5]

    print(f"\n{'=' * 60}")
    print(f"  SMART PERIOD PREDICTION")
    print(f"  Market: {market_name} | Period: {start_str} ~ {end_str}")
    print(f"{'=' * 60}")

    # --- Load data ---
    result = get_data(ticker)
    if result[0] is None:
        print(f"[ERROR] Failed to load data for {market_name}")
        return None
    data_raw, scaler, date_index = result
    input_dim = data_raw.shape[1]
    data_tensor = torch.FloatTensor(data_raw)

    # --- Step 1: Compute features for user's period ---
    input_features = compute_period_features(data_raw, date_index, start_str, end_str)
    if input_features is None:
        print("[ERROR] Not enough data in the specified period.")
        return None

    print(f"\n[FEATURES] Input period feature vector:")
    feat_names = ['MeanReturn', 'Volatility', 'MaxDrawdown', 'Skewness', 'Kurtosis', 'TrendStrength']
    for fn, fv in zip(feat_names, input_features):
        print(f"  {fn:15s}: {fv:+.6f}")

    # --- Step 2: Compute features for all predefined periods ---
    ref_features = {}
    for pname, (ps, pe, pdesc) in Config.SPECIAL_PERIODS.items():
        feats = compute_period_features(data_raw, date_index, ps, pe)
        if feats is not None:
            ref_features[pname] = feats

    if not ref_features:
        print("[ERROR] Cannot compute reference period features from available data.")
        return None

    ranked = match_period(input_features, ref_features)
    best_period, best_dist = ranked[0]

    print(f"\n[MATCH] Period similarity ranking:")
    for i, (pname, dist) in enumerate(ranked):
        marker = " <<<" if i == 0 else ""
        desc = Config.SPECIAL_PERIODS[pname][2]
        print(f"  {i+1}. {pname:30s} distance={dist:.4f}{marker}")
        print(f"     {desc}")

    print(f"\n[RESULT] Best match: {best_period} (distance={best_dist:.4f})")

    # --- Step 3 & 4: Auto-select models per horizon, then train & predict ---
    os.makedirs(output_folder, exist_ok=True)
    all_results = []

    for win_len in seq_lens:
        for horizon in horizons:
            print(f"\n  --- W={win_len} | H={horizon} ---")

            # Select models based on horizon (different horizons → different model mix)
            models, weights, strategy = auto_select_models(
                best_period, horizon=horizon,
                period_start=start_str, period_end=end_str,
                summary_csv_path=summary_csv_path,
                market_name=market_name,
            )
            print(f"  [MODELS] Strategy: {strategy} | Horizon band: {_get_horizon_band(horizon)}")
            for m in models:
                print(f"    {m:12s}: weight={weights.get(m, 0):.3f}")

            loaders = build_walk_forward_loaders(
                data_tensor, date_index, start_str, win_len, horizon, Config.BATCH_SIZE
            )
            if loaders is None:
                print(f"  [SKIP] Not enough pre-period data for W={win_len} H={horizon}")
                continue

            train_loader, val_loader, test_loader = loaders
            model_preds = {}  # model_name -> result dict

            gc.collect()
            torch.cuda.empty_cache()

            for model_name in models:
                model = get_fresh_model(model_name, input_dim, win_len)
                if model is None:
                    continue
                model.to(Config.DEVICE)
                model = train_model(model_name, model, train_loader, val_loader)

                res = evaluate_on_period(model, test_loader, date_index, start_str, end_str)
                if res is not None:
                    model_preds[model_name] = res
                    print(f"    {model_name:12s}: MAE={res['MAE']:.4f} | R2={res['R2']:.4f}")

                    # Save prediction curve
                    plot_prediction_curve(res, model_name, f"Custom_{start_str}_{end_str}",
                                          market_name, win_len, horizon, output_folder)

                del model
                gc.collect()
                torch.cuda.empty_cache()

            # --- Weighted Ensemble ---
            if len(model_preds) >= 2:
                ens_preds = None
                ens_actuals = None
                ens_dates = None
                total_w = 0.0

                for mname, res in model_preds.items():
                    w = weights.get(mname, 1.0 / len(model_preds))
                    if ens_preds is None:
                        ens_preds = res['preds'] * w
                        ens_actuals = res['actuals']
                        ens_dates = res['dates']
                    else:
                        ens_preds += res['preds'] * w
                    total_w += w

                if ens_preds is not None and total_w > 0:
                    ens_preds /= total_w
                    e_mae = mean_absolute_error(ens_actuals, ens_preds)
                    e_r2 = r2_score(ens_actuals, ens_preds)
                    e_rmse = np.sqrt(mean_squared_error(ens_actuals, ens_preds))
                    print(f"    {'Ensemble':12s}: MAE={e_mae:.4f} | R2={e_r2:.4f} (weighted)")

                    # Plot ensemble curve
                    ens_result = {'preds': ens_preds, 'actuals': ens_actuals, 'dates': ens_dates}
                    plot_prediction_curve(ens_result, 'WeightedEnsemble',
                                          f"Custom_{start_str}_{end_str}",
                                          market_name, win_len, horizon, output_folder)

                    all_results.append({
                        'Window': win_len, 'Horizon': horizon,
                        'Matched_Period': best_period, 'Strategy': strategy,
                        'Models': '+'.join(models),
                        'Ensemble_MAE': e_mae, 'Ensemble_R2': e_r2, 'Ensemble_RMSE': e_rmse,
                    })

            elif len(model_preds) == 1:
                mname = list(model_preds.keys())[0]
                res = model_preds[mname]
                all_results.append({
                    'Window': win_len, 'Horizon': horizon,
                    'Matched_Period': best_period, 'Strategy': strategy,
                    'Models': mname,
                    'Ensemble_MAE': res['MAE'], 'Ensemble_R2': res['R2'], 'Ensemble_RMSE': res['RMSE'],
                })

    # Save results
    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(os.path.join(output_folder, "SmartPredict_Summary.csv"), index=False)
        print(f"\n[SUCCESS] Results saved to {output_folder}/SmartPredict_Summary.csv")

    return all_results


# ==========================================
# 9. Main
# ==========================================
if __name__ == "__main__":
    set_seed(Config.SEED)

    print(f"\n{'=' * 50}")
    print(f"  SPECIAL PERIOD ANALYSIS - Walk Forward")
    print(f"  Seed: {Config.SEED} | Device: {Config.DEVICE}")
    print(f"{'=' * 50}")

    # --- Mode Selection ---
    print("\n========== Mode Selection ==========")
    print("  1. Classic Mode (manual selection of models/periods)")
    print("  2. Smart Predict (auto-match period & select best models)")
    print("====================================")
    mode_input = input("Enter mode (1 or 2): ").strip()

    if mode_input == '2':
        # --- Smart Predict Mode ---
        print("\n========== Smart Predict Setup ==========")
        market_list = list(Config.MARKETS.keys())
        for i, name in enumerate(market_list):
            print(f"  {i + 1}. {name} ({Config.MARKETS[name]})")
        mkt_input = input("Select market (number): ").strip()
        mkt_idx = int(mkt_input) - 1 if mkt_input.isdigit() else 0
        mkt_idx = max(0, min(mkt_idx, len(market_list) - 1))
        market_name = market_list[mkt_idx]
        ticker = Config.MARKETS[market_name]

        start_str = input("Enter period start date (YYYY-MM-DD): ").strip()
        end_str = input("Enter period end date   (YYYY-MM-DD): ").strip()

        selected_seq_lens = select_seq_lens()
        selected_horizons = select_horizons()

        # Look for existing summary CSV from prior runs
        import glob as _glob
        existing_csvs = sorted(_glob.glob(os.path.join(".", "SpecialPeriod_Seed*", "SpecialPeriod_Summary.csv")))
        summary_csv = existing_csvs[-1] if existing_csvs else None
        if summary_csv:
            print(f"[INFO] Found historical results: {summary_csv}")

        out_folder = os.path.join(f"{Config.RUN_NAME}{Config.SEED}", "SmartPredict",
                                  market_name, f"{start_str}_{end_str}")

        smart_predict(
            market_name=market_name,
            ticker=ticker,
            start_str=start_str,
            end_str=end_str,
            output_folder=out_folder,
            seq_lens=selected_seq_lens,
            horizons=selected_horizons,
            summary_csv_path=summary_csv,
        )

        print(f"\n{'=' * 50}")
        print(f"  SMART PREDICT COMPLETE")
        print(f"{'=' * 50}")
        exit(0)

    # --- Interactive Selection ---
    selected_models = select_models()
    selected_markets = select_markets()
    selected_periods = select_special_periods()
    selected_seq_lens = select_seq_lens()
    selected_horizons = select_horizons()

    OUTPUT_FOLDER = f"{Config.RUN_NAME}{Config.SEED}"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Save run config
    with open(os.path.join(OUTPUT_FOLDER, "run_config.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Seed: {Config.SEED}\n")
        f.write(f"Models: {selected_models}\n")
        f.write(f"Markets: {list(selected_markets.keys())}\n")
        f.write(f"Periods: {list(selected_periods.keys())}\n")
        f.write(f"Seq Lens: {selected_seq_lens}\n")
        f.write(f"Horizons: {selected_horizons}\n")

    full_report = []

    for market_name, ticker in selected_markets.items():
        print(f"\n{'#' * 10} MARKET: {market_name} {'#' * 10}")
        MARKET_FOLDER = os.path.join(OUTPUT_FOLDER, market_name)
        os.makedirs(MARKET_FOLDER, exist_ok=True)

        result = get_data(ticker)
        if result[0] is None:
            print(f"[WARN] Failed to load data for {market_name}, skipping.")
            continue
        data_raw, scaler, date_index = result
        input_dim = data_raw.shape[1]
        data_tensor = torch.FloatTensor(data_raw)

        print(f"[DATA] Total: {len(data_raw)} samples")
        print(f"[DATA] Range: {date_index[0].strftime('%Y-%m-%d')} ~ {date_index[-1].strftime('%Y-%m-%d')}")

        for period_name, (p_start, p_end, p_desc) in selected_periods.items():
            print(f"\n  {'=' * 40}")
            print(f"  PERIOD: {period_name} ({p_start} ~ {p_end})")
            print(f"  {p_desc}")
            print(f"  {'=' * 40}")

            # Check if period has data in this market
            p_start_idx, p_end_idx = find_date_index_range(date_index, p_start, p_end)
            if p_start_idx is None:
                print(f"  [SKIP] No data in this date range for {market_name}.")
                continue
            n_period_samples = p_end_idx - p_start_idx
            print(f"  [INFO] Period covers indices {p_start_idx}~{p_end_idx-1} ({n_period_samples} trading days)")
            print(f"  [INFO] Training on data before {date_index[p_start_idx].strftime('%Y-%m-%d')} ({p_start_idx} samples available)")

            PERIOD_FOLDER = os.path.join(MARKET_FOLDER, period_name)
            os.makedirs(PERIOD_FOLDER, exist_ok=True)

            for win_len in selected_seq_lens:
                for horizon in selected_horizons:
                    print(f"\n    [{period_name} | W={win_len} | H={horizon}]")

                    # Build walk-forward data loaders for this period
                    loaders = build_walk_forward_loaders(
                        data_tensor, date_index, p_start, win_len, horizon, Config.BATCH_SIZE
                    )
                    if loaders is None:
                        print(f"    [SKIP] Not enough pre-period data for W={win_len} H={horizon}")
                        continue

                    train_loader, val_loader, test_loader = loaders
                    period_comparison_rows = []

                    gc.collect()
                    torch.cuda.empty_cache()

                    for model_name in selected_models:
                        print(f"      > Training {model_name}...", end="\r")

                        # --- ARIMA: statistical baseline, no PyTorch ---
                        if model_name == 'ARIMA':
                            result = evaluate_arima_on_period(
                                data_raw, date_index, p_start, p_end, win_len, horizon
                            )
                            if result is None:
                                print(f"      > ARIMA -- Not enough predictions in period")
                                continue
                            row = {
                                'Market': market_name, 'Window': win_len, 'Horizon': horizon,
                                'Model': 'ARIMA', 'Period': period_name,
                                'MAE': result['MAE'], 'R2': result['R2'],
                                'RMSE': result['RMSE'], 'N_samples': result['N_samples']
                            }
                            full_report.append(row)
                            period_comparison_rows.append(row)
                            print(f"      > ARIMA: MAE={result['MAE']:.4f} | R2={result['R2']:.4f} | n={result['N_samples']}")
                            plot_prediction_curve(result, 'ARIMA', period_name, market_name, win_len, horizon, PERIOD_FOLDER)
                            continue

                        # --- Neural network models ---
                        model = get_fresh_model(model_name, input_dim, win_len)
                        model.to(Config.DEVICE)
                        model = train_model(model_name, model, train_loader, val_loader)

                        # Save weights
                        weight_path = os.path.join(PERIOD_FOLDER, f"{model_name}_W{win_len}_H{horizon}.pth")
                        torch.save(model.state_dict(), weight_path)

                        # Evaluate only on the special period date range
                        result = evaluate_on_period(model, test_loader, date_index, p_start, p_end)
                        if result is None:
                            print(f"      > {model_name} -- Not enough predictions in period")
                            del model; gc.collect(); torch.cuda.empty_cache()
                            continue

                        row = {
                            'Market': market_name, 'Window': win_len, 'Horizon': horizon,
                            'Model': model_name, 'Period': period_name,
                            'MAE': result['MAE'], 'R2': result['R2'],
                            'RMSE': result['RMSE'], 'N_samples': result['N_samples']
                        }
                        full_report.append(row)
                        period_comparison_rows.append(row)
                        print(f"      > {model_name}: MAE={result['MAE']:.4f} | R2={result['R2']:.4f} | n={result['N_samples']}")

                        # Plot prediction curve
                        plot_prediction_curve(result, model_name, period_name, market_name, win_len, horizon, PERIOD_FOLDER)

                        del model
                        gc.collect()
                        torch.cuda.empty_cache()

                    # --- Ensemble ---
                    if len(selected_models) > 1 and period_comparison_rows:
                        ens_data = None
                        ens_count = 0
                        for model_name in selected_models:
                            if model_name == 'ARIMA':
                                continue  # ARIMA has no neural weights
                            weight_path = os.path.join(PERIOD_FOLDER, f"{model_name}_W{win_len}_H{horizon}.pth")
                            if not os.path.exists(weight_path):
                                continue
                            model = get_fresh_model(model_name, input_dim, win_len)
                            model.load_state_dict(torch.load(weight_path, weights_only=True))
                            model.to(Config.DEVICE)
                            model.eval()

                            res = evaluate_on_period(model, test_loader, date_index, p_start, p_end)
                            if res is not None:
                                if ens_data is None:
                                    ens_data = {
                                        'preds_sum': res['preds'].copy(),
                                        'actuals': res['actuals'],
                                        'dates': res['dates']
                                    }
                                else:
                                    ens_data['preds_sum'] += res['preds']
                                ens_count += 1

                            del model; gc.collect(); torch.cuda.empty_cache()

                        if ens_data is not None and ens_count > 0:
                            ens_preds = ens_data['preds_sum'] / ens_count
                            e_mae = mean_absolute_error(ens_data['actuals'], ens_preds)
                            e_r2 = r2_score(ens_data['actuals'], ens_preds)
                            e_rmse = np.sqrt(mean_squared_error(ens_data['actuals'], ens_preds))
                            row = {
                                'Market': market_name, 'Window': win_len, 'Horizon': horizon,
                                'Model': 'Ensemble', 'Period': period_name,
                                'MAE': e_mae, 'R2': e_r2, 'RMSE': e_rmse,
                                'N_samples': len(ens_data['actuals'])
                            }
                            full_report.append(row)
                            period_comparison_rows.append(row)
                            print(f"      > Ensemble: MAE={e_mae:.4f} | R2={e_r2:.4f}")

                    # Save progress after each (period, win, horizon) combo
                    if full_report:
                        pd.DataFrame(full_report).to_csv(
                            os.path.join(OUTPUT_FOLDER, "SpecialPeriod_Summary.csv"), index=False
                        )

            # Period-level plots (across all windows/horizons for this period)
            period_rows = [r for r in full_report if r['Market'] == market_name and r['Period'] == period_name]
            if period_rows:
                df_p = pd.DataFrame(period_rows)
                for h in selected_horizons:
                    sub = df_p[df_p['Horizon'] == h]
                    if sub.empty:
                        continue
                    plt.figure(figsize=(12, 6))
                    sns.barplot(x='Window', y='R2', hue='Model', data=sub)
                    plt.title(f"{market_name} | {period_name} | H={h} | R² across Windows")
                    plt.axhline(0, color='black', linewidth=0.8)
                    plt.tight_layout()
                    plt.savefig(os.path.join(PERIOD_FOLDER, f"H{h}_R2_across_windows.png"), dpi=150)
                    plt.close()

        # --- Cross-period Heatmap for this market ---
        market_results = [r for r in full_report if r['Market'] == market_name]
        plot_heatmap(market_results, market_name, MARKET_FOLDER)

        # Cross-period comparison chart: all periods side by side
        if market_results:
            df_m = pd.DataFrame(market_results)
            for (wl, h), grp in df_m.groupby(['Window', 'Horizon']):
                plt.figure(figsize=(14, 7))
                sns.barplot(x='Period', y='R2', hue='Model', data=grp)
                plt.title(f"{market_name} | W={wl} H={h} | R² Comparison Across Periods")
                plt.xticks(rotation=30, ha='right')
                plt.axhline(0, color='black', linewidth=0.8)
                plt.tight_layout()
                plt.savefig(os.path.join(MARKET_FOLDER, f"W{wl}_H{h}_R2_cross_periods.png"), dpi=150)
                plt.close()

    # ==========================================
    # 9. Final Summary
    # ==========================================
    if full_report:
        df_final = pd.DataFrame(full_report)
        df_final.to_csv(os.path.join(OUTPUT_FOLDER, "SpecialPeriod_Summary.csv"), index=False)

        print(f"\n{'=' * 60}")
        print("FINAL SUMMARY")
        print(f"{'=' * 60}")

        # Average R2 per model per period
        summary = df_final.groupby(['Model', 'Period'])[['MAE', 'R2']].mean().round(4)
        print(summary.to_string())

        # Pivot tables
        pivot_r2 = df_final.pivot_table(index='Model', columns='Period', values='R2', aggfunc='mean')
        pivot_r2.to_csv(os.path.join(OUTPUT_FOLDER, "Pivot_R2_Model_vs_Period.csv"))

        pivot_mae = df_final.pivot_table(index='Model', columns='Period', values='MAE', aggfunc='mean')
        pivot_mae.to_csv(os.path.join(OUTPUT_FOLDER, "Pivot_MAE_Model_vs_Period.csv"))

        print(f"\n[SUCCESS] All results saved to: {OUTPUT_FOLDER}/")
    else:
        print("\n[WARN] No results generated.")

    print(f"\n{'=' * 50}")
    print(f"  SPECIAL PERIOD ANALYSIS COMPLETE")
    print(f"{'=' * 50}")
