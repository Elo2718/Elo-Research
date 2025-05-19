import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import time
from itertools import product

# Load Data 
df = pd.read_csv("Master_Dataset.csv")
df = df.sort_values("trade_date")
df['trade_target'] = (df['trade_target'] == 1).astype(int)

feature_cols = [col for col in df.columns if col.startswith("emb_")]
X = df[feature_cols].values[:-500]
y = df['trade_target'].values[:-500]

# TCN 
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        pad_len = (self.kernel_size - 1) * self.dilation
        x1 = F.pad(x, (pad_len, 0))
        out = self.conv1(x1)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = F.pad(out, (pad_len, 0))
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        y = self.network(x)
        return self.fc(y[:, :, -1])

# Walk-Forward CV 
def walk_forward_cv(X, y, config, train_size=10000, val_size=1000, n_splits=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_scores = []
    total_counts = {"longs": 0, "shorts": 0}

    for i in tqdm(range(n_splits)):
        train_start = i * val_size
        train_end = train_start + train_size
        val_end = train_end + val_size
        if val_end > len(X): break

        X_train, y_train = X[train_start:train_end], y[train_start:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]

        X_train_tcn = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2).to(device)
        y_train_tcn = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_val_tcn = torch.tensor(X_val, dtype=torch.float32).unsqueeze(2).to(device)

        model = TCN(X.shape[1], config['num_channels'], config['kernel_size']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(TensorDataset(X_train_tcn, y_train_tcn),
                                  batch_size=config['batch_size'], shuffle=True)

        model.train()
        for epoch in range(config['epochs']):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val_tcn).squeeze().cpu().numpy()
            preds = (logits > 0).astype(int)

            # count longs and shorts
            longs = int((preds == 1).sum())
            shorts = int((preds == 0).sum())
            total_counts['longs'] += longs
            total_counts['shorts'] += shorts

            score = f1_score(y_val, preds, zero_division=0)
            print(f"Fold {i+1} F1 Score: {score:.4f} | longs: {longs}, shorts: {shorts}")
            val_scores.append(score)

    avg_score = np.mean(val_scores)
    print(f"\nAverage F1 across folds: {avg_score:.4f}")
    print(f"Total predictions over all folds → longs: {total_counts['longs']}, shorts: {total_counts['shorts']}")
    return avg_score

# Grid Search 
param_space = {
    'num_channels': [[128, 64, 32]],
    'kernel_size': [3],
    'lr': [0.001],
    'batch_size': [128],
    'epochs': [10]
}

param_grid = [dict(zip(param_space, v)) for v in product(*param_space.values())]
print(f"\n Testing {len(param_grid)} configurations...\n")

best_config = None
best_score = -1

for i, config in enumerate(param_grid, 1):
    print(f"🔧 Config {i}: {config}")
    start = time.time()
    avg_f1 = walk_forward_cv(X, y, config)
    elapsed = time.time() - start
    print(f" Time: {elapsed:.2f}s | Avg F1: {avg_f1:.4f}\n")
    if avg_f1 > best_score:
        best_score = avg_f1
        best_config = config

# Final Output 
print(f"\n Best Configuration:")
print(f"{best_config} --> Avg F1 Score: {best_score:.4f}")

