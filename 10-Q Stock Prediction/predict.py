import os
import re
import requests
import torch
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from transformers import AutoTokenizer, AutoModel
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import math

# ---- Suppress BeautifulSoup warnings ----
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---- Configuration ----
HEADERS = {
    "User-Agent": "Elhaam Bhuiyan NLP Project - Baruch College elhaam.bhuiyan@baruchmail.cuny.edu"
}
EMBED_MODEL = "yiyanghkust/finbert-pretrain"
TCN_WEIGHTS_PATH = "model.pth"  # your trained model file
TICKER_CIK_CSV = "company_tickers.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- FinBERT Setup ----
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
bert_model = AutoModel.from_pretrained(EMBED_MODEL).to(device).eval()

# ---- TCN Definition ----
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride, padding=0, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride, padding=0, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)

    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation
        x1 = F.pad(x, (pad, 0))
        out = self.conv1(x1)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = F.pad(out, (pad, 0))
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.0):
        super().__init__()
        layers = []
        for i, ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i-1]
            layers.append(TemporalBlock(in_ch, ch, kernel_size, 1, 2**i, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        y = self.network(x)
        return self.fc(y[:, :, -1])

# ---- SEC Scraper ----
def get_latest_10q_url(cik):
    url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        forms = data["filings"]["recent"]["form"]
        accs  = data["filings"]["recent"]["accessionNumber"]
        docs  = data["filings"]["recent"]["primaryDocument"]
        for form, acc, doc in zip(forms, accs, docs):
            if form == "10-Q":
                acc_no = acc.replace("-", "")
                return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{doc}"
    except Exception as e:
        print(f"❌ Failed to fetch 10-Q URL: {e}")
    return None

# ---- Text Cleaning ----
def clean_text_from_url(url):
    try:
        r = requests.get(url, headers=HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        raw = soup.get_text(separator="\n", strip=True)
        txt = re.sub(r"[^A-Za-z0-9.,!?$%\-()\n ]+", " ", raw)
        txt = re.sub(r"\n+", "\n", txt)
        txt = re.sub(r"\s+", " ", txt)
        return txt
    except Exception as e:
        print(f"❌ Error cleaning text: {e}")
    return None

# ---- FinBERT Embedding ----
def get_finbert_embedding(text, chunk_size=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)
    ids = tokens["input_ids"][0]
    n = (len(ids) + chunk_size - 1) // chunk_size
    embs = []
    for i in range(n):
        chunk = ids[i*chunk_size:(i+1)*chunk_size]
        if len(chunk)==0: continue
        inp = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            out = bert_model(input_ids=inp)
            embs.append(out.last_hidden_state[:,0,:].squeeze().cpu().numpy())
    return np.mean(embs, axis=0) if embs else None

# ---- Prediction Pipeline ----
def predict_from_latest_10q(ticker, cik):
    print(f"\n🔍 Fetching latest 10-Q for {ticker}...")
    url = get_latest_10q_url(cik)
    if not url:
        raise RuntimeError("No 10-Q found")
    print("📄 Cleaning text…")
    text = clean_text_from_url(url)
    if not text:
        raise RuntimeError("Failed to clean text")
    print("🤖 Computing FinBERT embedding…")
    emb = get_finbert_embedding(text)
    if emb is None:
        raise RuntimeError("Failed to embed")
    print("📦 Loading TCN model…")
    model = TCN(num_inputs=emb.shape[0], num_channels=[128,64,32], kernel_size=3).to(device)
    model.load_state_dict(torch.load(TCN_WEIGHTS_PATH, map_location=device))
    model.eval()
    x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad():
        logit = model(x).item()
    return 1 if logit>0 else 0, logit

if __name__ == "__main__":

    # load ticker→CIK map
    df_map = pd.read_csv(TICKER_CIK_CSV)
    df_map["ticker"] = df_map["ticker"].astype(str).str.upper()

    t = input("Enter stock ticker (e.g. AAPL): ").strip().upper()
    if t not in df_map["ticker"].values:
        print(f"❌ '{t}' not in {TICKER_CIK_CSV}")
        exit(1)
    cik = df_map.loc[df_map["ticker"] == t, "cik_str"].values[0]
    print(f"✅ Found CIK: {cik}")

    try:
        pred, logit = predict_from_latest_10q(t, cik)
        prob = 1 / (1 + math.exp(-logit))  # sigmoid to get probability
        label = "LONG 📈" if pred == 1 else "SHORT 📉"
        print(f"\n🔮 Prediction for {t}: {label}")
        print(f"   • Logit:       {logit:.4f}")
        print(f"   • Probability: {prob*100:.1f}%")
    except Exception as e:
        print(f"❌ {e}")

