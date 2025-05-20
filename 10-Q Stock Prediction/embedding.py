import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")
model = AutoModel.from_pretrained("yiyanghkust/finbert-pretrain").to(device)
model.eval()

base_dir = "10-Q"
chunk_size = 512

def embed_text_chunks(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = tokens['input_ids'][0]

    n_chunks = (len(input_ids) + chunk_size - 1) // chunk_size
    chunk_embeddings = []

    for i in range(n_chunks):
        chunk = input_ids[i*chunk_size:(i+1)*chunk_size]
        if len(chunk) == 0:
            continue
        chunk = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids=chunk)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            chunk_embeddings.append(cls_embedding)

    if chunk_embeddings:
        return sum(chunk_embeddings) / len(chunk_embeddings)
    else:
        return None

# Encode all tickers
for ticker in os.listdir(base_dir):
    txt_dir = os.path.join(base_dir, ticker, "TXT")
    embed_dir = os.path.join(base_dir, ticker, "Embedding")
    os.makedirs(embed_dir, exist_ok=True)

    if not os.path.isdir(txt_dir):
        continue

    print(f"🔍 Encoding {ticker}...")

    for fname in tqdm(os.listdir(txt_dir), desc=ticker):
        if not fname.endswith(".txt"):
            continue

        txt_path = os.path.join(txt_dir, fname)
        csv_name = fname.replace(".txt", ".csv")
        embed_path = os.path.join(embed_dir, csv_name)

        if os.path.exists(embed_path):
            continue  # Skip existing

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()

            embedding = embed_text_chunks(text)
            if embedding is not None:
                pd.DataFrame([embedding]).to_csv(embed_path, index=False)
        except Exception as e:
            print(f"⚠Error processing {fname}: {e}")

print("Done encoding all FinBERT embeddings.")

