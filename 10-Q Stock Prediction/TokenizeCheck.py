from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")

def is_tokenizable(filepath, max_length=512):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer(text, truncation = True, max_length= max_length, return_tensors="pt")
        return tokens["input_ids"].shape[1] > 0
    except Exception as e:
        print(f"Error processing {filepath} : {e}")
        return False
    
base_dir = "10-Q"
count = 0

for ticker in os.listdir(base_dir):
    txt_dir = os.path.join(base_dir, ticker, "TXT")
    if not os.path.isdir(txt_dir):
        continue

    for fname in os.listdir(txt_dir):
        if fname.endswith(".txt"):
            path = os.path.join(txt_dir, fname)
            count += 1
            if not is_tokenizable(path):
                print(f"Not tokenizable: {path}")
            if count % 100 == 0:
                print(f"Processed {count} files...")

print(f" Completed. Total files checked: {count}")