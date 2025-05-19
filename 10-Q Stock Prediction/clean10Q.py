import os
import re
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

# Suppress warning from BeautifulSoup
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

base_dir = "10-Q"

def clean_local_txt(text):
    text = re.sub(r"[^a-zA-Z0-9.,!?$%\-()\n ]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    lines = [line.strip() for line in text.splitlines() if len(line.strip())]
    return "\n".join(lines)

for ticker in os.listdir(base_dir):
    txt_dir = os.path.join(base_dir, ticker, "TXT")
    if not os.path.isdir(txt_dir):
        continue # Skip if not a directory

    print(f"Processing {ticker}...")
    for fname in os.listdir(txt_dir):
        if fname.endswith(".txt"):
            fpath = os.path.join(txt_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                cleaned = clean_local_txt(content)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                print(f"Cleaned {fname}")
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    
print("All files processed.")