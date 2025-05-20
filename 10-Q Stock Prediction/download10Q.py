import os
import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import time
import shutil
import warnings

# Suppress XML parsed as HTML warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HEADERS = {
    "User-Agent": "Elhaam Bhuiyan NLP Project - Baruch College elhaam.bhuiyan@baruchmail.cuny.edu"
}

def get_10q_links(cik, ticker):
    cik_str = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
    try:
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"Error fetching SEC data for {ticker}: {e}")
        return []

    filings = data['filings']['recent']
    results = []

    for i in range(len(filings['form'])):
        if filings['form'][i] == '10-Q':
            acc_no = filings['accessionNumber'][i].replace("-", "")
            doc = filings['primaryDocument'][i]
            filing_date = filings['filingDate'][i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{doc}"
            results.append({'url': url, 'date': filing_date})

    return results

def convert_html_to_txt(url):
    try:
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception as e:
        print(f"Failed to parse {url}: {e}")
        return None

def save_txt(ticker, date, content):
    path = os.path.join("10-Q", ticker, "TXT")
    os.makedirs(path, exist_ok=True)
    filename = f"{ticker}-{date}.txt"
    filepath = os.path.join(path, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def process_ticker(ticker, cik):
    stock_folder = os.path.join("StockPrice", ticker)
    txt_folder = os.path.join("10-Q", ticker, "TXT")

    if not os.path.exists(stock_folder) or not any(fname.endswith(".csv") for fname in os.listdir(stock_folder)):
        print(f"{ticker} — Skipped (no stock price data)")
        return

    if os.path.exists(txt_folder) and len(os.listdir(txt_folder)) > 0:
        print(f"{ticker} — Already has 10-Q folder with files")
        return

    print(f"{ticker} — checking 10-Q filings...")

    filings = get_10q_links(cik, ticker)
    if not filings:
        print(f"🗑️ {ticker} — No 10-Qs found. Deleting stock data")
        shutil.rmtree(stock_folder, ignore_errors=True)
        print(f"{ticker} — Deleted StockPrice/{ticker} (no filings)\n")
        return

    count = 0
    for filing in filings:
        text = convert_html_to_txt(filing['url'])
        if text:
            save_txt(ticker, filing['date'], text)
            count += 1
            time.sleep(0.05)  # respectful delay

    if count == 0:
        print(f"🗑️ {ticker} — No valid 10-Qs saved. Deleting stock data")
        shutil.rmtree(stock_folder, ignore_errors=True)
        print(f"{ticker} — Deleted StockPrice/{ticker} (no usable files)\n")
    else:
        print(f"{ticker} — Saved {count} 10-Q .txt files\n")

# Load tickers and CIKs
df = pd.read_csv("company_tickers.csv").dropna(subset=["ticker", "cik_str"])

# Iterate over tickers with stock price data
for ticker in os.listdir("StockPrice"):
    row = df[df["ticker"] == ticker]
    if row.empty:
        print(f"⚠️  {ticker} not found in company_tickers.csv. Skipping...")
        continue
    cik = row.iloc[0]["cik_str"]
    process_ticker(ticker, cik)



