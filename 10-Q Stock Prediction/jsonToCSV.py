import json
import csv

#load json data
with open('company_tickers.json', 'r') as f:
    data = json.load(f)

#write to a CSV file
with open('company_tickers.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['cik_str' , 'ticker', 'title'])
    writer.writeheader()
    for entry in data.values():
        writer.writerow({
            'cik_str': entry['cik_str'],
            'ticker': entry['ticker'],
            'title': entry['title']
        })
print("Converstion completed successfully!")

