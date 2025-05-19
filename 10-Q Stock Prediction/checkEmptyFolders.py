import os

stockprice_dir = "StockPrice"
tenq_dir = "10-Q"

def find_empty_folders(root_dir):
    empty = []
    for dirpath, subfolders, filenames in os.walk(root_dir):
        if not subfolders and not filenames:
            empty.append(dirpath)
    return empty

empty_stock = find_empty_folders(stockprice_dir)
empty_10q = find_empty_folders(tenq_dir)

print(f"Empty folders in StockPrice: {len(empty_stock)}")
print(f"Empty folders in 10-Q: {len(empty_10q)}")