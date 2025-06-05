import pandas as pd
import os

def convert(xls_file):
    df = pd.read_excel(xls_file)

    json_file = "product_data.json"
    df.to_json(json_file, orient='records', lines=False, indent=4)

    abs_path = os.path.abspath(json_file)
    print(f" Converted {xls_file} to {abs_path}")
    return abs_path
