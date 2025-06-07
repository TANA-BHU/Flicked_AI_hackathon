import os
import pandas as pd
import requests
from urllib.parse import urlparse
from tqdm import tqdm

def download_images(image_csv_path, save_dir):
    df = pd.read_csv(image_csv_path)
    grouped = df.groupby("id")["image_url"].apply(list).to_dict()

    os.makedirs(save_dir, exist_ok=True)

    for product_id, urls in tqdm(grouped.items(), desc="Downloading product images"):
        product_dir = os.path.join(save_dir, str(product_id))
        os.makedirs(product_dir, exist_ok=True)
        
        for i, url in enumerate(urls):
            try:
                parsed = urlparse(url)
                ext = os.path.splitext(parsed.path)[1]
                img_path = os.path.join(product_dir, f"img_{i}{ext}")
                
                if not os.path.exists(img_path):  
                    r = requests.get(url, timeout=10)
                    with open(img_path, 'wb') as f:
                        f.write(r.content)
            except Exception as e:
                print(f"Failed to download {url}: {e}")

    print(f"Download complete to `{save_dir}/<product_id>/img_<i>.jpg`")
