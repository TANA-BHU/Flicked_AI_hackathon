import os
import json
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# === Load CLIP model ===
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# === Get embedding from image ===
def get_image_embedding(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs[0].cpu().numpy()

# === Step 1: Build FAISS index from catalog ===
def build_faiss_index(catalog_dir="catalog_images", save_path="faiss_store"):
    model, processor = load_clip_model()
    os.makedirs(save_path, exist_ok=True)

    embeddings = []
    id_map = []

    print("Generating embeddings from catalog images...")
    for product_id in tqdm(os.listdir(catalog_dir), desc="Catalog Folders"):
        folder = os.path.join(catalog_dir, product_id)

        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if not file.lower().endswith(('.jpg', '.png')):
                continue
            path = os.path.join(folder, file)
            try:
                with Image.open(path) as img:
                    img.verify()
                emb = get_image_embedding(path, model, processor)
                embeddings.append(emb)
                id_map.append({
                    "product_id": product_id,
                    "image_file": file
                })
            except Exception as e:
                print(f"Error on {path}: {e}")

    if not embeddings:
        print("No valid catalog embeddings found.")
        return

    embeddings = np.stack(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(save_path, "faiss.index"))
    with open(os.path.join(save_path, "id_map.json"), "w") as f:
        json.dump(id_map, f)

    print(f"FAISS index built with {len(embeddings)} items at `{save_path}`")

# === Step 2: Match YOLO detection crops to catalog ===
def match_detections_to_catalog(detection_dir="detections", faiss_dir="faiss_store"):
    model, processor = load_clip_model()
    index = faiss.read_index(os.path.join(faiss_dir, "faiss.index"))
    with open(os.path.join(faiss_dir, "id_map.json")) as f:
        id_map = json.load(f)

    results = []

    print(" Matching detection crops to catalog...")
    for subdir in tqdm(os.listdir(detection_dir), desc="Detection Folders"):
        folder = os.path.join(detection_dir, subdir)
        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            if not file.lower().endswith(".jpg"):
                continue

            path = os.path.join(folder, file)
            try:
                emb = get_image_embedding(path, model, processor)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

            emb = emb.astype("float32").reshape(1, -1)
            faiss.normalize_L2(emb)

            D, I = index.search(emb, k=1)
            score = float(D[0][0])
            best = id_map[I[0][0]]

            match_type = "no match"
            if score > 0.9:
                match_type = "exact"
            elif score > 0.75:
                match_type = "similar"

            results.append({
                "crop_file": file,
                "matched_product_id": best["product_id"],
                "catalog_image": best["image_file"],
                "similarity": round(score, 4),
                "match_type": match_type
            })

    out_path = os.path.join(detection_dir, "match_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Matching complete. Results saved to `{out_path}`")

# # === Run both steps ===
# if __name__ == "__main__":
#     # Build catalog index
#     build_faiss_index("data/catalog_images", "models/faiss_store")

#     # Match detection crops
#     match_detections_to_catalog("detections", "models/faiss_store")
