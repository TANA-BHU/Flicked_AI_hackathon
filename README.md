# Flickd AI Hackathon - CLI Pipeline

This repository contains a modular command-line interface (CLI) pipeline for product detection, matching, and vibe classification in videos using computer vision and retrieval-based AI.

---

##  Features

- Extract frames from input videos
-  Detect products using YOLOv8
-  Download catalog images and clean corrupt ones
-  Build FAISS index and match YOLO detections to catalog items using CLIP embeddings
-  Classify video vibes from extracted text
-  Merge all results into a final structured JSON output

---

## Directory Structure

```
Flickd_AI_Hackathon/
├── data/
│   ├── catalog_images/         # Downloaded product catalog images
│   ├── images.csv              # CSV of product image URLs
│   ├── product_data.xlsx       # Product metadata
│   └── videos/                 # Input video files
├── frames/
│   └── frames_output/          # Extracted video frames
├── models/
│   ├── faiss_store/            # FAISS index files
│   ├── detections/             # YOLO crops and detection results
│   └── yolo_model.pt           # Pretrained YOLOv8 model
├── final_output/               # Final merged JSON output
└── main.py                     # CLI script
```

---

##  How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run Individual Steps

```bash
python main.py --step extract --input-dir path/to/videos --output-dir path/to/frames
python main.py --step detect --model-path path/to/yolo_model.pt
python main.py --step catalog --csv-path data/images.csv
python main.py --step match
python main.py --step vibe
python main.py --step assemble --product-meta data/product_data.xlsx
```

### 3. Run the Full Pipeline

```bash
python main.py --step full
```

---

## 🛠️ CLI Arguments

| Argument          | Description                                         | Default                                |
|------------------|-----------------------------------------------------|----------------------------------------|
| `--step`         | Step to run: `extract`, `detect`, `catalog`, `match`, `vibe`, `assemble`, `full` | **Required** |
| `--fps`          | FPS for frame extraction                            | `1`                                    |
| `--model-path`   | YOLOv8 model directory or `.pt` file                | `models/`                              |
| `--input-dir`    | Input video directory                               | `data/videos/`                         |
| `--output-dir`   | Output frames directory                             | `frames/frames_output/`                |
| `--csv-path`     | CSV file with product image URLs                    | `data/images.csv`                      |
| `--catalog-dir`  | Directory to save downloaded catalog images         | `data/catalog_images/`                 |
| `--faiss-dir`    | Directory to store FAISS index                      | `models/faiss_store/`                  |
| `--detections-dir` | Directory to store YOLO crops and results         | `models/detections/`                   |
| `--vibe-dir`     | Directory with `.txt` files for vibe classification | `data/videos/`                         |
| `--product-meta` | Path to product metadata Excel file                 | `data/product_data.xlsx`               |
| `--final-out`    | Output directory for final JSON                     | `final_output/`                        |

---

## Output

After running `--step full`, the following files will be generated:

- `detections.json`: YOLO detection metadata
- `match_results.json`: CLIP + FAISS product matching results
- `vibe_predictions.json`: Predicted vibes from each video
- Final merged JSON with all metadata in the `final_output/` directory

---

### Final JSON Format
```json
{
  "crop_file": "2025-05-28_13-40-09_UTC_frame4_det0_person.jpg",
  "matched_product_id": "17198",
  "catalog_image": "img_5.jpg",
  "similarity": 0.8456,
  "match_type": "similar"
}


##  Project Highlights

- **YOLOv8** for object detection
- **CLIP + FAISS** for semantic product matching
- **Rule-based vibe classification** using `.txt` metadata
- **Fully CLI-controlled modular pipeline**

---

##  Author

Developed for the **Flickd AI Hackathon**  
© 2025 **Tanayendu Bari**

