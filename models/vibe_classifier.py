import os
import json

# Step 1: Define keywords per vibe (expand/tweak based on dataset)
vibe_keywords = {
    "Y2K": ["butterfly", "low-rise", "glitter", "vintage", "2000s"],
    "Boho": ["boho", "flowy", "earthy", "tribal", "fringe"],
    "Clean Girl": ["minimal", "slick", "bun", "dewy", "glow", "simple"],
    "Streetcore": ["urban", "street", "grunge", "baggy", "sneakers"],
    "Cottagecore": ["floral", "picnic", "cottage", "garden", "pastel"],
    "Party Glam": ["glam", "sparkle", "sequins", "night out", "heels"],
    "Coquette": ["lace", "girly", "pink", "bows", "soft", "romantic"]
}

def classify_vibe(text, top_k=3):
    text_lower = text.lower()
    scores = {vibe: 0 for vibe in vibe_keywords}

    for vibe, keywords in vibe_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                scores[vibe] += 1

    # Sort by score descending
    sorted_vibes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_vibes = [v for v, score in sorted_vibes if score > 0][:top_k]
    return top_vibes or ["Unclassified"]

def classify_vibes_from_txts(video_dir="videos"):
    results = []

    for file in os.listdir(video_dir):
        if not file.endswith(".txt"):
            continue

        video_id = file.replace(".txt", "")
        with open(os.path.join(video_dir, file), "r") as f:
            text = f.read()

        vibes = classify_vibe(text)
        results.append({
            "video_id": video_id,
            "vibes": vibes
        })

    with open("vibe_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Vibe classification complete → vibe_predictions.json")

# Run it
if __name__ == "__main__":
    classify_vibes_from_txts("videos")
