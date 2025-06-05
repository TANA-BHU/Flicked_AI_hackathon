from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import json
import glob

RELEVANT_CLASSES = ["person", "handbag", "backpack", "tie", "suitcase", "umbrella", "shoe", "hat"]
DEFAULT_YOLO_MODEL = "yolov8n.pt"

def resolve_model_path(model_path):
    """Handle folder or file model path, fallback to downloading a default model if not found"""
    if os.path.isdir(model_path):
        pt_files = glob.glob(os.path.join(model_path, "*.pt"))
        if pt_files:
            print(f"[INFO] Found model: {pt_files[0]}")
            return pt_files[0]
        else:
            print(f"[WARN] No .pt file found in '{model_path}', downloading default model '{DEFAULT_YOLO_MODEL}'...")
            return DEFAULT_YOLO_MODEL

    elif os.path.isfile(model_path):
        return model_path

    else:
        # If it's not a file or directory, assume it's a YOLO model name (e.g., 'yolov8n.pt')
        print(f"[INFO] Model path '{model_path}' not found, assuming it is a YOLO model name.")
        return model_path

def run_yolo_and_crop(
    model_path,
    frames_dir,
    output_dir,
    conf_thresh=0.3
):
    model_path = resolve_model_path(model_path)
    model = YOLO(model_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_data = []

    for video_id in os.listdir(frames_dir):
        video_frame_dir = os.path.join(frames_dir, video_id)
        if not os.path.isdir(video_frame_dir):
            continue

        output_subdir = os.path.join(output_dir, video_id)
        os.makedirs(output_subdir, exist_ok=True)

        for frame_file in sorted(os.listdir(video_frame_dir)):
            if not frame_file.endswith(".jpg"):
                continue

            frame_path = os.path.join(video_frame_dir, frame_file)
            img = cv2.imread(frame_path)
            frame_number = int(frame_file.split("frame")[-1].split(".")[0])

            detections = model.predict(frame_path, conf=conf_thresh, verbose=False)[0]

            for i, box in enumerate(detections.boxes):
                cls_id = int(box.cls.item())
                label = model.names[cls_id]

                if label not in RELEVANT_CLASSES:
                    continue

                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = img[y1:y2, x1:x2]

                crop_filename = f"{video_id}_frame{frame_number}_det{i}_{label}.jpg"
                crop_path = os.path.join(output_subdir, crop_filename)
                cv2.imwrite(crop_path, crop)

                results_data.append({
                    "video_id": video_id,
                    "frame_number": frame_number,
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "crop_path": crop_path
                })

    with open(os.path.join(output_dir, "detections.json"), "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"Detection complete. Crops and metadata saved to `{output_dir}/`")
