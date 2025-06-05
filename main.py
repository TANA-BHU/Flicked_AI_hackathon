import argparse
from frames.frame_detector import extract_all_videos
from models.download_catalog_images import download_images
from models.yolo_detector import run_yolo_and_crop
from models.clip_faiss_pipeline import build_faiss_index, match_detections_to_catalog
from models.vibe_classifier import classify_vibes_from_txts
from final_ouput.final_assambler import merge_into_final_output
from data.clean_images import clean_corrupt_images
from utils.convert_to_json import convert

def main():
    parser = argparse.ArgumentParser(description="Flickd AI Hackathon - CLI Pipeline")
    parser.add_argument('--step', type=str, required=True,
                        choices=['extract', 'detect', 'catalog', 'match', 'vibe', 'assemble', 'full'],
                        help='Step to run: extract | detect | catalog | match | vibe | assemble | full')
    parser.add_argument('--fps', type=int, default=1, help='FPS for frame extraction')
    parser.add_argument('--model-path', type=str, default='/home/elyssa/Flickd_AI_Hackathon/models', help='Path to YOLOv8 model')
    parser.add_argument('--input-dir', type=str, default='/home/elyssa/Flickd_AI_Hackathon/data/videos', help='Input video directory')
    parser.add_argument('--output-dir', type=str, default='/home/elyssa/Flickd_AI_Hackathon/frames/frames_output/', help='Output frames directory')
    parser.add_argument('--csv-path', type=str, default='/home/elyssa/Flickd_AI_Hackathon/data/images.csv', help='CSV file for catalog image URLs')
    parser.add_argument('--catalog-dir', type=str, default='/home/elyssa/Flickd_AI_Hackathon/data/catalog_images', help='Where to store downloaded catalog images')
    parser.add_argument('--faiss-dir', type=str, default='/home/elyssa/Flickd_AI_Hackathon/models/faiss_store', help='Directory to store FAISS index')
    parser.add_argument('--detections-dir', type=str, default='/home/elyssa/Flickd_AI_Hackathon/models/detections', help='Directory with YOLO crops')
    parser.add_argument('--vibe-dir', type=str, default='/home/elyssa/Flickd_AI_Hackathon/data/videos', help='Directory containing .txt files for vibe classification')
    parser.add_argument('--product-meta', type=str, default='/home/elyssa/Flickd_AI_Hackathon/data/product_data.xlsx', help='JSON with product metadata')
    parser.add_argument('--final-out', type=str, default='final_output', help='Directory to store final outputs')

    args = parser.parse_args()

    if args.step == 'extract' or args.step == 'full':
        print("Extracting frames...")
        extract_all_videos(args.input_dir, args.output_dir, fps=args.fps)

    if args.step == 'detect' or args.step == 'full':
        print("Running YOLOv8 detection...")
        run_yolo_and_crop(model_path=args.model_path, frames_dir=args.output_dir, output_dir=args.detections_dir)

    if args.step == 'catalog' or args.step == 'full':
        print("Downloading catalog images...")
        download_images(args.csv_path, save_dir=args.catalog_dir)

        print("Removing the unsupported images.. ")
        clean_corrupt_images(args.catalog_dir)

        print("Building FAISS index...")
        build_faiss_index(args.catalog_dir, save_path=args.faiss_dir)

    if args.step == 'match' or args.step == 'full':
        print(" Matching YOLO crops to catalog...")
        match_detections_to_catalog(args.detections_dir, faiss_dir=args.faiss_dir)

    if args.step == 'vibe' or args.step == 'full':
        print("Classifying vibes...")
        classify_vibes_from_txts(video_dir=args.vibe_dir)

    if args.step == 'assemble' or args.step == 'full':
        print("Merging into final JSON output...")
        product_data_json = convert(args.product_meta)
        print(f" json path: {product_data_json}")
        merge_into_final_output(
            vibe_path="vibe_predictions.json",
            detection_path=f"{args.detections_dir}/detections.json",
            match_path=f"{args.detections_dir}/match_results.json",
            product_meta_path=product_data_json,  # ✅ use converted JSON path
            output_dir=args.final_out
        )


    print("***Done!***")

if __name__ == "__main__":
    main()
                                                                                                                                                                                                                                                                                