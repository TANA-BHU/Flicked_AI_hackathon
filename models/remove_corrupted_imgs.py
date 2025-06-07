import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def clean_corrupt_images(root_dir, extensions=('.jpg', '.jpeg', '.png')):
    total_deleted = 0
    # List all folder names in the root directory
    for folder_name in tqdm(sorted(os.listdir(root_dir)), desc="Cleaning folders"):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(extensions):
                continue
            img_path = os.path.join(folder_path, filename)
            if not is_valid_image(img_path):
                try:
                    os.remove(img_path)
                    print(f"Deleted: {img_path}")
                    total_deleted += 1
                except Exception as e:
                    print(f"⚠️ Could not delete {img_path}: {e}")
    print(f"\Finished. Total corrupted images deleted: {total_deleted}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Delete corrupted/unreadable images from folders.")
    parser.add_argument("root_dir", type=str, help="Path to root directory containing image folders.")
    args = parser.parse_args()

    if os.path.isdir(args.root_dir):
        clean_corrupt_images(args.root_dir)
    else:
        print(f" Error: Directory does not exist -> {args.root_dir}")
