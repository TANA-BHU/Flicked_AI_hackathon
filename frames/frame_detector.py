import cv2
import os

def extract_frames_from_video(video_path, output_dir, fps=1):

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_id}_frame{saved_count:04d}.jpg"
            frame_path = os.path.join(video_output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_id} to {video_output_dir}")

def extract_all_videos(input_dir, output_dir, fps=1):

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(('.mp4', '.mov', '.avi', '.mkv')):
            video_path = os.path.join(input_dir, file)
            extract_frames_from_video(video_path, output_dir, fps=fps)