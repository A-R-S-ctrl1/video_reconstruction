import os
import cv2

def extract_frames(video_path, output_dir):
    """
    Extracts frames from a video and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames.")
