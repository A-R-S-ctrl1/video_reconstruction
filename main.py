import os
import argparse
import cv2

from extract import extract_frames
from filtering import filter_relevant_frames
from ssim_reorder import load_frames, calculate_ssim_matrix, reorder_frames_bidirectional
from save import save_video

def get_video_fps(video_path):
    """
    Attempts to read the FPS (frames per second) from the video metadata.
    Falls back to 25 FPS if unavailable or unreadable.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: Estimated or fallback FPS value.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 1:
        print("Warning: Could not detect FPS. Falling back to 25 fps.")
        return 25
    return fps

def main_pipeline(video_path, work_dir="temp_work", fps=25):
    """
    Full pipeline to reconstruct the original video from a corrupted one.

    Args:
        video_path (str): Path to the corrupted video.
        work_dir (str): Directory to store intermediate and output results.
        fps (int): Frames per second for the output video.

    Returns:
        str: Path to the final reconstructed video.
    """
    # Define directories
    raw_dir = os.path.join(work_dir, "frames_raw")
    filtered_dir = os.path.join(work_dir, "frames_filtered")
    output_video = os.path.join(work_dir, "reconstructed_video.mp4")

    print("\n--- Step 1: Extracting frames ---")
    extract_frames(video_path, raw_dir)

    print("\n--- Step 2: Filtering relevant frames ---")
    relevant_paths = filter_relevant_frames(raw_dir, filtered_dir)

    print("\n--- Step 3: Loading filtered frames ---")
    frames = load_frames(relevant_paths)

    print("\n--- Step 4: Computing SSIM similarity matrix ---")
    sim_matrix = calculate_ssim_matrix(frames)

    print("\n--- Step 5: Reordering frames ---")
    frame_order = reorder_frames_bidirectional(sim_matrix)
    ordered_paths = [relevant_paths[i] for i in frame_order]

    print("\n--- Step 6: Saving final video ---")
    save_video(ordered_paths, output_video, fps=fps)

    return output_video

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Reconstruct a corrupted video.")
    parser.add_argument("video_path", help="Path to the corrupted video")
    parser.add_argument("--fps", type=int, default=None, help="Optional FPS override for the output video")
    args = parser.parse_args()

    # Auto-detect FPS if not provided
    fps = args.fps if args.fps is not None else get_video_fps(args.video_path)

    # Run the full pipeline
    final_output = main_pipeline(args.video_path, fps=fps)
    print(f"\nReconstruction complete: {final_output}")
