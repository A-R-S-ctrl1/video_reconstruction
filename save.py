import cv2
from tqdm import tqdm

def save_video(frame_paths, output_path, fps=25):
    """
    Saves a sequence of image frames into a video file.

    Args:
        frame_paths (list of str): List of file paths to image frames (in desired order).
        output_path (str): Path where the final video will be saved.
        fps (int): Frames per second for the output video.

    Raises:
        ValueError: If the frame list is empty.
    """
    if not frame_paths:
        raise ValueError("No frames provided for saving.")

    # Read the first frame to determine output video dimensions
    sample_frame = cv2.imread(frame_paths[0])
    height, width, _ = sample_frame.shape

    # Initialize video writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # Write each frame into the video
    for path in tqdm(frame_paths, desc="Saving video"):
        frame = cv2.imread(path)
        out.write(frame)

    out.release()
    print(f"Final video saved as: {output_path}")
