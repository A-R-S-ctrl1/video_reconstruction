import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def load_frames(frame_paths, resize_shape=(320, 180)):
    """
    Loads and preprocesses frames from given file paths.

    Args:
        frame_paths (list of str): List of frame image file paths.
        resize_shape (tuple): Size to which frames will be resized (width, height).

    Returns:
        list of np.ndarray: List of valid, preprocessed image frames.
    """
    frames = []
    for p in frame_paths:
        img = cv2.imread(p)
        if img is None:
            continue  # Skip unreadable images

        if resize_shape:
            img = cv2.resize(img, resize_shape)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Skip overly dark or flat images (likely noise or corrupted)
        if np.mean(gray) < 5 or np.std(gray) < 2:
            continue

        frames.append(img)

    return frames

def calculate_ssim_matrix(frames):
    """
    Computes a pairwise SSIM similarity matrix between all frames.

    Args:
        frames (list of np.ndarray): List of preprocessed image frames.

    Returns:
        np.ndarray: SSIM similarity matrix of shape (n_frames, n_frames).
    """
    n = len(frames)
    similarity_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Calculating SSIM matrix"):
        gray_i = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        for j in range(i, n):
            if i == j:
                similarity_matrix[i][j] = 1.0  # Perfect self-similarity
                continue
            gray_j = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
            similarity, _ = ssim(gray_i, gray_j, full=True)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Symmetric

    return similarity_matrix

def reorder_frames_bidirectional(similarity_matrix):
    """
    Reorders frames using a greedy bidirectional traversal based on SSIM similarity.

    Starts from the frame with highest average similarity (most "central") and 
    extends the sequence in both directions by choosing the next most similar unvisited frame.

    Args:
        similarity_matrix (np.ndarray): SSIM similarity matrix.

    Returns:
        list of int: Ordered indices of frames.
    """
    n = len(similarity_matrix)
    visited = [False] * n

    # Start from the frame with highest average similarity to others
    avg_sim = np.mean(similarity_matrix, axis=1)
    start = np.argmax(avg_sim)

    order = [start]
    visited[start] = True
    left = start
    right = start

    while len(order) < n:
        next_left = next_right = None
        best_sim_left = best_sim_right = -1

        for i in range(n):
            if visited[i]:
                continue

            # Find best match to current left boundary
            if similarity_matrix[i][left] > best_sim_left:
                best_sim_left = similarity_matrix[i][left]
                next_left = i

            # Find best match to current right boundary
            if similarity_matrix[right][i] > best_sim_right:
                best_sim_right = similarity_matrix[right][i]
                next_right = i

        # Extend toward the more similar candidate
        if best_sim_left > best_sim_right:
            order.insert(0, next_left)
            visited[next_left] = True
            left = next_left
        else:
            order.append(next_right)
            visited[next_right] = True
            right = next_right

    return order
