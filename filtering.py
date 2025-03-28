import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
import open_clip

def estimate_optimal_clusters(embeddings, max_k=5):
    """
    Estimate the optimal number of clusters using silhouette score.

    Args:
        embeddings (np.ndarray): Embeddings of image frames.
        max_k (int): Maximum number of clusters to evaluate.

    Returns:
        int: Optimal number of clusters.
    """
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)
        scores.append((k, score))

    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k

def filter_relevant_frames(input_dir, output_dir, n_clusters=None, batch_size=32, auto_cluster=True):
    """
    Filters relevant frames from a set of extracted video frames using CLIP and KMeans clustering.

    Args:
        input_dir (str): Directory containing raw frames.
        output_dir (str): Directory to save filtered frames.
        n_clusters (int, optional): Number of clusters to use. Auto-selected if not specified.
        batch_size (int): Batch size for CLIP embedding computation.
        auto_cluster (bool): Whether to automatically estimate the number of clusters.

    Returns:
        list of str: Paths to relevant frames retained from the dominant cluster.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()

    # Collect all frame image paths
    image_paths = sorted([
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith('.jpg')
    ])

    # Preprocess images using CLIP transforms
    images, valid_paths = [], []
    for path in tqdm(image_paths, desc="Preprocessing frames"):
        try:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    # Compute CLIP embeddings for all valid images
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Computing CLIP embeddings"):
            batch = torch.cat(images[i:i+batch_size]).to(device)
            emb = model.encode_image(batch).cpu().numpy()
            embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0)

    # Estimate number of clusters if needed
    if auto_cluster or n_clusters is None:
        n_clusters = estimate_optimal_clusters(embeddings)
        print(f"Auto-selected number of clusters: {n_clusters}")

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    main_cluster = np.bincount(labels).argmax()  # Select the largest cluster

    # Save only the relevant frames (belonging to the main cluster)
    os.makedirs(output_dir, exist_ok=True)
    relevant_paths = []
    for i, label in enumerate(labels):
        if label == main_cluster:
            filename = os.path.basename(valid_paths[i])
            out_path = os.path.join(output_dir, filename)
            Image.open(valid_paths[i]).save(out_path)
            relevant_paths.append(out_path)

    print(f"Kept {len(relevant_paths)} relevant frames out of {len(image_paths)}.")
    return relevant_paths
