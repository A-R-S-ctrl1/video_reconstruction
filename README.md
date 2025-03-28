
# Video Reconstruction from Corrupted Footage

## Project Overview

This project solves the challenge of reconstructing a corrupted video where:
- Frames have been shuffled randomly
- Irrelevant or noisy frames have been injected

The goal is to recover the original video sequence using a general algorithm that works on any corrupted video.

---

## Pipeline Summary

### Step 1: Frame Extraction
- Script: `extract.py`
- Extracts all frames from the input video using OpenCV.

### Step 2: Frame Filtering (CLIP + KMeans)
- Script: `filtering.py`
- Computes semantic embeddings for all frames using OpenAI's CLIP model.
- Clusters frames using KMeans (auto-selects optimal number of clusters).
- Keeps only the dominant cluster, assumed to contain the relevant video content.

###  Step 3: Load & Preprocess
- Script: `ssim_reorder.py`
- Loads filtered frames and resizes them to reduce computation.
- Skips dark or low-contrast frames to remove noise.

###  Step 4: Similarity Matrix with SSIM
- Script: `ssim_reorder.py`
- Computes pairwise Structural Similarity Index (SSIM) between frames to assess visual similarity.

###  Step 5: Frame Reordering (Bidirectional)
- Script: `ssim_reorder.py`
- Uses a greedy traversal starting from the most "central" frame (highest average similarity).
- Builds a sequence forward and backward to preserve temporal coherence.

###  Step 6: Save Output Video
- Script: `save.py`
- Writes the reordered frames back into a `.mp4` video using OpenCV.

---

##  How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Input
Place your corrupted video anywhere and run the script with the path:
```bash
python main.py path/to/corrupted_video.mp4
```

You can optionally override the FPS like this:
```bash
python main.py path/to/corrupted_video.mp4 --fps 30
```

### 3. Output
The reconstructed video will be saved in:
```
temp_work/reconstructed_video.mp4
```

---

##  Generalization & Scalability Notes

- The method is domain-independent works on any video content.
- CLIP enables robust semantic filtering, not just pixel-wise comparisons.
- The number of visual clusters is automatically estimated using silhouette score.
- SSIM ensures visual continuity.

---

## Project Structure

```
video_reconstruction/
├── extract.py         # Extracts frames from video
├── filtering.py       # Filters relevant frames using CLIP + KMeans
├── ssim_reorder.py    # Computes SSIM and reorders frames
├── save.py            # Saves final output video
├── main.py            # Pipeline orchestrator with CLIP
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## Author

Aruna Ramasamy 
Machine Learning Engineer  
