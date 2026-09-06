"""
clean_videos.py
Person A's script: Steps 1-3 of dataset cleaning
  Step 1: Video integrity check (find broken/corrupt videos)
  Step 2: Duplicate detection (find identical files)
  Step 3: Label/folder consistency check (verify folder structure matches docs)

HOW TO RUN:
    python clean_videos.py

WHAT YOU NEED TO EDIT BEFORE RUNNING:
    1. Set RAW_DATA_DIR to where your videos actually live
    2. Set EXPECTED_LABELS to match your dataset (FF++ or Celeb-DF — see comments below)

OUTPUT:
    - logs/cleaning_log.csv        -> summary numbers (for your report)
    - logs/broken_videos.txt       -> list of corrupt video paths
    - logs/duplicate_videos.txt    -> list of duplicate video paths
    - logs/label_mismatches.txt    -> any folders that don't match expected labels
    - logs/clean_video_list.csv    -> FINAL list of valid, unique, correctly-labeled
                                       videos. Hand this file to Person B — it's the
                                       input to their face-detection step.
"""

import os
import hashlib
import csv
from pathlib import Path
import cv2  # from opencv-python

# ── EDIT THESE TWO SETTINGS FOR YOUR PROJECT ────────────────────────────────

# Where your raw videos are stored (point this at your actual data folder)
RAW_DATA_DIR = "data/raw"

# What folder names you EXPECT to see, based on official dataset docs.
# This project has BOTH FaceForensics++ (C23) and Celeb-DF, so we list
# every real folder name found in both datasets.
EXPECTED_LABELS = [
    # FaceForensics++_C23 folders
    "DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter",
    "FaceSwap", "NeuralTextures", "original",
    # cel (Celeb-DF) folders
    "Celeb-real", "Celeb-synthesis", "YouTube-real",
]

# Video file extensions to look for
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# ── SETUP ────────────────────────────────────────────────────────────────

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def find_all_videos(root_dir):
    """Walk through every subfolder and collect all video file paths."""
    video_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                video_paths.append(os.path.join(dirpath, fname))
    return video_paths


# ── STEP 1: VIDEO INTEGRITY CHECK ───────────────────────────────────────────

def check_video_integrity(video_paths):
    """
    Try to open each video and read its first frame.
    If it fails to open, or has 0 frames, or the first frame can't be read,
    we consider it broken.
    Returns: (list_of_good_paths, list_of_broken_paths)
    """
    good, broken = [], []

    for path in video_paths:
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            broken.append(path)
            cap.release()
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            broken.append(path)
            cap.release()
            continue

        ret, _ = cap.read()  # try reading the very first frame
        if not ret:
            broken.append(path)
        else:
            good.append(path)

        cap.release()

    return good, broken


# ── STEP 2: DUPLICATE DETECTION ─────────────────────────────────────────────

def hash_file(path, block_size=65536):
    """Compute an MD5 fingerprint of a file's contents."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            md5.update(chunk)
    return md5.hexdigest()


def find_duplicates(video_paths):
    """
    Hash every video. If two videos share the same hash, they're identical
    files (even if named differently or in different folders).
    Returns: (list_of_unique_paths, list_of_duplicate_paths)
    """
    seen_hashes = {}   # hash -> first path we saw with that hash
    unique, duplicates = [], []

    for path in video_paths:
        file_hash = hash_file(path)
        if file_hash in seen_hashes:
            duplicates.append(path)  # this one is a repeat -> mark for removal
        else:
            seen_hashes[file_hash] = path
            unique.append(path)

    return unique, duplicates


# ── STEP 3: LABEL / FOLDER CONSISTENCY CHECK ────────────────────────────────

def check_label_consistency(video_paths, expected_labels):
    """
    For each video, check that ONE of its parent folder names matches
    an expected label. Flags anything that doesn't match so you can
    catch mislabeled folders before they poison your dataset.
    Returns: (list_of_correctly_labeled_paths, list_of_mismatched_paths)
    """
    correctly_labeled, mismatched = [], []

    for path in video_paths:
        # get all folder names in this path, lowercase for safe comparison
        parts = [p.lower() for p in Path(path).parts]
        expected_lower = [label.lower() for label in expected_labels]

        if any(label in parts for label in expected_lower):
            correctly_labeled.append(path)
        else:
            mismatched.append(path)

    return correctly_labeled, mismatched


# ── MAIN: RUN ALL THREE STEPS IN ORDER ──────────────────────────────────────

def main():
    print(f"Scanning {RAW_DATA_DIR} for videos...")
    all_videos = find_all_videos(RAW_DATA_DIR)
    print(f"Found {len(all_videos)} video files.\n")

    # STEP 1
    print("Step 1: Checking video integrity...")
    good_videos, broken_videos = check_video_integrity(all_videos)
    print(f"  {len(good_videos)} OK, {len(broken_videos)} broken.\n")

    # STEP 2 (only check duplicates among the GOOD videos)
    print("Step 2: Checking for duplicates...")
    unique_videos, duplicate_videos = find_duplicates(good_videos)
    print(f"  {len(unique_videos)} unique, {len(duplicate_videos)} duplicates.\n")

    # STEP 3 (only check labels on the GOOD, UNIQUE videos)
    print("Step 3: Checking label/folder consistency...")
    labeled_ok, mismatched = check_label_consistency(unique_videos, EXPECTED_LABELS)
    print(f"  {len(labeled_ok)} correctly labeled, {len(mismatched)} mismatched.\n")

    # ── WRITE LOG FILES ──
    with open(f"{LOG_DIR}/broken_videos.txt", "w") as f:
        f.write("\n".join(broken_videos))

    with open(f"{LOG_DIR}/duplicate_videos.txt", "w") as f:
        f.write("\n".join(duplicate_videos))

    with open(f"{LOG_DIR}/label_mismatches.txt", "w") as f:
        f.write("\n".join(mismatched))

    # Final clean list -> this is what Person B's Kaggle notebook will read
    with open(f"{LOG_DIR}/clean_video_list.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path"])
        for path in labeled_ok:
            writer.writerow([path])

    # Summary log -> the numbers you need for your report/defense
    with open(f"{LOG_DIR}/cleaning_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "input_count", "removed_count", "output_count", "reason"])
        writer.writerow(["1_video_integrity", len(all_videos), len(broken_videos),
                          len(good_videos), "failed to open / unreadable"])
        writer.writerow(["2_duplicate_detection", len(good_videos), len(duplicate_videos),
                          len(unique_videos), "identical file hash (MD5)"])
        writer.writerow(["3_label_consistency", len(unique_videos), len(mismatched),
                          len(labeled_ok), "folder name did not match expected labels"])

    print("Done. Check the 'logs' folder for results.")
    print(f"Final clean list ({len(labeled_ok)} videos) saved to logs/clean_video_list.csv")
    print("Hand that file to Person B for the face-detection step (Step 4).")


if __name__ == "__main__":
    main()
