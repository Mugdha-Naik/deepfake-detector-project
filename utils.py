"""
utils.py — Shared pipeline utilities for the deepfake detection project.

WHY THIS FILE EXISTS
---------------------
Across the Step 4, Step 5, and Step 6 notebooks, these functions were each
redefined multiple times with SLIGHTLY different, inconsistent logic
(different path-matching rules, different frame-sampling strategies,
different MTCNN settings). That inconsistency caused real bugs — e.g. one
version of get_kaggle_video_path silently failed to match FF++ paths.

This file is the ONE canonical source. Every notebook should import from
here instead of re-defining these functions locally. See USAGE at the
bottom of this file for how to load it in a Kaggle notebook.

Before relying on this fully, the team should confirm the two flagged
decisions marked "TEAM DECISION" below — I consolidated the most complete
version I could find in your notebooks, but a couple of choices were
inconsistent across versions and need a deliberate pick, not a guess.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN


# ============================================================
# CONSTANTS — single source of truth
# ============================================================

CELEB_ROOT = "/kaggle/input/datasets/reubensuju/celeb-df-v2"
FF_ROOT = "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23"

# Canonical repo confirmed: Mugdha-Naik/deepfake-detector-project
# (Srishti and Nancy's forks were used for individual commits, but PRs
# have been merged back into this repo — this is the single source of truth.)
CSV_URL = "https://raw.githubusercontent.com/Mugdha-Naik/deepfake-detector-project/master/logs/clean_video_list.csv"

MIN_CONFIDENCE = 0.90
FRAME_INTERVAL = 10          # used by the "interval" sampling strategy
MAX_FRAMES_PER_VIDEO = 20    # used by the "evenly_spaced" sampling strategy

BLUR_THRESHOLD = 50
MIN_FACE_WIDTH = 64
MIN_FACE_HEIGHT = 64


# ============================================================
# PATH RESOLUTION
# Consolidated from the Step 4 notebook's version, which was the
# most precise and confirmed-working prefix match. (Two later notebooks
# introduced a looser "/cel/"/"/ff/" substring check that likely failed
# to match FF++ paths at all — do not revert to that version.)
# ============================================================

def get_kaggle_video_path(relative_path):
    """Convert a path from clean_video_list.csv into the matching Kaggle dataset path."""
    path = str(relative_path).replace("\\", "/")

    if "data/raw/cel/" in path:
        relative = path.split("data/raw/cel/", 1)[1]
        return os.path.join(CELEB_ROOT, relative)

    elif "data/raw/FaceForensics++_C23/" in path:
        relative = path.split("data/raw/FaceForensics++_C23/", 1)[1]
        return os.path.join(FF_ROOT, relative)

    return None


def verify_path_resolution(df, video_path_col="video_path", n=2):
    """
    Sanity check: confirm get_kaggle_video_path resolves BOTH real (Celeb-DF)
    and fake (FF++) rows correctly before running a full extraction.
    Run this after loading any new CSV, every session.
    """
    sample_celeb = df[df[video_path_col].str.contains("cel", case=False, na=False)].head(n)
    sample_ff = df[df[video_path_col].str.contains("FaceForensics", case=False, na=False)].head(n)

    for _, row in pd.concat([sample_celeb, sample_ff]).iterrows():
        p = get_kaggle_video_path(row[video_path_col])
        exists = os.path.exists(p) if p else False
        print(row[video_path_col], "->", p, "| exists:", exists)
        if not exists:
            print("  WARNING: path did not resolve — check dataset roots and CSV format")


# ============================================================
# LABEL DERIVATION
# Consolidated from the most complete version (Step 6 notebook, which
# added "deepfakedetection" folder handling that earlier versions missed).
# ============================================================

def get_label_from_path(path):
    """Derive REAL/FAKE/UNKNOWN from the source folder path. Never silently guesses."""
    path = str(path).lower().replace("\\", "/")

    if "/celeb-real/" in path or "/youtube-real/" in path or "/original/" in path:
        return "REAL"

    fake_folders = [
        "/celeb-synthesis/", "/deepfakes/", "/faceswap/",
        "/faceshifter/", "/face2face/", "/neuraltextures/", "/deepfakedetection/",
    ]
    for folder in fake_folders:
        if folder in path:
            return "FAKE"

    return "UNKNOWN"  # flag and check manually — never assume real or fake


# ============================================================
# FACE DETECTOR
# TEAM DECISION: notebooks used min_face_size=40 in one place and
# min_face_size=20 in later ones, with no documented reason for the
# change. This consolidates the LATER (min_face_size=20) version since
# it was used in the most recent, most-debugged notebooks — but confirm
# this was an intentional choice, not just drift.
# ============================================================

def load_mtcnn(device=None):
    """Load the project's standard MTCNN face detector."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False,
    )
    print("MTCNN loaded. Device:", device)
    return mtcnn, device


# ============================================================
# FACE EXTRACTION
# Two sampling strategies existed across notebooks with no clear team
# decision on which is canonical. Both are kept here, explicitly named,
# so the choice is conscious rather than accidental drift.
#   "interval"      -> every Nth frame (used in Step 4 / Step 5 notebooks)
#   "evenly_spaced"  -> K frames evenly spread across the video (used in
#                        the Step 6 notebook's redo of Step 4)
# TEAM DECISION: pick one as the project standard and note it in your
# methodology write-up — mixing strategies across conditions would
# itself be a confound in your faithfulness/stability comparisons.
# ============================================================

def extract_faces_from_video(video_path, label_name, faces_dir, video_id,
                              mtcnn, sampling="interval",
                              frame_interval=FRAME_INTERVAL,
                              max_frames=MAX_FRAMES_PER_VIDEO,
                              min_confidence=MIN_CONFIDENCE):
    """
    Run face detection on one video, save the largest confident face per
    sampled frame, and return a list of metadata dicts (one per saved face).
    """
    records = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open:", video_path)
        return records

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if sampling == "evenly_spaced":
        if total_frames <= 0:
            cap.release()
            return records
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    elif sampling == "interval":
        frame_indices = range(0, total_frames, frame_interval) if total_frames > 0 else []
    else:
        raise ValueError("sampling must be 'interval' or 'evenly_spaced'")

    for frame_number in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        boxes, probabilities = mtcnn.detect(image)

        if boxes is None or probabilities is None:
            continue

        valid_faces = []
        for box, confidence in zip(boxes, probabilities):
            if confidence is None or confidence < min_confidence:
                continue
            x1, y1, x2, y2 = box
            area = max(0, x2 - x1) * max(0, y2 - y1)
            valid_faces.append((area, box, confidence))

        if not valid_faces:
            continue

        valid_faces.sort(key=lambda x: x[0], reverse=True)
        area, box, confidence = valid_faces[0]
        x1, y1, x2, y2 = box

        h, w = rgb_frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        face = rgb_frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_filename = f"{video_id}_frame_{int(frame_number):06d}.jpg"
        face_path = os.path.join(faces_dir, face_filename)
        Image.fromarray(face).save(face_path, quality=95)

        records.append({
            "video_path": video_path,
            "label_name": label_name,
            "frame_number": int(frame_number),
            "face_path": face_path,
            "confidence": float(confidence),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "face_width": x2 - x1, "face_height": y2 - y1,
        })

    cap.release()
    return records


# ============================================================
# QUALITY FILTERING
# ============================================================

def calculate_blur_score(image_path):
    """Laplacian variance — higher means sharper. Returns NaN if unreadable."""
    if not os.path.exists(image_path):
        return np.nan
    image = cv2.imread(image_path)
    if image is None:
        return np.nan
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def apply_quality_filters(df, blur_threshold=BLUR_THRESHOLD,
                           min_width=MIN_FACE_WIDTH, min_height=MIN_FACE_HEIGHT):
    """
    Adds blur_score, blur_pass, size_pass, quality_pass columns to df
    (must already have face_path, face_width, face_height columns).
    Returns (full_df_with_flags, filtered_df_passing_only).
    """
    df = df.copy()
    df["blur_score"] = df["face_path"].apply(calculate_blur_score)
    df["blur_pass"] = df["blur_score"] >= blur_threshold
    df["size_pass"] = (df["face_width"] >= min_width) & (df["face_height"] >= min_height)
    df["quality_pass"] = df["blur_pass"] & df["size_pass"]

    filtered_df = df[df["quality_pass"]].copy()
    return df, filtered_df


# ============================================================
# USAGE (copy this into the top of each Kaggle notebook)
# ============================================================
#
# !pip install -q torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# !pip install -q facenet-pytorch==2.6.0
# !pip install -q --force-reinstall --no-cache-dir "Pillow==9.5.0"
#
# NOTE ON VERSION: Pillow 10.0+ removed is_directory/is_path from
# PIL._util, which torchvision 0.17.2's ImageFont import needs. Pillow
# 10.2.0 (tried earlier) is PAST that removal and still fails — 9.5.0 is
# confirmed to predate it. If this notebook is run via "Save & Run All
# (Commit)", the whole thing executes fresh in ONE process — there is no
# stale import to fix by restarting; if you see the is_directory
# ImportError there, it is a real version mismatch, not a caching issue.
# (For interactive/live sessions specifically — not commits — restarting
# the kernel after a package change is still good practice, but it will
# NOT fix a genuinely incompatible version pin like Pillow 10.2.0 was.)
#
# !git clone https://github.com/Mugdha-Naik/deepfake-detector-project.git /kaggle/working/repo
# import sys
# sys.path.append("/kaggle/working/repo")
# from utils import (
#     get_kaggle_video_path, verify_path_resolution, get_label_from_path,
#     load_mtcnn, extract_faces_from_video, calculate_blur_score,
#     apply_quality_filters, CELEB_ROOT, FF_ROOT, CSV_URL,
# )
#
# Put this file at the repo root (or in a clearly named folder like
# `pipeline/utils.py` — just update the sys.path/import line to match).

