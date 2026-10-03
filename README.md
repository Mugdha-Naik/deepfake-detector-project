# Deepfake Detector Project

Explainable deepfake video detection: testing whether Grad-CAM explanations from an
XceptionNet detector stay **faithful** and **stable** under video compression and
cross-dataset shift (FaceForensics++, Celeb-DF).

This README covers how to run the pipeline, how `utils.py` works, and — most
importantly — **how to avoid re-running earlier steps every session**, which cost
real time earlier in this project.

---

## 1. Project structure

```
deepfake-detector-project/
├── utils.py                      # shared functions/constants — single source of truth
├── logs/
│   └── clean_video_list.csv      # output of Steps 1-3 (video integrity, dedup, labels)
├── step4_face_extraction.ipynb   # Step 4: face detection + cropping
├── step5_labeling_quality.ipynb  # Step 5: labeling, balancing, quality filtering
├── step6_split_integrity.ipynb   # Step 6: train/test split integrity check
└── README.md
```

Each step is its **own notebook**. Do not add new steps as more cells onto an
existing notebook — this is what caused a 285-cell file earlier in the project.

---

## 2. Pipeline order (what depends on what)

```
Steps 1-3 (clean_video_list.csv)
        ↓
    Step 4 (face_extraction) ─── independent test run
        ↓
    Step 5 (labeling + quality filtering) ─── own balanced extraction, doesn't need Step 4's output
        ↓
    Step 6 (split integrity) ─── needs Step 5's saved output
        ↓
    Step 7 (class balance documentation) ─── needs Step 5/6's saved output
        ↓
    Step 8 (consolidated cleaning log) ─── needs Steps 4, 5, 6's saved output
```

Only Step 6 (and 7, 8) strictly require a *previous* step's saved output as
input. Step 4 and Step 5 each do their own extraction independently.

---

## 3. Running any notebook on Kaggle — do this every session

### 3.1 Enable GPU
Settings sidebar → Accelerator → **GPU T4 x2**. Do this *before* running
anything — changing it later requires a session restart.

### 3.2 Install dependencies — copy this exactly

```python
# ============================================================
# ENVIRONMENT SETUP — run this first, every session
# ============================================================
import torch, torchvision
print("Pre-installed torch:", torch.__version__)
print("Pre-installed torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

!pip install -q --no-deps facenet-pytorch==2.6.0
```

**Do not** reinstall `torch`/`torchvision`/`torchaudio`, and do not manually pin
`Pillow`. This was tried and caused real failures:
- Forcing an old `torchvision` broke on `Pillow >= 10.0` (`ImportError: cannot
  import name 'is_directory' from 'PIL._util'`)
- `facenet-pytorch` itself requires `Pillow >= 10.2` — so no single Pillow
  version satisfies both. `--no-deps` sidesteps this by not letting pip touch
  anything Kaggle already has installed and working.

### 3.3 Patch + verify (second cell — always run this)

```python
import os
import PIL._util as _util

if not hasattr(_util, "is_directory"):
    _util.is_directory = lambda path: os.path.isdir(path)
if not hasattr(_util, "is_path"):
    _util.is_path = lambda path: isinstance(path, (bytes, str, os.PathLike))

import torch, torchvision, PIL
import importlib.metadata

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Pillow:", PIL.__version__)
print("facenet-pytorch:", importlib.metadata.version("facenet-pytorch"))
print("CUDA available:", torch.cuda.is_available())

from facenet_pytorch import MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_test_mtcnn = MTCNN(keep_all=True, device=device)
print("facenet-pytorch import + MTCNN init: OK")
```

If this prints `OK` at the end, the environment is ready. If it errors, see
**Section 6 (Troubleshooting)** before trying random fixes.

---

## 4. How to use `utils.py`

### 4.1 Load it into any notebook

```python
import os, sys

REPO_URL = "https://github.com/Mugdha-Naik/deepfake-detector-project.git"
REPO_DIR = "/kaggle/working/repo"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull")

sys.path.append(REPO_DIR)

from utils import (
    get_kaggle_video_path, verify_path_resolution, get_label_from_path,
    load_mtcnn, extract_faces_from_video, calculate_blur_score,
    apply_quality_filters, CELEB_ROOT, FF_ROOT, CSV_URL,
)
print("utils.py loaded successfully")
```

Only import the names you actually need in that notebook — e.g. Step 6 only
needs `CELEB_ROOT, FF_ROOT`, not `load_mtcnn` or `extract_faces_from_video`.

### 4.2 What's in it

| Function / constant | Purpose |
|---|---|
| `CELEB_ROOT`, `FF_ROOT`, `CSV_URL` | Dataset paths and the canonical video-list URL — single source of truth |
| `get_kaggle_video_path(path)` | Converts a `clean_video_list.csv` path into the real Kaggle dataset path |
| `verify_path_resolution(df)` | Sanity check — run this after loading any CSV, confirms both REAL and FAKE paths actually resolve |
| `get_label_from_path(path)` | Derives REAL/FAKE/UNKNOWN from the folder path |
| `load_mtcnn(device=None)` | Loads the project's standard face detector |
| `extract_faces_from_video(...)` | Runs detection on one video, saves face crops, returns metadata |
| `calculate_blur_score(path)` | Laplacian variance — used for quality filtering |
| `apply_quality_filters(df)` | Applies blur + size thresholds, returns `(full_df, filtered_df)` |

### 4.3 Updating `utils.py`

1. Edit the function/constant in `utils.py` **once**, in this repo (GitHub web
   editor, or locally + push).
2. Commit and push to `master`.
3. Every notebook picks up the fix automatically on its next `git clone`/`git
   pull` cell — **no need to edit anything inside the individual step
   notebooks**.

This is the entire point of having a shared file: fix a bug once, not once per
notebook. If you ever find yourself redefining a function inside a step
notebook instead of editing `utils.py`, stop — fix it in `utils.py` instead,
or the notebooks will drift out of sync again.

---

## 5. How to NOT repeat earlier steps

This was the single biggest time-sink earlier in the project. The fix:

### 5.1 Always click "Save Version" when a notebook finishes

- Use **Save & Run All (Commit)**, with GPU accelerator selected.
- This re-runs the whole notebook fresh in a clean container and publishes
  everything in `/kaggle/working/` as a **permanent Kaggle Dataset**.
- Downloading the `.ipynb` file does **not** save this output — only "Save
  Version" does. A live/draft session's output disappears when the session
  ends if you skip this step.

### 5.2 Chain steps via "Add Input", not by re-running

To use an earlier step's output in a later notebook:

1. Open the later notebook (e.g. Step 6).
2. Click **+ Add Input** → find the earlier notebook's saved output (e.g.
   "Your Work" → the Step 5 notebook).
3. It mounts at `/kaggle/input/<notebook-name>/...` — **copy the exact path**
   from the Input panel (click the file, use the copy-path button). Do not
   guess the path.
4. Paste that path into the placeholder variable in the later notebook (e.g.
   `QUALITY_FILTERED_PATH = "..."`) and run from there.

Never re-run Step 4 or 5's extraction just to get to Step 6 — if their output
was saved via 5.1, Step 6 only needs to *read* that saved CSV.

### 5.3 Stopping a session is safe, once committed

Once a notebook has successfully run via "Save Version", its session can be
stopped freely — the saved output is independent of the live session. Stop a
finished notebook's session before starting a new one if Kaggle limits
concurrent sessions on your account.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'is_directory' from 'PIL._util'` | Something reinstalled `torchvision` to an old version, or the patch cell (3.3) wasn't run | Use the exact install cell in 3.2 (no torch/torchvision reinstall), make sure cell 3.3 runs before any `torchvision`/`facenet_pytorch` import |
| `ModuleNotFoundError: No module named 'facenet_pytorch'` after `from utils import ...` | A notebook imported something from `utils.py` that needs `facenet_pytorch`, but never installed it | Run the Section 3.2 install cell first, in that notebook, before importing from `utils.py` |
| `ValueError: numpy.dtype size changed...` or `NameError: name '_C' is not defined` | Session's installed packages are in an inconsistent state from multiple reinstalls | **Stop the session completely** (not just restart), start fresh, use the install cell in 3.2 exactly as written |
| `FileNotFoundError` reading a CSV from `/kaggle/input/...` | Placeholder path wasn't updated after adding the dataset | Add the dataset via "Add Input", copy the real path from the Input panel, don't guess it |
| All extracted faces are one class (e.g. all REAL) | Used `.head(N)` on an unsorted/unshuffled CSV, or `get_label_from_path` didn't match the folder structure | Use Step 5's balanced sampling (`real_df.sample(...)` + `fake_df.sample(...)`), and run `verify_path_resolution(df)` to confirm both classes resolve |

If a new error doesn't match anything above, check whether it's one of the
well-known, harmless pip dependency warnings (`bigframes`, `jax`, `cesium`,
`dopamine-rl`, etc. complaining about `numpy`/`gym` versions) — these are
unrelated to this project and can be ignored.

---

## 7. Team workflow notes

- **One canonical repo**: `Mugdha-Naik/deepfake-detector-project`. Push here
  directly or via PR — don't let work sit only in a personal fork.
- **One file per step** — if starting a new step, create a new notebook, don't
  add cells to an existing one.
- Before adding a new function used by more than one step, put it in
  `utils.py` rather than copying it into each notebook.
