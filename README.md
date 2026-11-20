# Deepfake Detector Project
 
**Faithful and Reliable Explainable AI for Deepfake Detection**
Silicon Institute of Technology, Sambalpur, Dept. of CSE. Supervisor: Mrs. Tapaswini Dash.
Team: Mugdha Naik (lead, core ML), Srishti Kumari, Nancy Kumari/Kashyap, MD. Erbaz Khan.
 
**Research question.** How faithful and stable are Grad-CAM explanations of an XceptionNet
deepfake detector under compression and cross-dataset shift (FaceForensics++ to Celeb-DF)?
Secondary: do true/false positives and negatives (TP / FP / FN) show different explanation patterns?
 
**Scope.** XceptionNet only; Grad-CAM is the primary explanation method; SHAP is an optional small check.
Future work: temporal transformer, adversarial testing, multi-architecture comparison.
 
This README explains the project status, how to run each notebook on Kaggle, the rules that keep the
results valid, and how to avoid the Kaggle problems that cost the most time.
 
---
 
## 1. Status at a glance
 
| Part | What | Status |
|---|---|---|
| Phase 1 (Steps 1 to 8) | Data cleaning, official splits, face extraction, class balance, cleaning log | Done |
| Phase 2, Step 2.1 | Frozen manifest and 17 leakage/integrity checks | Done |
| 2.2 | Compressed conditions B and D | Done |
| 2.3 / 2.4 | Loader, model, training | Done |
| 2.5 | Predictions and metrics, conditions A to D | Done |
| 2.6 | Grad-CAM maps for A to D | Done, all checks passed |
| 2.7 | Faithfulness test | Notebook written, being run |
| 2.8 | Stability test | Planned |
| 2.9 | TP / FP / FN outcome analysis | Planned |
| 2.10 | Reliability rule (Trusted / Flagged), optional SHAP | Planned |
| 2.11 | Consolidation, reproducibility, tag `phase2-v1` | Planned |
| Phase 3 | Report, slides, defense (and optional demo) | Planned |
 
**No faithfulness or stability results exist yet.** Do not quote any such number until the matching
step's output file exists in the repo.
 
---
 
## 2. The four experimental conditions
 
The detector is trained only on clean FF++ and never sees compression or Celeb-DF during training.
 
| Cond | Test data | Compression | Stability pairing |
|---|---|---|---|
| A | FF++ official test (clean) | native (C23 mirror) | reference |
| B | the same FF++ test videos, re-encoded (CRF 40) | high | paired with A |
| C | Celeb-DF official test list sample (clean) | native | not pairable with A (distribution level only) |
| D | the same Celeb-DF videos, re-encoded (CRF 40) | high | paired with C |
 
B and D are a **double-compression approximation**, because the FF++ mirror is already C23. State this
as a limitation wherever B and D are reported.
 
---
 
## 3. Frozen decisions (do not change)
 
These were fixed before looking at test results. Changing any of them invalidates every later number.
 
| Item | Value |
|---|---|
| Decision threshold | **0.2217** (Youden's J on the FF++ validation split, chosen once) |
| Labels | REAL = 0, FAKE = 1; FAKE is the positive class; output = P(FAKE) |
| Manifest | `faces_manifest.csv`, SHA-256 starts with `80eae250` |
| Model | timm `legacy_xception`, ImageNet weights, one output, 299 x 299, mean/std 0.5, best epoch 8 |
| Augmentation | horizontal flip only (no compression or blur in training) |
| Grad-CAM target | the model's predicted class at the frozen threshold |
| Faithfulness | drop in probability of the predicted class when the top k% Grad-CAM cells are masked, minus the drop for random cells of equal area (k = 20 primary; 10 and 30 sensitivity) |
 
**Never retrain, change the threshold, or regenerate the manifest after seeing test results.**
Every notebook from 2.5 onward checks the checkpoint hash, manifest hash and threshold and stops if they differ.
 
---
 
## 4. Results so far
 
### Data (Phase 1)
 
13,529 videos checked (0 removed). Official splits: FF++ 720 / 140 / 140 by video ID; Celeb-DF official
test list only (518 videos). Sampled 960 videos; final 28,184 faces from 951 videos (REAL 14,055,
FAKE 14,129). Blur is kept as a flag, not a filter.
 
### Detector (Step 2.5; face level, frozen threshold, 95% CI by resampling videos)
 
| Cond | Accuracy | ROC-AUC [95% CI] | Recall | Specificity | F1 |
|---|---|---|---|---|---|
| A FF++ clean | 0.839 | 0.917 [0.882, 0.946] | 0.789 | 0.890 | 0.833 |
| B FF++ compressed | 0.527 | 0.611 [0.556, 0.666] | 0.135 | 0.935 | 0.226 |
| C Celeb-DF clean | 0.646 | 0.704 [0.661, 0.746] | 0.823 | 0.470 | 0.699 |
| D Celeb-DF compressed | 0.538 | 0.577 [0.532, 0.623] | 0.260 | 0.815 | 0.360 |
 
- Video-level AUC on A: 0.937. Validation AUC (epoch 8): 0.9735 (video level 0.9922).
- Blur-only baseline AUC: A 0.580, B 0.568, C 0.626, D 0.611. The detector beats it on A and C, not on B or D.
- Clean/compressed twins change their prediction in 41.34% (A to B) and 53.47% (C to D) of pairs, mostly FAKE to REAL.
- These are findings, not bugs: CRF 40 on top of C23 is very strong; the FF++ threshold does not transfer to Celeb-DF.
### Grad-CAM (Step 2.6)
 
- Maps for every face: A 5,883, B 5,883, C 8,763, D 8,763 (10 x 10 resolution, from `model.act4`).
- For this network (global average pooling plus one linear layer) Grad-CAM equals classic CAM. Numerical check passed (relative error 1.42e-07). Maps are coarse and the channel weights do not depend on the image.
- Model-randomisation sanity test: mean correlation 0.095 (limit 0.5).
- fp32 predictions agree with Step 2.5 on 99.95% / 99.86% / 99.75% / 99.75% of faces (A to D); disagreements are faces right at the threshold (fp16 versus fp32).
---
 
## 5. Repository structure
 
```
deepfake-detector-project/
├── utils.py              # data-pipeline helpers (extraction, quality, split, log helpers)
├── dfx_model.py          # model constants and builders (REAL=0, FAKE=1)
├── dfx_data.py           # dataset class and loaders
├── logs/                 # Steps 1-3 outputs (clean_video_list.csv, cleaning_log.csv)
├── step*.ipynb           # Phase 1 notebooks (one per step)
├── p2_*.ipynb            # Phase 2 notebooks (one per step; see section 6)
├── data/
│   ├── phase1/           # step5, step6b, step7, cleaning (small CSVs, charts)
│   └── phase2/
│       ├── manifest/     # faces_manifest.csv, manifest_meta.json, split_counts.csv
│       ├── conditions/   # faces_manifest_B/D.csv, conditions_meta.json, encode_stats.csv, pair_samples.png
│       ├── model/        # config.json, threshold.json, training_log.csv, training_curves.png
│       ├── eval/         # Step 2.5 metrics, flips, blur baseline, ROC/confusion plots
│       ├── gradcam/      # Step 2.6 checks, summary, figures
│       └── faithfulness/ # Step 2.7 summaries and figures
├── docs/
├── frontend/             # optional demo UI (see section 10)
└── README.md
```
 
**Not committed (too large):** `best.pt`, `last.pt`, face crops, compressed crops, videos, `cams_*.npz`.
Keep these as Kaggle datasets or on a shared drive, and make sure at least two team members hold a copy.
 
Module names use the `dfx_` prefix because a root-level `datasets.py` is shadowed by the Hugging Face
`datasets` package. Keep `utils.py` for the data pipeline only; model and explanation code lives in the
`dfx_*` modules and the notebooks.
 
Each step is its **own notebook**. Never add a new step as extra cells in an old notebook.
Delete Kaggle's boilerplate first cell before committing a notebook.
 
---
 
## 6. Pipeline and notebook run order
 
```
Phase 1:  Steps 1-3 (clean_video_list.csv) -> Step 4 -> Step 5 (scaled, split-aware) -> Step 6 (splits)
          -> Step 7 (class balance) -> Step 8 (cleaning log)
 
Phase 2:  2.1 manifest -> 2.2 compressed B/D -> 2.3/2.4 model + training -> 2.5 predictions + metrics
          -> 2.6 Grad-CAM -> 2.7 faithfulness -> 2.8 stability -> 2.9 outcomes -> 2.10 reliability -> 2.11 report
```
 
| Step | Notebook (Kaggle name / repo file) | Main inputs | Main outputs |
|---|---|---|---|
| 2.1 | manifest notebook (use the repo file name) | Step 5 output, `video_split_map.csv` | `faces_manifest.csv`, `manifest_meta.json` |
| 2.2 | `phase2_step2` | manifest, test videos | `faces_manifest_B/D.csv`, compressed crops |
| 2.3 / 2.4 | `p2_4_train.ipynb` (or `p2_3_4_model_and_train.ipynb`) | manifest, clean crops | `best.pt`, `config.json`, `threshold.json`, training log |
| 2.5 | `p2_5_predictions_metrics.ipynb` (Kaggle `phase2-step5`) | model, manifests, crops | `predictions_A..D.csv`, `metrics_*.csv`, plots |
| 2.6 | `p2_6_gradcam.ipynb` (Kaggle `phase2-step6`) | model, manifests, crops, 2.5 output | `cams_A..D.npz`, checks, figures |
| 2.7 | `p2_7_faithfulness.ipynb` | everything above + 2.6 output | `faithfulness_A..D.csv`, summaries, figures |
 
Each notebook reads the previous step's output through **Add Input** and stops with a clear message if a
hash or file is wrong. Add a row here when you add a notebook.
 
---
 
## 7. Running any notebook on Kaggle
 
### 7.1 Session setup
- Settings sidebar: Accelerator **GPU T4**, Internet **On**. Set this before running anything.
- Use **Save & Run All (Commit)** to save a finished notebook. Never use Quick Save on a notebook that other notebooks use as an input (the latest version is what mounts).
- Inputs mount only at session start. Stop and restart the session after changing inputs.
- Attach each file **once**. Duplicate copies of `faces_manifest.csv`, `manifest_meta.json` or `video_split_map.csv` make the notebooks stop with a "2 copies" error. The `phase2_step2` output also contains a `repo` folder; ignore it if a duplicate-file error mentions it.
### 7.2 Install dependencies
 
```python
import torch, torchvision
print("Pre-installed torch:", torch.__version__, "| torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
 
!pip install -q --no-deps facenet-pytorch==2.6.0
# if timm is missing:  !pip install -q --no-deps timm huggingface_hub safetensors
```
 
**Never reinstall `torch`, `torchvision` or `torchaudio`, and do not pin Pillow.** Install everything else
with `--no-deps`. Forcing old versions broke earlier runs (`ImportError: cannot import name 'is_directory'
from 'PIL._util'`), and `facenet-pytorch` needs a newer Pillow than old torchvision accepts. `--no-deps`
avoids touching what Kaggle already has working.
 
### 7.3 Patch and verify (data-extraction notebooks)
 
```python
import os
import PIL._util as _util
if not hasattr(_util, "is_directory"):
    _util.is_directory = lambda path: os.path.isdir(path)
if not hasattr(_util, "is_path"):
    _util.is_path = lambda path: isinstance(path, (bytes, str, os.PathLike))
 
import torch, torchvision, PIL, importlib.metadata
from facenet_pytorch import MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_test = MTCNN(keep_all=True, device=device)
print("facenet-pytorch import + MTCNN init: OK")
```
 
### 7.4 Load the shared code
 
```python
import os, sys
REPO_URL = "https://github.com/Mugdha-Naik/deepfake-detector-project.git"
REPO_DIR = "/kaggle/working/repo"
if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
else:
    os.system(f"cd {REPO_DIR} && git pull")
sys.path.append(REPO_DIR)
 
import dfx_model, dfx_data            # Phase 2 notebooks
# from utils import ...               # only the names the notebook needs (Phase 1)
```
 
Fix a function **once**, in the repo, push to `master`, and every notebook picks it up on its next clone/pull.
 
---
 
## 8. Chaining steps and sharing large files
 
1. Finish a notebook with **Save & Run All**. This publishes `/kaggle/working/` as the notebook's saved output. Downloading the `.ipynb` does not save the output.
2. In the next notebook click **+ Add Input**, choose the earlier notebook or dataset, and **copy the exact mounted path** from the Input panel. Do not guess paths.
3. Never re-run an earlier extraction just to reach a later step. Read the saved output instead.
**Crops and Kaggle mounting.** Sessions that mounted notebooks holding tens of thousands of small images
sometimes stayed on "Adding data..." for over an hour or failed with `ERRORED_MOUNTING_DATASET`.
The reliable fix is to zip the crops into single files (`step5_faces.zip`, `compressed_faces.zip`),
publish each as a Kaggle dataset, and attach those. The Phase 2 notebooks look for a folder named
`step5_faces`, `faces_B` or `faces_D`, or unzip the zip automatically. If Kaggle already extracted the zip
when you published the dataset, the folders are found directly.
 
---
 
## 9. Troubleshooting
 
| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'is_directory' from 'PIL._util'` | torchvision was reinstalled, or the patch cell did not run | Use the install cell in 7.2 exactly; run the patch in 7.3 before importing torchvision/facenet_pytorch |
| `ModuleNotFoundError: No module named 'facenet_pytorch'` | `utils.py` imported before the install cell ran | Run the install cell first |
| `ValueError: numpy.dtype size changed` or `NameError: name '_C' is not defined` | Inconsistent packages from repeated reinstalls | Stop the session completely, start fresh, use 7.2 exactly |
| `FileNotFoundError` reading from `/kaggle/input/...` | Placeholder path not updated or input not attached | Add Input, copy the real path from the Input panel |
| `RuntimeError: ... 2 copies ...` | The same file is attached twice | Remove the duplicate input and restart the session |
| Assertion: manifest / checkpoint / threshold mismatch | A different manifest, model or threshold is attached | Attach exactly the frozen files (section 3); do not edit the check |
| `AssertionError: Set the Accelerator to GPU T4` | Notebook is on CPU | Change the accelerator and restart |
| `NameError: name 'loader_for' is not defined` (or another helper) | A defining cell was skipped or the session restarted | Run all cells from the top in order |
| Session stuck on "Adding data..." or `ERRORED_MOUNTING_DATASET` | Too many small files mounted from a notebook | Use zipped dataset inputs (section 8); stop the session; Stop is greyed out until the session starts |
| Output page shows `faces: empty` | Kaggle display glitch | Expand the folder; the files are there |
| Red "Firebase auth/internal-error" banner after Save and Run All | Harmless | The run still starts; do not click again |
| All extracted faces are one class | `.head(N)` on an unshuffled CSV, or label mismatch | Use balanced sampling and run `verify_path_resolution(df)` |
 
Harmless pip dependency warnings (`bigframes`, `jax`, `cesium`, `dopamine-rl` and similar) can be ignored.
 
---
 
## 10. Optional demo (frontend and backend)
 
The research does not need a UI. A demo UI (React + Vite, in `frontend/`) currently reads **dummy data**.
Before showing it:
- Label the model "XceptionNet (FF++ trained)" and remove any LSTM or other-architecture claims.
- Replace invented dashboard numbers with the real A to D table above, and show faithfulness and stability as "Pending" until Steps 2.7 and 2.8 are final.
- Show the threshold (0.2217) and the note "Research demo. Trained on FaceForensics++ only; results on other images can be unreliable."
- A live demo needs a backend (Python, FastAPI): MTCNN, the model, Grad-CAM from `model.act4`, ffmpeg for compression. Check it reproduces the Step 2.5 and 2.6 outputs for the same faces before connecting the frontend.
- The recommended first version is a **static demo** built from real exported examples (no backend or GPU needed).
---
 
## 11. Limitations to state in any report
 
One architecture; coarse 10 x 10 Grad-CAM (equal to classic CAM for this network); masking artefacts in the
faithfulness test; approximate (double) compression; small FP / FN groups; simple Step 1 to 3 integrity
checks; test AUC on A below validation AUC; detector near chance under CRF 40; FF++ threshold over-flags
real Celeb-DF faces (specificity 0.47).
 
---
 
## 12. Team workflow
 
- One canonical repo: `Mugdha-Naik/deepfake-detector-project`. Work on a branch and open a pull request; do not leave work only in a personal fork or only in Kaggle.
- One notebook per step. Shared functions go in `utils.py` (data pipeline) or the `dfx_*` modules (model and explanation code), never copied between notebooks.
- Commit titles follow the step, for example `Phase 2 step 2.5: predictions and metrics for conditions A to D`.
- Tag milestones: `data-v1`, `data-v1-conditions`, `phase2-v1`.
- Keep a numbers sheet (`numbers_sheet.md`) so the report, slides and script quote the same values from the same files.
