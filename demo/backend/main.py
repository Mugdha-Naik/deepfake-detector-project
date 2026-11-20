"""Mini demo backend: analyses ONE pre-cropped face image (CPU). Frozen settings, do not change.
Run from demo/backend with:  uvicorn main:app --port 8000
Needs: best.pt in this folder (or set DFX_CKPT), and dfx_model.py from the repo root
(found automatically at ../../dfx_model.py, or copy it next to this file)."""
import io, os, sys, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))   # repo root
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import dfx_model

THRESHOLD = 0.2217                     # frozen (Youden's J on FF++ validation)
MEAN = STD = (0.5, 0.5, 0.5)
SIZE, GRID, K, R = 299, 10, 20, 5      # faithfulness: top 20% cells, 5 random masks, zero fill
CKPT = os.environ.get("DFX_CKPT", os.path.join(os.path.dirname(__file__), "best.pt"))

torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
model, info = dfx_model.build_xception(pretrained=False)   # no download; weights come from best.pt
dfx_model.load_checkpoint(model, CKPT, map_location="cpu")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
MODEL_SHA = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()[:8]

app = FastAPI(title="Deepfake XAI demo (face-image only)")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

def preprocess(im):
    im = im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)      # torchvision Resize default
    x = torch.from_numpy(np.asarray(im).astype("float32") / 255.0).permute(2, 0, 1)
    m = torch.tensor(MEAN).view(3, 1, 1); s = torch.tensor(STD).view(3, 1, 1)
    return ((x - m) / s).unsqueeze(0)

def gradcam(x):
    """Same as Step 2.6: features -> head, target = predicted class (+logit FAKE, -logit REAL)."""
    with torch.no_grad():
        feats = model.forward_features(x)
    feats = feats.detach().requires_grad_(True)
    logit = model.forward_head(feats).reshape(-1)
    prob = float(torch.sigmoid(logit.detach())[0])
    tgt = 1.0 if prob >= THRESHOLD else -1.0
    grad = torch.autograd.grad(tgt * logit.sum(), feats)[0]
    w = grad.mean(dim=(2, 3))
    cam = torch.relu((w[:, :, None, None] * feats.detach()).sum(1))[0]
    mx = float(cam.max())
    cam = cam / mx if mx > 1e-12 else torch.zeros_like(cam)
    return prob, tgt, cam

@torch.no_grad()
def p_pred(x, tgt):
    p = torch.sigmoid(model(x).reshape(-1))
    return p if tgt > 0 else 1 - p

def mask_from(sel):   # sel: bool (100,) -> (1,1,299,299), nearest upsampling as in Step 2.7
    return F.interpolate(sel.view(1, 1, GRID, GRID).float(), size=(SIZE, SIZE), mode="nearest")

def faithfulness(x, cam, tgt):
    n = int(round(GRID * GRID * K / 100))
    p0 = p_pred(x, tgt)
    top = torch.zeros(GRID * GRID, dtype=torch.bool)
    top[torch.argsort(cam.reshape(-1), descending=True, stable=True)[:n]] = True
    d_cam = p0 - p_pred(x * (1 - mask_from(top)), tgt)               # zero fill = mid-grey
    g = torch.Generator().manual_seed(0)
    d_rand = []
    for _ in range(R):
        r = torch.zeros(GRID * GRID, dtype=torch.bool)
        r[torch.randperm(GRID * GRID, generator=g)[:n]] = True
        d_rand.append(p0 - p_pred(x * (1 - mask_from(r)), tgt))
    return float(d_cam - torch.stack(d_rand).mean())

@app.get("/health")
def health():
    return {"status": "ok", "model": "XceptionNet (timm %s), trained on FF++ only" % info["timm_name"],
            "model_sha8": MODEL_SHA, "threshold": THRESHOLD, "device": "cpu"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), with_faithfulness: bool = True):
    try:
        im = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Not a valid image")
    x = preprocess(im)
    prob, tgt, cam = gradcam(x)
    out = {"p_fake": prob, "threshold": THRESHOLD, "verdict": "FAKE" if tgt > 0 else "REAL",
           "explained_class": "FAKE" if tgt > 0 else "REAL",
           "heatmap_10x10": cam.numpy().round(4).tolist(),
           "faithfulness_k20": faithfulness(x, cam, tgt) if with_faithfulness else None,
           "stability": None, "reliability": None,
           "note": "Stability and Trusted/Flagged need a clean/compressed twin: shown only for stored examples."}
    return out
