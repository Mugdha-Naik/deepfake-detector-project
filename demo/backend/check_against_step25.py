"""Usage: python check_against_step25.py face1.jpg 0.8123 face2.jpg 0.0412 ...
Pairs of (cropped face file, P(FAKE) from predictions_A.csv column prob_fake for the same face).
Server must be running on port 8000. Tolerance 0.01 (Step 2.5 used fp16 autocast, this is fp32)."""
import sys, requests
a = sys.argv[1:]; ok = True
for img, exp in zip(a[::2], a[1::2]):
    r = requests.post("http://localhost:8000/analyze?with_faithfulness=false", files={"file": open(img, "rb")}).json()
    d = abs(r["p_fake"] - float(exp)); ok &= d < 0.01
    print(f"{img}: backend {r['p_fake']:.4f} vs step2.5 {float(exp):.4f}  diff {d:.4f}  verdict {r['verdict']}")
print("MATCH: safe to use" if ok else "MISMATCH: check RGB order, normalisation, best.pt; otherwise use the static demo")
