import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import ProgressBar from "../components/ProgressBar";

const stages = ["Extracting frames", "Detecting faces", "Running neural model", "Generating heatmap", "Compiling report"];
const logs = [
  "[init]  Loading model weights...",
  "[video] Extracting frames @ 5 fps",
  "[face]  Face detector started",
  "[face]  Faces cropped and aligned",
  "[model] Running forward pass...",
  "[model] Analysing temporal sequence",
  "[xai]   Generating attention heatmap",
  "[done]  Report ready ✓",
];

export default function Processing() {
  const nav = useNavigate();
  const { state } = useLocation();
  const [p, setP] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setP((v) => Math.min(v + 2, 100)), 90);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    if (p >= 100) { const t = setTimeout(() => nav("/result", { state }), 800); return () => clearTimeout(t); }
  }, [p]);
  const stage = Math.min(Math.floor(p / 20), 4);
  const shown = logs.filter((_, i) => p >= i * 12);
  return (
    <div className="page">
      <span className="chip">Processing</span>
      <h1>Analysing <em>{state?.name || "media"}</em></h1>
      <div className="glass center">
        <div className="scanner"><div className="scan-line" /></div>
        <h2 className="big">{p}%</h2>
        <ProgressBar value={p} />
        <div className="stages">
          {stages.map((s, i) => (
            <div key={s} className={"stage " + (i < stage ? "done" : i === stage ? "active" : "")}>
              <i>{i < stage ? "✓" : i + 1}</i>{s}
            </div>
          ))}
        </div>
        <div className="terminal">
          {shown.map((l) => <div key={l}><span>$</span> {l}</div>)}
          <div className="cursor">▋</div>
        </div>
      </div>
    </div>
  );
}