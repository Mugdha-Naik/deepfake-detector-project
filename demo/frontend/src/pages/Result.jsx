import { useLocation, Link } from "react-router-dom";
import Heatmap from "../components/Heatmap";

export default function Result() {
  const { state } = useLocation();
  const score = 94.3, r = 70, c = 2 * Math.PI * r;
  return (
    <div className="page">
      <span className="chip red">Analysis Complete</span>
      <h1>Verdict: <em className="danger glitch">Likely Deepfake</em></h1>
      <p className="sub">File: {state?.name || "sample_video.mp4"}</p>
      <div className="grid2">
        <div className="glass center">
          <svg width="190" height="190" viewBox="0 0 190 190">
            <defs>
              <linearGradient id="g">
                <stop offset="0" style={{ stopColor: "var(--a)" }} />
                <stop offset="1" style={{ stopColor: "var(--b)" }} />
              </linearGradient>
            </defs>
            <circle cx="95" cy="95" r={r} className="ring-bg" />
            <circle cx="95" cy="95" r={r} className="ring" style={{ "--c": c, "--o": c * (1 - score / 100) }} />
          </svg>
          <h2 className="big ring-text">{score}%</h2>
          <p>Fake Confidence</p>
        </div>
        <div className="glass">
          <h3>Attention Heatmap</h3>
          <Heatmap />
          <p className="note">Warm regions show where the model detected manipulation.</p>
        </div>
      </div>
      <div className="grid3">
        {[["Face Blending", "High"], ["Temporal Flicker", "Medium"], ["Frames Scanned", "240"]].map(([a, b]) => (
          <div className="glass step" key={a}><p>{a}</p><h2>{b}</h2></div>
        ))}
      </div>
      <Link to="/" className="btn inline">↻ Analyze Another</Link>
    </div>
  );
}