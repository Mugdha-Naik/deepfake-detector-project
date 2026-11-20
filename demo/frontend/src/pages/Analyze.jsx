import { useEffect, useState } from "react";
import Upload from "../components/Upload";

function Typing({ text, speed = 28 }) {
  const [n, setN] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setN((v) => (v < text.length ? v + 1 : v)), speed);
    return () => clearInterval(t);
  }, [text, speed]);
  return <>{text.slice(0, n)}<span className="cursor">|</span></>;
}

export default function Analyze() {
  return (
    <div className="page">
      <span className="chip">AI • Cybersecurity Research</span>
      <h1>Detect <em>Deepfakes</em> in Seconds</h1>
      <p className="sub">
        <Typing text="Upload media and our neural pipeline will analyse facial artifacts, temporal inconsistencies and compression traces." />
      </p>
      <Upload />
      <div className="grid3">
        {[["01", "Upload", "Select a video or image"], ["02", "Analyse", "Frame-level feature extraction"], ["03", "Verdict", "Confidence score + heatmap"]].map(([n, t, d]) => (
          <div className="glass step" key={n}><b>{n}</b><h3>{t}</h3><p>{d}</p></div>
        ))}
      </div>
      <div className="warn">⚠ Research prototype — results are probabilistic and not legal evidence.</div>
    </div>
  );
}