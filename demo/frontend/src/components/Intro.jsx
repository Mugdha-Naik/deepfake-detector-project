import { useEffect, useState } from "react";

const lines = [
  "Initializing neural engine...",
  "Loading detection model...",
  "Calibrating face analyzer...",
  "System ready ✓",
];

export default function Intro({ onDone }) {
  const [p, setP] = useState(0);
  const [out, setOut] = useState(false);

  useEffect(() => {
    const t = setInterval(() => setP((v) => Math.min(v + 1, 100)), 28);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (p >= 100) {
      setOut(true);
      const t = setTimeout(onDone, 700);
      return () => clearTimeout(t);
    }
  }, [p]);

  const line = lines[Math.min(Math.floor(p / 26), 3)];

  return (
    <div className={"intro " + (out ? "out" : "")} onClick={onDone}>
      <div className="intro-scan" />
      <div className="intro-core">
        <div className="hex"><i /><i /><i /><span>◈</span></div>
        <h1 className="intro-title">
          {"DeepGuard".split("").map((c, i) => (
            <span key={i} style={{ animationDelay: i * 70 + "ms" }}>{c}</span>
          ))}
          <b>AI</b>
        </h1>
        <p className="intro-sub">Deepfake Detection System</p>
        <div className="intro-bar"><div style={{ width: p + "%" }} /></div>
        <div className="intro-line">{line} <span>{p}%</span></div>
      </div>
    </div>
  );
}