import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
export default function Upload() {
  const nav = useNavigate();
  const ref = useRef();
  const [file, setFile] = useState(null);
  const [drag, setDrag] = useState(false);
  const pick = (f) => f && setFile(f);
  return (
    <div className="glass upload">
      <div
        className={"dropzone " + (drag ? "drag" : "")}
        onClick={() => ref.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
      >
        <div className="pulse-ring"><span>⬆</span></div>
        <h3>{file ? file.name : "Drag & drop your video / image here"}</h3>
        <p>{file ? (file.size / 1048576).toFixed(2) + " MB ready" : "or click to browse • MP4, AVI, JPG, PNG"}</p>
        <input ref={ref} type="file" hidden accept="video/*,image/*" onChange={(e) => pick(e.target.files[0])} />
      </div>
      <button className="btn" disabled={!file} onClick={() => nav("/processing", { state: { name: file.name } })}>
        🔍 Start Deepfake Analysis
      </button>
    </div>
  );
}