const rows = [["Original", 100, "48 MB", "96.1%"], ["Low Compression", 70, "34 MB", "93.4%"], ["Medium Compression", 45, "22 MB", "88.7%"], ["High Compression", 20, "9 MB", "79.2%"]];
export default function Compression() {
  return (
    <div className="page">
      <span className="chip">Robustness Study</span>
      <h1>Compression <em>Impact</em></h1>
      <p className="sub">How video compression reduces detection accuracy.</p>
      <div className="glass">
        {rows.map(([n, w, s, a], i) => (
          <div className="bar-row" key={n}>
            <span>{n}</span>
            <div className="track"><div className="fill" style={{ width: w + "%", animationDelay: i * 150 + "ms" }} /></div>
            <b>{s}</b><b className="acc">{a}</b>
          </div>
        ))}
      </div>
      <div className="warn">Higher compression destroys subtle artifacts, which lowers detection accuracy.</div>
    </div>
  );
}