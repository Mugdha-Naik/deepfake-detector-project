import { useEffect, useState } from "react";
function Count({ to, suf = "" }) {
  const [v, setV] = useState(0);
  useEffect(() => {
    let n = 0;
    const t = setInterval(() => { n += to / 40; if (n >= to) { n = to; clearInterval(t); } setV(n); }, 25);
    return () => clearInterval(t);
  }, [to]);
  return <>{Number.isInteger(to) ? Math.round(v) : v.toFixed(1)}{suf}</>;
}
const stats = [["Accuracy", 96.1, "%"], ["Precision", 94.8, "%"], ["Recall", 95.3, "%"], ["Videos Scanned", 1280, ""]];
const bars = [60, 85, 45, 92, 70, 55, 88, 76, 64, 95];
const table = [["Xception", "96.1%", "94.8%", "95.3%"], ["EfficientNet", "94.7%", "93.2%", "93.9%"], ["ResNet50", "91.2%", "90.1%", "90.6%"]];
export default function Dashboard() {
  return (
    <div className="page">
      <span className="chip">Analytics</span>
      <h1>Research <em>Dashboard</em></h1>
      <div className="grid4">
        {stats.map(([a, b, s]) => <div className="glass step stat" key={a}><p>{a}</p><h2 className="big"><Count to={b} suf={s} /></h2></div>)}
      </div>
      <div className="glass">
        <h3>Detections per Batch</h3>
        <div className="chart">{bars.map((h, i) => <div key={i} style={{ height: h + "%", animationDelay: i * 80 + "ms" }} />)}</div>
      </div>
      <div className="glass table-wrap">
        <h3>Model Comparison</h3>
        <table>
          <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th></tr></thead>
          <tbody>{table.map((r) => <tr key={r[0]}>{r.map((c) => <td key={c}>{c}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </div>
  );
}