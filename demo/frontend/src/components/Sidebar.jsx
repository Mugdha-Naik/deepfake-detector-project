import { NavLink } from "react-router-dom";
const items = [
  ["/", "🛡️", "Analyze Video"],
  ["/processing", "⚙️", "Processing"],
  ["/result", "🎯", "Result"],
  ["/compression", "🗜️", "Compression"],
  ["/dashboard", "📊", "Dashboard"],
];
export default function Sidebar() {
  return (
    <aside className="sidebar">
      <h3>Navigation</h3>
      {items.map(([to, icon, label]) => (
        <NavLink key={to} to={to} end className="side-link">
          <span>{icon}</span>{label}
        </NavLink>
      ))}
      <div className="side-card">
        <p>Model</p><b>Xception + LSTM</b>
        <div className="mini-bar"><div /></div>
      </div>
    </aside>
  );
}