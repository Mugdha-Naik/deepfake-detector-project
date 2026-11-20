import { NavLink } from "react-router-dom";
export default function Navbar() {
  return (
    <header className="navbar">
      <div className="logo"><span className="logo-mark">◈</span> DeepGuard<b>AI</b></div>
      <nav className="nav-links">
        <NavLink to="/">Analyze</NavLink>
        <NavLink to="/dashboard">Dashboard</NavLink>
        <NavLink to="/compression">Compression</NavLink>
      </nav>
      <div className="status"><i /> System Online</div>
    </header>
  );
}