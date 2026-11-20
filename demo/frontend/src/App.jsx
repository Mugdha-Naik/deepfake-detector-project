import "./App.css";
import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import ParticleBg from "./components/ParticleBg";
import Intro from "./components/Intro";
import Analyze from "./pages/Analyze";
import Processing from "./pages/Processing";
import Result from "./pages/Result";
import Compression from "./pages/Compression";
import Dashboard from "./pages/Dashboard";

export default function App() {
  const [intro, setIntro] = useState(true);

  useEffect(() => {
    let last = null;
    const move = (e) => {
      const card = e.target.closest ? e.target.closest(".glass") : null;
      if (last && last !== card) last.style.transform = "";
      last = card;
      if (!card) return;
      const r = card.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width;
      const y = (e.clientY - r.top) / r.height;
      card.style.setProperty("--mx", x * 100 + "%");
      card.style.setProperty("--my", y * 100 + "%");
      card.style.transform = `perspective(900px) rotateX(${(0.5 - y) * 5}deg) rotateY(${(x - 0.5) * 5}deg)`;
    };
    window.addEventListener("mousemove", move);
    return () => window.removeEventListener("mousemove", move);
  }, []);

  return (
    <BrowserRouter>
      {intro && <Intro onDone={() => setIntro(false)} />}
      <div className="bg-orb o1" /><div className="bg-orb o2" /><div className="bg-grid" />
      <ParticleBg />
      <div className={"app " + (intro ? "" : "ready")}>
        <Navbar />
        <div className="app-layout">
          <Sidebar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Analyze />} />
              <Route path="/processing" element={<Processing />} />
              <Route path="/result" element={<Result />} />
              <Route path="/compression" element={<Compression />} />
              <Route path="/dashboard" element={<Dashboard />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}