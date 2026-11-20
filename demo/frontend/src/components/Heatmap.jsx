export default function Heatmap({ size = 14 }) {
  const cells = Array.from({ length: size * size }, (_, i) => {
    const x = (i % size) - size / 2, y = Math.floor(i / size) - size / 2;
    const d = Math.sqrt(x * x + y * y) / (size / 2);
    return Math.max(0, 1 - d) * 0.9 + ((i * 37) % 10) / 100;
  });
  return (
    <div className="heatmap" style={{ gridTemplateColumns: `repeat(${size},1fr)` }}>
      {cells.map((v, i) => (
        <div key={i} style={{ background: `hsla(${220 - v * 220},95%,55%,${0.25 + v * 0.75})`, animationDelay: i * 4 + "ms" }} />
      ))}
    </div>
  );
}