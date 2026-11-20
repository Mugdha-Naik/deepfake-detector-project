export default function ProgressBar({ value }) {
  return (
    <div className="track"><div className="fill" style={{ width: value + "%" }} /></div>
  );
}