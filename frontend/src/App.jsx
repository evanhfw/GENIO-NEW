import { useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState("");
  const [results, setResults] = useState(null);
  const [roadWidth, setRoadWidth] = useState("");
  const [areaLength, setAreaLength] = useState("");
  const [points, setPoints] = useState([]);
  const [canvasRef, setCanvasRef] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPoints([]);

    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.src = URL.createObjectURL(file);

    video.addEventListener("loadeddata", async () => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");

      video.currentTime = 0.1;
      await new Promise((resolve) => video.addEventListener("seeked", resolve));

      ctx.drawImage(video, 0, 0);
      setCanvasRef(canvas);
      URL.revokeObjectURL(video.src);
    });
  };

  const handleCanvasClick = (e) => {
    if (!canvasRef) return;

    const rect = canvasRef.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (points.length < 2) {
      setPoints([...points, [x, y]]);
    }
  };

  const handleRoadWidthChange = (event) => {
    setRoadWidth(event.target.value);
  };

  const handleAreaLengthChange = (event) => {
    setAreaLength(event.target.value);
  };

  const uploadVideo = async () => {
    if (!selectedFile || !roadWidth || !areaLength || points.length < 2) {
      setStatus("Harap lengkapi semua field dan pilih 2 titik!");
      return;
    }

    try {
      setStatus("Memproses...");

      // Simulasi perhitungan
      const distancePixels = Math.sqrt(
        Math.pow(points[1][0] - points[0][0], 2) +
          Math.pow(points[1][1] - points[0][1], 2)
      );

      const mockResults = {
        points: points,
        average_speed: ((areaLength / (distancePixels / 100)) * 3.6).toFixed(2),
        travel_time: (areaLength / (roadWidth / 3.6)).toFixed(2),
      };

      setResults(mockResults);
      setStatus("Proses selesai!");
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  return (
    <div className="App">
      <h1>Video Speed Detection Analyzer</h1>

      <div className="upload-section">
        <div className="input-group">
          <label>Upload Video:</label>
          <input type="file" accept="video/*" onChange={handleFileChange} />
        </div>

        {canvasRef && (
          <div className="video-preview">
            <canvas
              ref={(ref) => ref && setCanvasRef(ref)}
              onClick={handleCanvasClick}
              style={{ cursor: "crosshair", border: "1px solid #ccc" }}
            />
            <p>Klik untuk memilih titik awal dan akhir (2 titik)</p>
            {points.length > 0 && (
              <p>
                Titik terpilih:{" "}
                {points
                  .map((p) => `(${p[0].toFixed(0)}, ${p[1].toFixed(0)})`)
                  .join(", ")}
              </p>
            )}
          </div>
        )}

        <div className="input-group">
          <label>Lebar Jalan (meter):</label>
          <input
            type="number"
            value={roadWidth}
            onChange={handleRoadWidthChange}
            placeholder="Masukkan lebar jalan"
            required
          />
        </div>

        <div className="input-group">
          <label>Panjang Area (meter):</label>
          <input
            type="number"
            value={areaLength}
            onChange={handleAreaLengthChange}
            placeholder="Masukkan panjang area"
            required
          />
        </div>

        <button
          onClick={uploadVideo}
          disabled={
            !selectedFile || !roadWidth || !areaLength || points.length < 2
          }
        >
          Analisis Video
        </button>
        {status && <p className="status">{status}</p>}
      </div>

      {results && (
        <div className="results">
          <h2>Hasil Analisis</h2>
          <div className="analysis-results">
            <p>
              Koordinat Titik:{" "}
              {results.points
                .map((point) => `(${point[0]}, ${point[1]})`)
                .join(", ")}
            </p>
            <p>Kecepatan Rata-rata: {results.average_speed} km/h</p>
            <p>Waktu Tempuh: {results.travel_time} detik</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
