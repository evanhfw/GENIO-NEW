import { useState, useEffect } from "react";
import VideoPlayer from "./VideoPlayer";
import "./App.css";

function HLSPlayer() {
  const [videoUrl, setVideoUrl] = useState(
    "https://cctvjss.jogjakota.go.id/atcs/ATCS_Kleringan_Abu_Bakar_Ali.stream/playlist.m3u8"
  );
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState({});
  const [processingPlaylistUrl, setProcessingPlaylistUrl] = useState("");
  const [backendUrl] = useState("http://localhost:8000");

  // Configuration values
  const [source, setSource] = useState([
    [486, 153],
    [610, 149],
    [881, 275],
    [505, 286],
  ]);
  const [targetWidth, setTargetWidth] = useState(5);
  const [targetHeight, setTargetHeight] = useState(44);

  // Function to update a specific point in source
  const updateSourcePoint = (index, axis, value) => {
    const newSource = [...source];
    newSource[index][axis] = parseInt(value);
    setSource(newSource);
  };

  // Poll for status updates
  useEffect(() => {
    if (isProcessing) {
      const interval = setInterval(() => {
        fetch(`${backendUrl}/api/status`)
          .then((response) => response.json())
          .then((data) => {
            setStatus(data);
            if (data.has_output && data.playlist_url) {
              const fullPlaylistUrl = `${backendUrl}${data.playlist_url}`;
              console.log("Setting playlist URL to:", fullPlaylistUrl);
              setProcessingPlaylistUrl(fullPlaylistUrl);
              setIsPlaying(true);
              if (data.segments_count > 5) {
                // We have enough segments, no need to poll so frequently
                clearInterval(interval);
              }
            }
          })
          .catch((err) => {
            console.error("Error fetching status:", err);
            setError("Failed to get processing status");
          });
      }, 2000); // Poll every 2 seconds

      return () => clearInterval(interval);
    }
  }, [isProcessing, backendUrl]);

  const startProcessing = async () => {
    setError("");

    if (!videoUrl.endsWith(".m3u8")) {
      setError("URL must point to an m3u8 playlist file");
      return;
    }

    try {
      // Send configuration to backend
      const config = {
        hls_stream_url: videoUrl,
        source: source.flat(), // Backend expects a flat array
        target_width: targetWidth,
        target_height: targetHeight,
      };

      // Query parameters
      const params = new URLSearchParams();

      // Add parameters to the query string
      params.append("hls_stream_url", config.hls_stream_url);

      // Convert source array to string representation expected by backend
      params.append("source", JSON.stringify(config.source));

      params.append("target_width", config.target_width);
      params.append("target_height", config.target_height);

      // Make POST request to backend
      const response = await fetch(
        `${backendUrl}/config?${params.toString()}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const data = await response.json();

      if (data.processing_started) {
        setIsProcessing(true);
        // Once processing starts, we'll rely on the status polling to update the UI
      } else {
        setError("Failed to start processing. Check configuration.");
      }
    } catch (err) {
      console.error("Error starting processing:", err);
      setError("Failed to communicate with backend service");
    }
  };

  // Function to view raw m3u8 content for debugging
  const viewM3u8Content = async () => {
    try {
      const response = await fetch(processingPlaylistUrl);
      const text = await response.text();
      console.log("M3U8 Content:", text);
      alert(`M3U8 Content:\n${text}`);
    } catch (err) {
      console.error("Error fetching m3u8 content:", err);
      alert(`Error fetching m3u8: ${err.message}`);
    }
  };

  return (
    <div className="hls-player-container">
      <h1>Video Stream Processor</h1>

      <div className="config-section">
        <h2>Configuration</h2>

        <div className="input-group">
          <label>M3U8 Playlist URL:</label>
          <input
            type="text"
            value={videoUrl}
            onChange={(e) => setVideoUrl(e.target.value)}
            placeholder="https://example.com/playlist.m3u8"
            className="url-input"
          />
        </div>

        <div className="source-points">
          <h3>Source Points (Polygon)</h3>
          {source.map((point, i) => (
            <div key={i} className="point-inputs">
              <label>Point {i + 1}:</label>
              <input
                type="number"
                value={point[0]}
                onChange={(e) => updateSourcePoint(i, 0, e.target.value)}
                placeholder="X"
              />
              <input
                type="number"
                value={point[1]}
                onChange={(e) => updateSourcePoint(i, 1, e.target.value)}
                placeholder="Y"
              />
            </div>
          ))}
        </div>

        <div className="target-dimensions">
          <h3>Target Dimensions</h3>
          <div className="input-group">
            <label>Width:</label>
            <input
              type="number"
              value={targetWidth}
              onChange={(e) => setTargetWidth(parseInt(e.target.value))}
            />
          </div>
          <div className="input-group">
            <label>Height:</label>
            <input
              type="number"
              value={targetHeight}
              onChange={(e) => setTargetHeight(parseInt(e.target.value))}
            />
          </div>
        </div>

        <button
          className="process-button"
          onClick={startProcessing}
          disabled={isProcessing}
        >
          {isProcessing ? "Processing..." : "Start Processing"}
        </button>

        {error && <p className="error-message">{error}</p>}

        {isProcessing && !isPlaying && (
          <div className="status-section">
            <h3>Processing Status</h3>
            <p>Status: {status.status || "Initializing..."}</p>
            {status.has_output && (
              <p>Output Segments: {status.segments_count || 0}</p>
            )}
            {!status.has_output && <p>Waiting for output...</p>}
            <div className="loader"></div>
          </div>
        )}

        {isProcessing && (
          <div className="video-container">
            <h2>Processed Video Stream</h2>
            {isPlaying && processingPlaylistUrl ? (
              <VideoPlayer
                key={processingPlaylistUrl}
                src={processingPlaylistUrl}
              />
            ) : (
              <div className="loading-video">
                <div className="spinner"></div>
                <p>Processing video stream...</p>
              </div>
            )}

            {isProcessing && status.has_output && !isPlaying && (
              <div className="test-direct-playback">
                <h4>Debugging Options</h4>
                <p>Playlist URL: {processingPlaylistUrl}</p>
                <div className="debug-buttons">
                  <button
                    className="debug-button"
                    onClick={() => window.open(processingPlaylistUrl, "_blank")}
                  >
                    Test Direct M3U8 Access
                  </button>
                  <button className="debug-button" onClick={viewM3u8Content}>
                    View M3U8 Content
                  </button>
                  <button
                    className="debug-button"
                    onClick={() => window.open(`${backendUrl}/hls`, "_blank")}
                  >
                    Open Built-in Player
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="instructions">
        <h3>How to use:</h3>
        <ol>
          <li>Enter the URL of an m3u8 playlist file or use the default</li>
          <li>
            Adjust the polygon points if needed (these define the region of
            interest)
          </li>
          <li>Set target width and height for the transformation</li>
          <li>Click "Start Processing" to begin the video analysis</li>
          <li>Wait for the processed video stream to appear</li>
        </ol>
      </div>
    </div>
  );
}

export default HLSPlayer;
