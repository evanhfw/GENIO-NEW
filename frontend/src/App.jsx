import { useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState("");
  const [results, setResults] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const uploadVideo = async () => {
    if (!selectedFile) {
      setStatus("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("video", selectedFile);

    try {
      setStatus("Uploading...");
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setResults(data);
        setStatus("Upload successful! Processing...");
      } else {
        setStatus(`Error: ${data.detail}`);
      }
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  return (
    <div className="App">
      <h1>Video Speed Detection Analyzer</h1>

      <div className="upload-section">
        <input type="file" accept="video/*" onChange={handleFileChange} />
        <button onClick={uploadVideo} disabled={!selectedFile}>
          Upload Video
        </button>
        {status && <p className="status">{status}</p>}
      </div>

      {results && (
        <div className="results">
          <h2>Analysis Results</h2>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
