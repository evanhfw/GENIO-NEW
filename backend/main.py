from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict, deque
import threading
import os
import time
import subprocess
import json
from pathlib import Path
import shutil

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration storage with empty initial values
config = {
    "hls_stream_url": "",
    "source": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # Default placeholder
    "target_width": 5,
    "target_height": 44,
    "view_transformer": None,
    "hls_output_dir": "output/hls",
    "hls_segment_time": 2,  # segment duration in seconds
    "hls_playlist_size": 10,  # number of segments to keep in playlist
    "hls_fps": 20,  # output framerate for HLS
    "is_configured": False  # Flag to indicate if configuration has been set
}
config_lock = threading.Lock()

# Function to clear the HLS output directory
def clear_hls_directory():
    if os.path.exists(config["hls_output_dir"]):
        # Delete all files in directory
        try:
            shutil.rmtree(config["hls_output_dir"])
            print(f"Cleared HLS directory: {config['hls_output_dir']}")
        except Exception as e:
            print(f"Error clearing HLS directory: {e}")
    # Recreate the empty directory
    os.makedirs(config["hls_output_dir"], exist_ok=True)

# Clear HLS directory at startup
clear_hls_directory()

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(self.source, self.target)

    def transform_points(self, points: np.ndarray):
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


# Initialize view transformer
with config_lock:
    target_points = np.array([
        [0, 0],
        [config["target_width"] - 1, 0],
        [config["target_width"] - 1, config["target_height"] - 1],
        [0, config["target_height"] - 1],
    ])
    config["view_transformer"] = ViewTransformer(source=config["source"], target=target_points)


view_transformer = ViewTransformer(source=config["source"], target=target_points)
byte_tracker = sv.ByteTrack()
coordinates = defaultdict(lambda: deque(maxlen=30))
model = None  # We'll initialize this when needed
model_lock = threading.Lock()

# Variabel untuk thread HLS generator
hls_thread = None
stop_hls_thread = threading.Event()

def initialize_model():
    """Initialize YOLO model if not already loaded"""
    global model
    with model_lock:
        if model is None:
            print("Loading YOLO model...")
            model = YOLO(r"models/yolo11n_vehicle.pt")
            print("YOLO model loaded successfully")

def generate_frames():
    with config_lock:
        # Check if configuration is valid
        if not config["is_configured"] or not config["hls_stream_url"]:
            print("Error: Invalid configuration, HLS stream URL not set")
            return
    
    # Initialize the model
    initialize_model()
    
    # Try to open the video capture
    cap = cv2.VideoCapture(config["hls_stream_url"])
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {config['hls_stream_url']}")
        return
    
    with config_lock:  # Lock while initializing components
        polygon_zone = sv.PolygonZone(polygon=config["source"])
        vt = config["view_transformer"]
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up HLS output with ffmpeg
    hls_path = Path(config["hls_output_dir"])
    m3u8_path = hls_path / "playlist.m3u8"
    
    ffmpeg_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # Sesuaikan dengan lokasi FFmpeg Anda
    
    # Create ffmpeg process for HLS output
    ffmpeg_cmd = [
        ffmpeg_path,  # Menggunakan path lengkap
        '-y',  # overwrite output files
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', '30',
        '-i', '-',  # input from pipe
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Changed from veryfast to ultrafast for lower latency
        '-tune', 'zerolatency',  # Optimize for streaming
        '-profile:v', 'baseline',  # More compatible profile
        '-level', '3.0',
        '-g', '30',  # One keyframe per second at 30fps
        '-keyint_min', '30',  # Force keyframes at regular intervals
        '-sc_threshold', '0',  # Disable scene change detection for consistent segments
        '-bufsize', '5000k',  # Increase buffer size
        '-maxrate', '5000k',  # Set maximum bitrate
        '-f', 'hls',
        '-hls_time', '1',  # Shorter segments (1 second instead of 2)
        '-hls_list_size', '5',  # Keep fewer segments in playlist for quicker updates
        '-hls_flags', 'delete_segments+append_list+discont_start',  # Delete old segments, append to list, mark discontinuities
        '-hls_segment_type', 'mpegts',  # Use MPEG-TS segments for better compatibility
        '-hls_segment_filename', f'{hls_path}/segment_%03d.ts',
        str(m3u8_path)
    ]
    
    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    except Exception as e:
        print(f"Error starting ffmpeg process: {e}")
        cap.release()
        return
    
    while cap.isOpened() and not stop_hls_thread.is_set():
        ret, frame = cap.read()
        if not ret:
            # Reconnect if stream ends
            cap.release()
            time.sleep(2)  # Wait before reconnecting
            
            with config_lock:
                # Check if config has changed during reconnection
                if not config["is_configured"] or not config["hls_stream_url"]:
                    break
                cap = cv2.VideoCapture(config["hls_stream_url"])
            
            if not cap.isOpened():
                print(f"Error: Failed to reconnect to stream")
                break
                
            continue

        # Object detection and tracking
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        with config_lock:  # Use current polygon_zone
            polygon_zone = sv.PolygonZone(polygon=config["source"])
            detections = detections[polygon_zone.trigger(detections)]
            
        detections = byte_tracker.update_with_detections(detections)

        # Speed calculation
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        
        # Handle empty detections case
        if points.size == 0:
            annotated_frame = frame.copy()
            # Write to HLS output
            ffmpeg_process.stdin.write(annotated_frame.tobytes())
            continue

        try:
            with config_lock:
                vt = config["view_transformer"]
            points = vt.transform_points(points)
        except Exception as e:
            print(f"Error transforming points: {e}")
            continue

        labels = []
        
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)
            if len(coordinates[tracker_id]) >= 15:
                distance = abs(coordinates[tracker_id][-1] - coordinates[tracker_id][0])
                speed = distance / (len(coordinates[tracker_id]) / 30) * 3.6
                labels.append(f"#{tracker_id} {speed:.2f} km/h")
            else:
                labels.append(f"#{tracker_id}")

        # Annotate frame
        with config_lock:
            annotated_frame = sv.draw_polygon(scene=frame, polygon=config["source"], color=sv.Color.RED)
        
        annotated_frame = sv.BoxAnnotator().annotate(scene=annotated_frame, detections=detections)
        annotated_frame = sv.LabelAnnotator().annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Write to HLS output
        try:
            ffmpeg_process.stdin.write(annotated_frame.tobytes())
        except Exception as e:
            print(f"Error writing to ffmpeg: {e}")
            break
    
    # Clean up
    cap.release()
    try:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
    except:
        pass
    print("HLS generator thread finished")

# Fungsi untuk memulai thread HLS
def start_hls_thread():
    global hls_thread, stop_hls_thread
    
    # Stop existing thread if running
    if hls_thread and hls_thread.is_alive():
        stop_hls_thread.set()
        hls_thread.join(timeout=5)
        stop_hls_thread.clear()
    
    # Start new thread
    hls_thread = threading.Thread(target=generate_frames)
    hls_thread.daemon = True
    hls_thread.start()
    print("HLS generator thread started")

# Jalankan thread HLS saat startup
@app.on_event("startup")
async def startup_event():
    # Only clear the HLS directory, but don't start the thread
    clear_hls_directory()
    # Don't start HLS thread automatically
    # start_hls_thread() - removed

@app.on_event("shutdown")
async def shutdown_event():
    global stop_hls_thread
    stop_hls_thread.set()
    if hls_thread:
        hls_thread.join(timeout=5)

# Modifikasi generate_frames untuk API streaming
def generate_frames_api():
    with config_lock:
        # Check if configuration is valid
        if not config["is_configured"] or not config["hls_stream_url"]:
            # Return error frames
            for _ in range(10):  # Send a few error frames
                # Create a black frame with error message
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Stream not configured", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
    
    # Initialize the model
    initialize_model()
    
    cap = cv2.VideoCapture(config["hls_stream_url"])
    if not cap.isOpened():
        # Return error frames if can't open stream
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Cannot open stream", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        for _ in range(10):  # Send a few error frames
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    with config_lock:
        polygon_zone = sv.PolygonZone(polygon=config["source"])
        vt = config["view_transformer"]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            
            # Try to reconnect
            with config_lock:
                if not config["is_configured"] or not config["hls_stream_url"]:
                    break
                cap = cv2.VideoCapture(config["hls_stream_url"])
            
            if not cap.isOpened():
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Connection lost", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                break
            
            continue
            
        # Object detection and tracking (sama seperti di generate_frames)
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        with config_lock:
            polygon_zone = sv.PolygonZone(polygon=config["source"])
            detections = detections[polygon_zone.trigger(detections)]
        
        detections = byte_tracker.update_with_detections(detections)
        
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    
        
        # Annotate frame
        with config_lock:
            annotated_frame = sv.draw_polygon(scene=frame, polygon=config["source"], color=sv.Color.RED)
        
        annotated_frame = sv.BoxAnnotator().annotate(scene=annotated_frame, detections=detections)
        
        # Convert to JPEG for web streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(generate_frames_api(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/")
async def home():
    return HTMLResponse(content=f"""
    <html>
        <head>
            <title>Live Streaming CCTV Jogja</title>
            <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        </head>
        <body>
            <h1>Live Streaming CCTV Kleringan Abu Bakar Ali</h1>
            <video id="video" controls style="width: 1280px; height: 720px;"></video>
            <script>
                var video = document.getElementById('video');
                if(Hls.isSupported()) {{
                    var hls = new Hls();
                    hls.loadSource('/hls/playlist.m3u8');
                    hls.attachMedia(video);
                    hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                        video.play();
                    }});
                }}
                else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                    video.src = '/hls/playlist.m3u8';
                    video.addEventListener('loadedmetadata', function() {{
                        video.play();
                    }});
                }}
            </script>
        </body>
    </html>
    """)

@app.get("/hls")
async def hls_player():
    """HTML page for HLS player"""
    return HTMLResponse(content=f"""
    <html>
        <head>
            <title>HLS Streaming CCTV</title>
            <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                .debug {{ background: #f0f0f0; padding: 10px; margin-top: 10px; }}
                .error {{ color: red; }}
                video {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>HLS Streaming Player</h1>
            <video id="video" controls style="width: 100%; max-width: 1280px; height: auto;"></video>
            <div class="debug">
                <h3>Debug Info:</h3>
                <div id="status"></div>
                <div id="error" class="error"></div>
            </div>
            <script>
                var video = document.getElementById('video');
                var status = document.getElementById('status');
                var errorDiv = document.getElementById('error');
                
                function updateStatus(message) {{
                    status.innerHTML += '<p>' + message + '</p>';
                }}
                
                function showError(message) {{
                    errorDiv.innerHTML += '<p>' + message + '</p>';
                }}
                
                updateStatus('Testing HLS.js support: ' + (Hls.isSupported() ? 'Supported' : 'Not supported'));
                
                if(Hls.isSupported()) {{
                    updateStatus('Initializing HLS.js');
                    var hls = new Hls({{
                        debug: true,
                        enableWorker: true
                    }});
                    
                    hls.on(Hls.Events.ERROR, function(event, data) {{
                        showError('HLS Error: ' + JSON.stringify(data));
                    }});
                    
                    hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                        updateStatus('Manifest loaded, trying to play');
                        video.play().catch(e => showError('Play failed: ' + e));
                    }});
                    
                    updateStatus('Loading source: /hls/playlist.m3u8');
                    hls.loadSource('/hls/playlist.m3u8');
                    hls.attachMedia(video);
                }}
                else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                    updateStatus('Using native HLS support');
                    video.src = '/hls/playlist.m3u8';
                    video.addEventListener('loadedmetadata', function() {{
                        updateStatus('Metadata loaded, trying to play');
                        video.play().catch(e => showError('Play failed: ' + e));
                    }});
                    
                    video.addEventListener('error', function() {{
                        showError('Video error: ' + video.error.message);
                    }});
                }} else {{
                    showError('HLS is not supported in this browser');
                }}
            </script>
        </body>
    </html>
    """)

@app.get("/hls/{path:path}")
async def serve_hls_files(path: str):
    """Serve HLS playlist and segment files"""
    file_path = os.path.join(config["hls_output_dir"], path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Read the file content
    content = open(file_path, 'rb').read()
    
    # Fix m3u8 files to use absolute URLs and optimize for streaming
    if path.endswith('.m3u8'):
        try:
            # Try to decode assuming it's utf-8 text
            text_content = content.decode('utf-8')
            
            # Add additional HLS directives to optimize streaming
            if not '#EXT-X-VERSION:' in text_content:
                text_content = "#EXT-X-VERSION:3\n" + text_content
            
            # Add low latency directives if not present
            if not '#EXT-X-TARGETDURATION:' in text_content:
                text_content = "#EXT-X-TARGETDURATION:1\n" + text_content
            
            # Add part duration hint for low latency
            if not '#EXT-X-PART-INF:' in text_content:
                text_content = "#EXT-X-PART-INF:PART-TARGET=0.5\n" + text_content
            
            # Make sure segment URLs are absolute
            if 'segment_' in text_content and not 'http://' in text_content:
                # Replace with absolute URLs
                base_url = f"http://localhost:8000/hls/"
                text_content = text_content.replace('segment_', f'{base_url}segment_')
            
            # Convert back to bytes
            content = text_content.encode('utf-8')
        except Exception as e:
            print(f"Error modifying m3u8: {e}")
            # If decoding fails, leave the content as is
            pass
    
    # Create proper response with CORS headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Origin, Accept, Range",
        "Access-Control-Expose-Headers": "Content-Length, Content-Range, Accept-Ranges",
        "Cache-Control": "no-cache"
    }
    
    if path.endswith('.m3u8'):
        return Response(
            content=content, 
            media_type="application/vnd.apple.mpegurl",
            headers=headers
        )
    elif path.endswith('.ts'):
        # Add appropriate cache headers for ts segments
        return Response(
            content=content, 
            media_type="video/MP2T",
            headers={
                **headers,
                "Cache-Control": "public, max-age=86400",  # Cache segments for 24 hours
            }
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

@app.get("/config")
def get_config():
    """Get current configuration"""
    # Check if the HLS processing is active
    is_active = hls_thread is not None and hls_thread.is_alive()
    
    # Check if HLS output is ready
    hls_output_ready = False
    playlist_path = os.path.join(config["hls_output_dir"], "playlist.m3u8")
    if os.path.exists(playlist_path):
        # Check if we have at least one segment file
        dir_contents = os.listdir(config["hls_output_dir"])
        ts_files = [f for f in dir_contents if f.endswith(".ts")]
        hls_output_ready = len(ts_files) > 0
    
    return {
        "hls_stream_url": config["hls_stream_url"],
        "source": config["source"].tolist(),
        "target_width": config["target_width"],
        "target_height": config["target_height"],
        "hls_output_dir": config["hls_output_dir"],
        "hls_segment_time": config["hls_segment_time"],
        "hls_playlist_size": config["hls_playlist_size"],
        "hls_fps": config["hls_fps"],
        "is_configured": config["is_configured"],
        "is_processing": is_active,
        "hls_output_ready": hls_output_ready
    }

@app.post("/config")
def update_config(
    hls_stream_url: str = None,
    source: str = None,  # Changed to string to accept JSON string
    target_width: int = None,
    target_height: int = None,
    hls_output_dir: str = None,
    hls_segment_time: int = None,
    hls_playlist_size: int = None,
    hls_fps: int = None
):
    """Update configuration parameters"""
    global hls_thread, stop_hls_thread
    
    with config_lock:
        if hls_stream_url:
            config["hls_stream_url"] = hls_stream_url
        
        if source:
            try:
                # Parse the JSON string to a Python list
                source_list = json.loads(source)
                
                # Convert to numpy array and reshape
                source_array = np.array(source_list)
                if len(source_array) == 8:  # If it's a flat array [x1,y1,x2,y2,x3,y3,x4,y4]
                    source_array = source_array.reshape(4, 2)
                elif len(source_array) != 4:  # If it's not a 4x2 array
                    raise ValueError("Source must have exactly 4 points")
                
                config["source"] = source_array
                print(f"Updated source to: {config['source']}")
            except Exception as e:
                print(f"Error parsing source: {e}")
                raise HTTPException(400, f"Invalid source format: {str(e)}")
        
        if target_width or target_height:
            config["target_width"] = target_width or config["target_width"]
            config["target_height"] = target_height or config["target_height"]
            
            # Regenerate target points and view transformer
            target_points = np.array([
                [0, 0],
                [config["target_width"] - 1, 0],
                [config["target_width"] - 1, config["target_height"] - 1],
                [0, config["target_height"] - 1],
            ])
            config["view_transformer"] = ViewTransformer(
                source=config["source"], 
                target=target_points
            )
        
        if hls_output_dir:
            config["hls_output_dir"] = hls_output_dir
            os.makedirs(hls_output_dir, exist_ok=True)
            
        if hls_segment_time:
            config["hls_segment_time"] = hls_segment_time
            
        if hls_playlist_size:
            config["hls_playlist_size"] = hls_playlist_size
            
        if hls_fps:
            config["hls_fps"] = hls_fps
        
        # Mark as configured if we have the minimum required configuration
        if config["hls_stream_url"]:
            config["is_configured"] = True
        
        restart_thread = False
        if hls_stream_url:
            restart_thread = True
        
        if source:
            restart_thread = True
        
        if target_width or target_height:
            restart_thread = True
        
        if hls_output_dir:
            restart_thread = True
        
        if hls_segment_time:
            restart_thread = True
        
        if hls_playlist_size:
            restart_thread = True
        
        if hls_fps:
            restart_thread = True
        
    # Start or restart HLS thread if configuration has been updated
    if restart_thread and config["is_configured"]:
        start_hls_thread()
        
    return {"message": "Configuration updated", "new_config": get_config(), "processing_started": config["is_configured"]}

@app.get("/status")
def get_status():
    """Get current processing status"""
    return {
        "is_configured": config["is_configured"],
        "is_processing": hls_thread is not None and hls_thread.is_alive()
    }

@app.get("/api/status")
def api_status():
    """Get detailed API status for frontend"""
    # Check if the HLS processing is active
    is_active = hls_thread is not None and hls_thread.is_alive()
    
    # Check if HLS output is ready
    hls_output_ready = False
    hls_segments = []
    playlist_path = os.path.join(config["hls_output_dir"], "playlist.m3u8")
    
    if os.path.exists(playlist_path):
        # Check if we have segment files
        dir_contents = os.listdir(config["hls_output_dir"])
        ts_files = [f for f in dir_contents if f.endswith(".ts")]
        hls_output_ready = len(ts_files) > 0
        hls_segments = ts_files
    
    # Get the m3u8 playlist URL for frontend
    base_url = "/hls/playlist.m3u8"
    
    return {
        "status": "running" if is_active else "stopped",
        "is_configured": config["is_configured"],
        "has_stream_url": bool(config["hls_stream_url"]),
        "has_output": hls_output_ready,
        "segments_count": len(hls_segments),
        "playlist_url": base_url if hls_output_ready else None,
        "model_loaded": model is not None
    }
