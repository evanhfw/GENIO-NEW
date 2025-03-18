from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict, deque
import uuid
import threading
import os
import time
import subprocess
from pathlib import Path

app = FastAPI()

# Configuration storage with default values
config = {
    "hls_stream_url": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Kleringan_Abu_Bakar_Ali.stream/playlist.m3u8",
    "source": np.array([[486, 153], [610, 149], [881, 275], [505, 286]]),
    "target_width": 5,
    "target_height": 44,
    "view_transformer": None,
    "hls_output_dir": "output/hls",
    "hls_segment_time": 2,  # segment duration in seconds
    "hls_playlist_size": 5,  # number of segments to keep in the playlist
    "hls_fps": 20  # output framerate for HLS
}
config_lock = threading.Lock()

# Create output directory if it doesn't exist
os.makedirs(config["hls_output_dir"], exist_ok=True)

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
model = YOLO(r"models/yolo11n_vehicle.pt")

# Variabel untuk thread HLS generator
hls_thread = None
stop_hls_thread = threading.Event()

def generate_frames():
    cap = cv2.VideoCapture(config["hls_stream_url"])
    
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
    
    # Create ffmpeg process for HLS output
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # overwrite output files
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(config["hls_fps"]),
        '-i', '-',  # input from pipe
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-f', 'hls',
        '-hls_time', str(config["hls_segment_time"]),
        '-hls_list_size', str(config["hls_playlist_size"]),
        '-hls_flags', 'delete_segments',
        '-hls_segment_filename', f'{hls_path}/%03d.ts',
        str(m3u8_path)
    ]
    
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    
    while cap.isOpened() and not stop_hls_thread.is_set():
        ret, frame = cap.read()
        if not ret:
            # Reconnect if stream ends
            cap.release()
            time.sleep(2)  # Wait before reconnecting
            cap = cv2.VideoCapture(config["hls_stream_url"])
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
        ffmpeg_process.stdin.write(annotated_frame.tobytes())
    
    # Clean up
    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
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
    start_hls_thread()

@app.on_event("shutdown")
async def shutdown_event():
    global stop_hls_thread
    stop_hls_thread.set()
    if hls_thread:
        hls_thread.join(timeout=5)

# Modifikasi generate_frames untuk API streaming
def generate_frames_api():
    cap = cv2.VideoCapture(config["hls_stream_url"])
    
    with config_lock:
        polygon_zone = sv.PolygonZone(polygon=config["source"])
        vt = config["view_transformer"]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Object detection and tracking (sama seperti di generate_frames)
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_tracker.update_with_detections(detections)
        
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        
        # Sama seperti generate_frames tetapi hanya untuk API streaming
        # ...
        
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
            <title>HLS Streaming CCTV Jogja</title>
            <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        </head>
        <body>
            <h1>HLS Streaming CCTV Kleringan Abu Bakar Ali</h1>
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

@app.get("/hls/{path:path}")
async def serve_hls_files(path: str):
    """Serve HLS playlist and segment files"""
    file_path = os.path.join(config["hls_output_dir"], path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if path.endswith('.m3u8'):
        return Response(content=open(file_path, 'rb').read(), media_type="application/vnd.apple.mpegurl")
    elif path.endswith('.ts'):
        return Response(content=open(file_path, 'rb').read(), media_type="video/MP2T")
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

@app.get("/config")
def get_config():
    """Get current configuration"""
    return {
        "hls_stream_url": config["hls_stream_url"],
        "source": config["source"].tolist(),
        "target_width": config["target_width"],
        "target_height": config["target_height"],
        "hls_output_dir": config["hls_output_dir"],
        "hls_segment_time": config["hls_segment_time"],
        "hls_playlist_size": config["hls_playlist_size"],
        "hls_fps": config["hls_fps"]
    }

@app.post("/config")
def update_config(
    hls_stream_url: str = None,
    source: list = None,
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
                config["source"] = np.array(source).reshape(4, 2)
            except:
                raise HTTPException(400, "Invalid source format. Expected 4 points [x,y]")
        
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
        
    # Restart HLS thread jika konfigurasi yang mempengaruhi proses berubah
    if restart_thread:
        start_hls_thread()
        
    return {"message": "Configuration updated", "new_config": get_config()}
