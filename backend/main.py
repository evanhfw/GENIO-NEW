from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict, deque

app = FastAPI()

# Configuration
HLS_STREAM_URL = "https://cctvjss.jogjakota.go.id/atcs/ATCS_Kleringan_Abu_Bakar_Ali.stream/playlist.m3u8"
SOURCE = np.array([[486, 153], [610, 149], [881, 275], [505, 286]])
TARGET_WIDTH = 5
TARGET_HEIGHT = 44

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

# Initialize models and components
model = YOLO(r"models/yolo11n_vehicle.pt")

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(self.source, self.target)

    def transform_points(self, points: np.ndarray):
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
byte_tracker = sv.ByteTrack()
coordinates = defaultdict(lambda: deque(maxlen=30))

def generate_frames():
    cap = cv2.VideoCapture(HLS_STREAM_URL)
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection and tracking
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_tracker.update_with_detections(detections)

        # Speed calculation
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        
        # Handle empty detections case
        if points.size == 0:
            annotated_frame = frame.copy()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        try:
            points = view_transformer.transform_points(points)
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
        annotated_frame = sv.draw_polygon(scene=frame, polygon=SOURCE, color=sv.Color.RED)
        annotated_frame = sv.BoxAnnotator().annotate(scene=annotated_frame, detections=detections)
        annotated_frame = sv.LabelAnnotator().annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/")
async def home():
    return HTMLResponse(content=f"""
    <html>
        <head>
            <title>Live Streaming CCTV Jogja</title>
        </head>
        <body>
            <h1>Live Streaming CCTV Kleringan Abu Bakar Ali</h1>
            <img src="/video_feed" style="width: 1280px; height: 720px;">
        </body>
    </html>
    """)

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')
