import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict, deque

VIDEO_SOURCE = r"../input.mp4"
MODEL = YOLO(r"../Weekly Report/Weekly Report 1/weights yolo/yolo11n.pt")
SOURCE = np.array([[486, 153], [610, 149], [881, 275], [505, 286]])
TARGET_WIDTH = 5
TARGET_HEIGHT = 44

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target=np.ndarray):
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)

        self.m = cv2.getPerspectiveTransform(self.source, self.target)

    def transform_points(self, points: np.ndarray):
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


video_info = sv.VideoInfo.from_video_path(VIDEO_SOURCE)
byte_tracker = sv.ByteTrack(frame_rate=video_info.fps)

thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
bounding_box_annonator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

frame_generator = sv.get_video_frames_generator(VIDEO_SOURCE)
polygon_zone = sv.PolygonZone(polygon=SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

for frame in frame_generator:
    result = MODEL(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    detections = detections[polygon_zone.trigger(detections)]
    detections = byte_tracker.update_with_detections(detections)

    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    points = view_transformer.transform_points(points)

    labels = []

    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)
        if len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_end - coordinate_start)
            time = len(coordinates[tracker_id]) / video_info.fps
            speed = distance / time * 3.6
            labels.append(f"#{tracker_id} {speed:.2f} km/h")

    annotated_frame = frame.copy()
    annotated_frame = sv.draw_polygon(
        scene=annotated_frame, polygon=SOURCE, color=sv.Color.RED
    )
    annotated_frame = bounding_box_annonator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    cv2.imshow(winname="annotated_frame", mat=annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
