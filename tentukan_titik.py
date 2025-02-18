import cv2
import numpy as np

VIDEO_PATH = r"../input.mp4"
image_points = []  # Menyimpan titik dari kursor


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Klik kiri
        image_points.append((x, y))  # Tambahkan koordinat
        print(f"Titik ditambahkan: ({x}, {y})")
        cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", frame_copy)


cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
frame_copy = frame.copy()

cv2.imshow("Select Points", frame_copy)
cv2.setMouseCallback("Select Points", click_event)

print(
    "Klik pada 4 titik di perempatan jalan (urutkan searah jarum jam mulai dari kiri atas)."
)
while len(image_points) < 4:
    if cv2.waitKey(1) & 0xFF == 27:  # Tekan ESC untuk keluar
        print("Pemilihan dibatalkan.")
        break

cv2.destroyAllWindows()
cap.release()

# Konversi ke format numpy array
if len(image_points) == 4:
    image_points = np.array(image_points, dtype=np.float32)
    print("Koordinat image_points:")
    print(image_points)
else:
    print("Kurang dari 4 titik yang dipilih.")
