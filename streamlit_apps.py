import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.title("Pilih Titik di Video")

uploaded_file = st.file_uploader("Unggah Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Simpan video ke file sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("Gagal membaca video. Coba unggah ulang file video.")
    else:
        if 'image_points' not in st.session_state:
            st.session_state.image_points = []
        
        frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_copy)
        
        st.text("Klik pada 4 titik di gambar.")
        
        max_width = 800  # Maksimum lebar gambar agar tidak terpotong
        scale_factor = max_width / frame.shape[1] if frame.shape[1] > max_width else 1
        new_width = int(frame.shape[1] * scale_factor)
        new_height = int(frame.shape[0] * scale_factor)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=frame_pil.resize((new_width, new_height)),
            height=new_height,
            width=new_width,
            drawing_mode="point",
            key="canvas"
        )
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if len(objects) > 0:
                points = [(int(obj["left"] / scale_factor), int(obj["top"] / scale_factor)) for obj in objects]
                st.session_state.image_points = points[:4]
        
        if len(st.session_state.image_points) == 4:
            st.success("Titik berhasil dipilih!")
            image_points_np = np.array(st.session_state.image_points, dtype=np.float32)
            st.write("Koordinat yang dipilih:", image_points_np)