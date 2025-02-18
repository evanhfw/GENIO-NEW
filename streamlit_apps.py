import streamlit as st
import cv2
import numpy as np
import tempfile
import json
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
        # Inisialisasi session state jika belum ada
        if 'image_points' not in st.session_state:
            st.session_state.image_points = []
        if 'confirmed' not in st.session_state:
            st.session_state.confirmed = False

        # Inisialisasi input panjang dan lebar jalan jika belum ada
        if 'road_length' not in st.session_state:
            st.session_state.road_length = 0.0
        if 'road_width' not in st.session_state:
            st.session_state.road_width = 0.0

        # Konversi frame ke format yang bisa ditampilkan
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

                # Batasi hanya 4 titik
                if len(points) > 4:
                    points = points[:4]

                st.session_state.image_points = points

        # Input panjang dan lebar jalan
        st.session_state.road_length = st.number_input("Masukkan panjang jalan (meter)", min_value=0.0, format="%.2f")
        st.session_state.road_width = st.number_input("Masukkan lebar jalan (meter)", min_value=0.0, format="%.2f")

        # Tombol untuk mengonfirmasi titik dan input jalan
        if st.button("Tentukan Titik"):
            if len(st.session_state.image_points) == 4:
                st.session_state.confirmed = True
            else:
                st.error("Harap pilih tepat 4 titik sebelum menekan tombol ini.")

        # Menampilkan output dalam format JSON jika sudah dikonfirmasi
        if st.session_state.confirmed:
            json_output = {
                "koordinat": {
                    f"titik_{i+1}": list(st.session_state.image_points[i])
                    for i in range(4)
                },
                "panjang": st.session_state.road_length,
                "lebar": st.session_state.road_width
            }

            st.json(json.dumps(json_output, indent=4))
