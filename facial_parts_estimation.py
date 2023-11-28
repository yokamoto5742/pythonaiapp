import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO, BufferedReader

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

camera_img = st.camera_input(label="インカメラ画像")

if camera_img is not None:

    bytes_data = camera_img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img)
    eye_center = (results.multi_face_landmarks[0].landmark[33].x + results.multi_face_landmarks[0].landmark[133].x) / 2
    th = results.multi_face_landmarks[0].landmark[468].x - eye_center
    if th < 0:
        state = "右を向いている"
    else:
        state = "左を向いている"

    output_img = img.copy()
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=output_img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            image=output_img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        mp_drawing.draw_landmarks(
            image=output_img,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    ret, enco_img = cv2.imencode(
        ".png",
        cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    )

    BtyesIO_img = BytesIO(enco_img.tostring())
    BufferedReader_img = BufferedReader(BtyesIO_img)

    st.text("目線は: " + state)
    st.image(output_img, caption="予測結果")
    st.download_button(
        label="ダウンロード",
        data=BufferedReader_img,
        file_name="output.png",
        mime="image/png"
    )
