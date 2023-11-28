import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO, BufferedReader

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)

upload_img = st.file_uploader("画像をアップロードしてください", type=["png", "jpg"])

if upload_img is not None:
    bytes_data = upload_img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    results = pose.process(img)
    output_img = img.copy()
    mp_drawing.draw_landmarks(
        output_img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
    )
    right_th = results.pose_landmarks.landmark[20].y - results.pose_landmarks.landmark[12].y

    if right_th < 0:
        right_state = "挙がっている"
    else:
        right_state = "挙がっていない"

    left_th = results.pose_landmarks.landmark[19].y - results.pose_landmarks.landmark[11].y

    if left_th < 0:
        left_state = "挙がっている"
    else:
        left_state = "挙がっていない"

    ret, enco_img = cv2.imencode(
        ".png",
        cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    )
    BtyesIO_img = BytesIO(enco_img.tostring())
    BufferedReader_img = BufferedReader(BtyesIO_img)

    st.text("右手の状態: " + right_state)
    st.text("左手の状態: " + left_state)
    st.image(output_img, caption="予測結果")
    st.download_button(
        label="ダウンロード",
        data=BufferedReader_img,
        file_name="output.png",
        mime="image/png"
    )
