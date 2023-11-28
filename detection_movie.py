import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import pandas as pd
import os  # os モジュールをインポート

model = YOLO("yolov8n.pt")

upload_file = st.file_uploader("動画ファイルをアップロードしてください", type="mp4")

if upload_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(upload_file.read())

    cap = cv2.VideoCapture(temp_file.name)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_file_name = 'object_detection_app_results.mp4'

    # 既存のファイルがあれば削除
    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    writer = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))

    frame_number = 0
    persons = []

    while cap.isOpened():
        ret, img = cap.read()

        if ret:
            if frame_number % int(fps) == 0:  # 1秒ごとに処理
                results = model(img, conf=0.5, classes=[0])
                img = results[0].plot(labels=True, conf=0.5)
                categories = results[0].boxes.cls
                person_num = len(categories)
                writer.write(img)
                persons.append(person_num)

            frame_number += 1
        else:
            break

    cap.release()
    writer.release()

    # 秒数と人数を記録
    person_data = pd.DataFrame({"sec": range(len(persons)), "person": persons})
    st.line_chart(person_data, x="sec", y="person")
    st.dataframe(person_data)
