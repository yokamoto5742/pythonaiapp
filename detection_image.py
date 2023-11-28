import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

upload_img = st.file_uploader("画像をアップロードしてください", type=["png", "jpg"])

if upload_img is not None:
    bytes_data = upload_img.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    results = model(cv2_img, conf=0.5, classes=[0])
    output_img1 = results[0].plot(labels=True, conf=0.5)
    output_img2 = cv2.cvtColor(output_img1, cv2.COLOR_BGR2RGB)

    categories = results[0].boxes.cls
    person_num = len(categories)

    st.image(output_img2, caption="出力画像")
    st.text("人数: {}名".format(person_num))
