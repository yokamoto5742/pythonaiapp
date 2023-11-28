import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO, BufferedReader

model1 = torch.hub.load('bryandlee/animegan2-pytorch:main', "generator", pretrained="face_paint_512_v2")

model2 = torch.hub.load('bryandlee/animegan2-pytorch:main', "generator", pretrained="celeba_distill")

face2paint = torch.hub.load('bryandlee/animegan2-pytorch:main', "face2paint")

col1, col2 = st.columns(2)

upload_img = st.sidebar.file_uploader("画像をアップロードしてください", type=["png", "jpg"])

select_model = st.sidebar.selectbox("モデル選択:", ["face_paint_512_v2", "celeba_distill"])

if upload_img is not None:
    bytes_data = upload_img.getvalue()
    tg_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    tg_img = cv2.cvtColor(tg_img, cv2.COLOR_BGR2RGB)
    original_img = tg_img.copy()

    tg_img = Image.fromarray(tg_img)
    if select_model == "face_paint_512_v2":
        output_img = face2paint(model1, tg_img, size=512)
    elif select_model == "celeba_distill":
        output_img = face2paint(model2, tg_img, size=512)

    ret, enco_img = cv2.imencode(".png", cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR))
    BytesIO_img = BytesIO(enco_img.tostring())
    BufferedReader_img = BufferedReader(BytesIO_img)

    with col1:
        st.header("入力画像")
        st.image(original_img, use_column_width=True)

    with col2:
        st.header("出力画像")
        st.image(output_img, use_column_width=True)

        st.download_button(
            label="ダウンロード",
            data=BufferedReader_img,
            file_name="output.png",
            mime="image/png",
            )
