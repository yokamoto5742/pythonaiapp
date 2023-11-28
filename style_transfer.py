import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ratio = 512 / max(img.shape[:2])
    img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)

    img = img / 255.0
    img = img.astype(np.float32)
    img = img[tf.newaxis, :]
    return img


module = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

col1, col2 = st.columns(2)

upload_img = st.sidebar.file_uploader("画像をアップロードしてください", type=["png", "jpg"])

upload_style_img = st.sidebar.file_uploader("画風の画像をアップロードしてください", type=["png", "jpg"])

if (upload_img is not None) & (upload_style_img is not None):
    bytes_data = upload_img.getvalue()
    tg_img = cv2.imdecode(np.frombuffer(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    bytes_data = upload_style_img.getvalue()
    style_img = cv2.imdecode(np.frombuffer(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    output_img = cv2.cvtColor(tg_img, cv2.COLOR_BGR2RGB)
    output_style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)

    tg_img = preprocess(tg_img)
    style_img = preprocess(style_img)

    results = module(tf.constant(tg_img), tf.constant(style_img))[0][0]
    results = results.numpy()

    with col1:
        st.header("画像")
        st.image(output_img)

    with col2:
        st.header("画風")
        st.image(output_style_img)

    st.title("変換後の画像")
    st.image(results)
