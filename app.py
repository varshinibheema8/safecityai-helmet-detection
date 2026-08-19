import streamlit as st
import torch
from PIL import Image
import numpy as np

st.title("SafeCityAI - Helmet Violation Detector")
st.write("Upload a traffic image and the model will flag riders without helmets.")

# loading model once, takes a few seconds
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', trust_repo=True)

img_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if img_file is not None:
    img = Image.open(img_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    conf_thresh = st.slider('Confidence Threshold', 0.0, 1.0, 0.5)

    run = st.button('Run Detection')
    if run:
        results = model(img)
        df = results.pandas().xyxy[0]
        df = df[df['confidence'] >= conf_thresh]

        out_img = np.squeeze(results.render())
        st.image(out_img, caption='Detection Result', use_column_width=True)

        st.write("Violations detected:", len(df))

        for i in range(len(df)):
            row = df.iloc[i]
            st.write(row['name'], "-", round(float(row['confidence']), 2))