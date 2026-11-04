import os
import streamlit as st
import requests
from PIL import Image
import io
import base64

# Récupère l'URL de l'API depuis la variable d'environnement ou utilise 127.0.0.1 par défaut
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Brain Tumor AI Diagnostics", layout="wide")
st.title("🧠 Automated Brain Tumor Analysis Solution")

uploaded_file = st.file_uploader("Upload Brain MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image_bytes = uploaded_file.getvalue()

    with col1:
        st.subheader("Uploaded MRI Image")
        st.image(Image.open(io.BytesIO(image_bytes)), use_column_width=True)

    with col2:
        st.subheader("Analysis Controls")
        task = st.radio("Select Task", ["Classification (InceptionV3)", "Detection & Localization (YOLOv8)"])

        if st.button("Run Analysis"):
            files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}

            try:
                if "Classification" in task:
                    res = requests.post(f"{API_URL}/predict/classify", files=files, timeout=30)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Predicted Class: **{data['class']}** ({data['confidence']}%)")
                        st.json(data["probabilities"])
                    else:
                        st.error(f"API Error: {res.status_code}")
                else:
                    res = requests.post(f"{API_URL}/predict/detect", files=files, timeout=30)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Found {len(data['detections'])} tumor region(s)")
                        img_data = base64.b64decode(data["annotated_image_base64"])
                        st.image(Image.open(io.BytesIO(img_data)), use_column_width=True)
                    else:
                        st.error(f"API Error: {res.status_code}")

            except requests.exceptions.ConnectionError:
                st.error(f"❌ Could not connect to API server at {API_URL}. Check if backend service is running.")