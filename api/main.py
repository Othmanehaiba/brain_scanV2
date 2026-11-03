from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io
import base64

app = FastAPI(title="Brain Tumor Analysis API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["Glioma", "Meningioma", "NoTumor", "Pituitary"]

# Load models
classif_model = tf.keras.models.load_model("models/classification/inceptionv3_brain_tumor.keras")
detect_model = YOLO("models/detection/yolov8_brain_tumor.pt")


@app.post("/predict/classify")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((299, 299))
    img_array = np.expand_dims(np.array(image), axis=0)
    img_preprocessed = tf.keras.applications.inception_v3.preprocess_input(img_array)

    preds = classif_model.predict(img_preprocessed)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    return {
        "class": CLASSES[class_idx],
        "confidence": round(confidence * 100, 2),
        "probabilities": {CLASSES[i]: round(float(preds[i]), 4) for i in range(len(CLASSES))}
    }


@app.post("/predict/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = detect_model(img)[0]
    annotated_img = results.plot()

    # Convert annotated image to base64 string
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    detections = []
    for box in results.boxes:
        detections.append({
            "class": CLASSES[int(box.cls[0])],
            "confidence": round(float(box.conf[0]) * 100, 2),
            "bbox": [round(float(c), 2) for c in box.xyxy[0].tolist()]
        })

    return {
        "detections": detections,
        "annotated_image_base64": img_str
    }