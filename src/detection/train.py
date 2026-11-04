from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='config/data.yaml',
    epochs=30,
    imgsz=640,
    batch=8,         # Réduit de 16 à 8 pour économiser la RAM/VRAM
    workers=2,       # Limite l'utilisation des threads CPU
    project='runs/detect',
    name='yolov8_brain_tumor'
)
print("Entraînement de YOLOv8 terminé !")