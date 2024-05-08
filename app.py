from fastapi import FastAPI, UploadFile, File
from PIL import Image
from ultralytics import YOLO
import io
import uvicorn
import os

labels = {
            0: "with_mask",
            1: "mask_weared_incorrect",
            2: "without_mask"
        }

YOLO_MODEL = os.path.join("models", "yolov8n-mask.pt")

detector = YOLO(YOLO_MODEL, task='detect')

def ResponseModel(data, message):
    return {
        "data": [data],
        "code": 200,
        "message": message,
    }

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Mask Detection API"}

@app.post("/detect_mask")
async def detect_logo(image: UploadFile = File(...)):
    # Read the uploaded image
    img = Image.open(io.BytesIO(await image.read()))

    if img is not None:
        # Perform logo detection using YOLO model
        prediction = detector.predict(img, conf=0.25, verbose=False, device='cpu', stream=True)

        # Extract the detected logos
        masks = []
        for result in prediction:
            boxes = result.boxes
            for box in boxes:
                logo = int(box.cls.to("cpu").numpy())
                masks.append(logo)

        # Use a list comprehension to convert the class values to class names
        class_names = [labels[val] for val in masks]

        return ResponseModel(class_names, "Logo detection successful")
    
    else:
        return ResponseModel(None, "No image uploaded")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)