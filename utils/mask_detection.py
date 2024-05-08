from ultralytics import YOLO
import cv2
import os
import numpy as np

MASK_DETECTION_PATH = os.path.join("models", "yolov8n-mask.pt")

labels = {
            0: "with_mask",
            1: "mask_weared_incorrect",
            2: "without_mask"
        }

# function to convert file buffer to cv2 image
def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

# Make class for face detection with image and save the inference image
class MaskDetection:
    def __init__(self, img_file_buffer, device="cpu"):
        """
        Initialize the FaceDetection class.

        Args:
            image_path (str): The path to the image file.
            model_path (str): The path to the YOLOv8 model file.
            device (str, optional): The device to run the model on. Defaults to "cpu".
        """
        self.img_file_buffer = img_file_buffer
        self.device = device

    def detect_mask(self):
        """
        Detect faces in the image using YOLOv8 model.

        Returns:
            list: A list of bounding boxes representing the detected faces.
        """
        # Load the YOLOv8 model
        model = YOLO(MASK_DETECTION_PATH)

        # Read the image
        open_cv_image = create_opencv_image_from_stringio(self.img_file_buffer)


        # Run YOLOv8 on the image
        prediction = model.predict(source=open_cv_image, device=self.device, verbose=True, conf=0.6)[0]

        for result in prediction:
            # Draw bounding box
            boxes = result.boxes

            for box in boxes:
                label = box.cls.to("cpu").numpy().astype(int)[0]
                mask = labels[label]
                x1, y1, x2, y2 = np.squeeze(box.xyxy.tolist())
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(open_cv_image, f"{mask}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return open_cv_image

