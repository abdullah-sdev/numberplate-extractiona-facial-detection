import cv2
import pytesseract as tess
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
from PIL import Image
import torch

tess.pytesseract.tesseract_cmd = r'D:\FYP_Projects\Old\ANPR_practisev5\resx\PyTesseract\tesseract.exe'

# Define the database model
Base = declarative_base()


class CarData(Base):
    __tablename__ = 'car_data'

    id = Column(String, primary_key=True)
    number_plate = Column(String)
    driver_face = Column(String)
    car_picture = Column(String)
    timestamp = Column(DateTime)


# Connect to the database
# engine = create_engine('your_database_connection_string')
# Session = sessionmaker(bind=engine)
# session = Session()

# Load the pre-trained YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='ANPRwv5.pt', source='local')

# Load the class labels for YOLOv5
class_labels = []
with open('labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Define the confidence threshold and NMS threshold
confidence_threshold = 0.5
nms_threshold = 0.4

# Load the image
image_path = 'resx/pics/2.jpg'
image = cv2.imread(image_path)

# Preprocess the image
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
model.setInput(blob)

# Perform object detection
layer_names = model.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
outputs = model.forward(output_layers)

# Process the detected objects
bounding_boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > confidence_threshold:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])

            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            bounding_boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove overlapping bounding boxes
indices = cv2.dnn.NMSBoxes(bounding_boxes, confidences, confidence_threshold, nms_threshold)

# Iterate over the filtered bounding boxes and process the number plate for each
for i in indices:
    index = i[0]
    x, y, width, height = bounding_boxes[index]

    # Extract the number plate region from the image
    number_plate_region = image[y:y + height, x:x + width]

    # Apply OCR on the number plate region
    number_plate_text = tess.image_to_string(Image.fromarray(number_plate_region))

    # Store the car data in the database
    car_data = CarData(
        id='unique_id',  # Generate a unique ID for each entry
        numberplate=number_plate_text,
        driver_face='path_to_driver_face_image.jpg',  # Store the driver face image path or base64-encoded image
        car_picture=image_path,
        timestamp=datetime.now()
    )

    # session.add(car_data)
    # session.commit()