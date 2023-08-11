import cv2
import pytesseract as tess
from datetime import datetime
import numpy as np
from PIL import Image
import pandas as pd
import torch

tess.pytesseract.tesseract_cmd = r'D:\FYP_Projects\Old\ANPR_practisev5\resx\PyTesseract\tesseract.exe'

# Load the pre-trained YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='ANPRwv5.pt', source='local')

# Load the class labels for YOLOv5
class_labels = []
with open('labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Define the confidence threshold
confidence_threshold = 0.5

# Load the image
image_path = 'resx/pics/5.jpg'
image = cv2.imread(image_path)

# Preprocess the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
image = Image.fromarray(image)  # Convert image to PIL format

# Perform object detection
results = model(image)

# Process the detected number plates
bounding_boxes = results.xyxy[0][:, :4].tolist()
confidences = results.xyxy[0][:, 4].tolist()

# Create an empty DataFrame to store the number plate data
number_plate_data = pd.DataFrame(columns=['Number Plate', 'Timestamp'])

# Iterate over the detected number plates
for i in range(len(bounding_boxes)):
    confidence = confidences[i]

    if confidence > confidence_threshold:
        x1, y1, x2, y2 = bounding_boxes[i]

        # Extract the number plate region from the image
        number_plate_region = image.crop((x1, y1, x2, y2))

        # Apply OCR on the number plate region
        number_plate_text = tess.image_to_string(number_plate_region)

        # Store the number plate data in the DataFrame
        number_plate_data = number_plate_data.append({
            'Number Plate': number_plate_text,
            'Timestamp': datetime.now()
        }, ignore_index=True)


# Load the existing number plate data from the Excel file
existing_data = pd.read_excel('number_plate_data.xlsx')

# Append the new number plate data to the existing data
combined_data = pd.concat([existing_data, number_plate_data], ignore_index=True)

# Save the combined data to the Excel file
combined_data.to_excel('number_plate_data.xlsx', index=False)


# # Save the number plate data to an Excel file
number_plate_data.to_excel('number_plate_data.xlsx', index=False)