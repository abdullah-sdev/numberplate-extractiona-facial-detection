import cv2
import easyocr
import pandas as pd
import torch
import torchvision

# Load the pre-trained YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='ANPRwv5.pt', source='local')

# Define the confidence threshold and NMS threshold
confidence_threshold = 0.5
nms_threshold = 0.4

# Load the image
image_path = 'resx/pics/5.jpg'
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Process the detected objects
bounding_boxes = results.xyxy[0][:, :4].tolist()  # Access the bounding box coordinates
confidences = results.xyxy[0][:, 4].tolist()  # Access the confidence scores
class_ids = results.xyxy[0][:, 5].tolist()  # Access the class IDs

# Convert the bounding boxes and confidences to tensors
bounding_boxes = torch.tensor(bounding_boxes)
confidences = torch.tensor(confidences)

# Apply non-maximum suppression to remove overlapping bounding boxes
indices = torchvision.ops.nms(bounding_boxes, confidences, confidence_threshold)

# Create an empty DataFrame to store the car data
car_data = pd.DataFrame(columns=['Number Plate', 'Driver Face', 'Car Picture', 'Timestamp'])

# Iterate over the filtered bounding boxes and process the number plate for each
for index in indices:
    x, y, width, height = bounding_boxes[index]

    # Extract the number plate region from the image
    number_plate_region = image[int(y):int(y + height), int(x):int(x + width)]

    # Apply OCR on the number plate region using EasyOCR
    reader = easyocr.Reader(['en'])
    number_plate_results = reader.readtext(number_plate_region)

    # Extract the number plate text from the OCR results
    number_plate_text = number_plate_results[0][1]



    # Store the car data in the DataFrame
    car_data = car_data.append({
        'Number Plate': number_plate_text,
        'Driver Face': 'path_to_driver_face_image.jpg',  # Store the driver face image path or base64-encoded image
        'Car Picture': image_path,
        'Timestamp': pd.Timestamp.now()
    }, ignore_index=True)












# Save the car data to an Excel file
car_data.to_excel('car_data.xlsx', index=False)
