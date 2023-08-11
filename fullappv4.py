import cv2
import pytesseract as tess
from datetime import datetime
import numpy as np
from PIL import Image
import pandas as pd
import torch
import matplotlib.pyplot as plt


def load_model(model_path):
    # Load the pre-trained YOLOv5 model
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
    return model


def load_class_labels(labels_path):
    # Load the class labels for YOLOv5
    class_labels = []
    with open(labels_path, 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]
    return class_labels


def preprocess_image(image):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert image to PIL format
    image_pil = Image.fromarray(image_rgb)
    return image_pil


def perform_object_detection(model, image):
    # Perform object detection
    results = model(image)
    return results, np.array(image)


def extract_number_plates(results, image, confidence_threshold=0.5):
    # Process the detected number plates
    bounding_boxes = results.xyxy[0][:, :4].tolist()
    confidences = results.xyxy[0][:, 4].tolist()

    number_plates = []
    plate_bounding_boxes = []  # Store the bounding box coordinates

    for i in range(len(bounding_boxes)):
        confidence = confidences[i]

        if confidence > confidence_threshold:
            x1, y1, x2, y2 = map(int, bounding_boxes[i])

            # Convert the PIL Image to a NumPy array
            image_np = np.array(image)

            # Crop the number plate region from the image
            number_plate_region = image_np[y1:y2, x1:x2]
            number_plates.append(number_plate_region)

            # Store the bounding box coordinates
            plate_bounding_boxes.append((x1, y1, x2, y2))

    return number_plates, plate_bounding_boxes




def perform_ocr(image):
    # Apply OCR on the image
    number_plate_text = tess.image_to_string(image)
    return number_plate_text


# def save_data_to_excel(data, file_path):
#     # Create a DataFrame from the data
#     number_plate_data = pd.DataFrame(data, columns=['Number Plate', 'Timestamp'])
#     # Append the data to the existing Excel file
#     with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#         number_plate_data.to_excel(writer, index=False, header=not writer.sheets)

# def save_data_to_excel(data, file_path):
#     try:
#         # Check if the file already exists
#         existing_data = pd.read_excel(file_path)
#         # Append the new data to the existing data
#         number_plate_data = pd.concat([existing_data, pd.DataFrame(data, columns=['Number Plate', 'Timestamp'])])
#     except FileNotFoundError:
#         # If the file doesn't exist, create a new DataFrame
#         number_plate_data = pd.DataFrame(data, columns=['Number Plate', 'Timestamp'])
#
#     # Save the data to the Excel file
#     with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
#         number_plate_data.to_excel(writer, index=False, header=not writer.sheets)

def save_data_to_excel(data, file_path):
    try:
        # Check if the file already exists
        existing_data = pd.read_excel(file_path)
        # Append the new data to the existing data
        number_plate_data = pd.concat([existing_data, pd.DataFrame(data, columns=['Number Plate', 'Timestamp'])])

        # Save the combined data to a new sheet
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
            number_plate_data.to_excel(writer, index=False, sheet_name='Sheet1', header=not writer.sheets)
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        number_plate_data = pd.DataFrame(data, columns=['Number Plate', 'Timestamp'])

        # Save the data to the Excel file
        number_plate_data.to_excel(file_path, index=False)


def display(image, bounding_boxes):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for (x1, y1, x2, y2) in bounding_boxes:
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Paths and configurations
    model_path = 'ANPRwv5.pt'
    labels_path = 'labels.txt'
    image_path = 'resx/pics/79.png'
    excel_file_path = 'number_plate_data.xlsx'
    confidence_threshold = 0.1

    tess.pytesseract.tesseract_cmd = r'D:\FYP_Projects\Old\ANPR_practisev5\resx\PyTesseract\tesseract.exe'

    # Load the pre-trained YOLOv5 model
    model = load_model(model_path)

    # Load the class labels for YOLOv5
    class_labels = load_class_labels(labels_path)

    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image
    image_pil = preprocess_image(image)

    # Perform object detection
    results, image = perform_object_detection(model, image_pil)

    # Extract the number plates and bounding box coordinates
    number_plates, bounding_boxes = extract_number_plates(results, image, confidence_threshold)

    # Extract the number plates
    # number_plates = extract_number_plates(results, image, confidence_threshold)

    # Perform OCR on the number plates
    data = []
    bounding_boxes = []
    for number_plate in number_plates:
        number_plate_text = perform_ocr(number_plate)
        timestamp = datetime.now()
        data.append([number_plate_text, timestamp])
        bounding_boxes.append(number_plate)  # Add the bounding box coordinates

    # Save the number plate data to an Excel file
    save_data_to_excel(data, excel_file_path)

    # Display the image
    display(image, bounding_boxes)


