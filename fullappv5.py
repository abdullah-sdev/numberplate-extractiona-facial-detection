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


def save_data_to_excel(data, file_path):
    # Create a DataFrame from the data
    number_plate_data = pd.DataFrame(data, columns=['Number Plate', 'Timestamp'])

    # Get the existing sheets in the Excel file
    existing_sheets = pd.read_excel(file_path, sheet_name=None)

    # Determine the new sheet name
    sheet_name = 'Sheet{}'.format(len(existing_sheets) + 1)

    # Save the data to a new sheet in the Excel file
    with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
        number_plate_data.to_excel(writer, index=False, sheet_name=sheet_name)



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
    video_path = 'resx/pics/VID_20230530_144000.mp4'  # Specify the path to your video file
    excel_file_path = 'number_plate_data.xlsx'
    confidence_threshold = 0.5

    tess.pytesseract.tesseract_cmd = r'D:\FYP_Projects\Old\ANPR_practisev5\resx\PyTesseract\tesseract.exe'

    # Load the pre-trained YOLOv5 model
    model = load_model(model_path)

    # Load the class labels for YOLOv5
    class_labels = load_class_labels(labels_path)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error opening video file")
        exit()

    # Read and process frames from the video
    data = []
    frame_number = 0
    while True:
        # Read the next frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Preprocess the frame
        frame_pil = preprocess_image(frame)

        # Perform object detection on the frame
        results, frame = perform_object_detection(model, frame_pil)

        # Extract the number plates and bounding box coordinates
        number_plates, bounding_boxes = extract_number_plates(results, frame, confidence_threshold)

        # Perform OCR on the number plates
        for number_plate in number_plates:
            number_plate_text = perform_ocr(number_plate)
            timestamp = datetime.now()
            data.append([number_plate_text, timestamp])

        # Display the frame with bounding boxes
        display(frame, bounding_boxes)

        frame_number += 1

    # Save the number plate data to an Excel file
    save_data_to_excel(data, excel_file_path)

    # Release the video file and close any open windows
    video_capture.release()
    cv2.destroyAllWindows()

