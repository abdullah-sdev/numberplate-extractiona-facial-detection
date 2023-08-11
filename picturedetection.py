import torch
import easyocr
# import cv2

# Images
pathImg = 'resx/pics/'
# imgs = [f'{pathImg}1.jpg', f'{pathImg}2.jpg']     # or file, Path, PIL, OpenCV, numpy, list
# imgs = ["https://ultralytics.com/images/bus.jpg", f'{pathImg}1.jpg', f'{pathImg}2.jpg']
imgs = [f'{pathImg}5.jpg']
#imgs =

# WebCam
# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)






# Model
# model = torch.hub.load("ultralytics/yolov5", "yolov5n")  # or yolov5n - yolov5x6, custom
# model = torch.hub.load('ultralytics/yolov5', 'custom', 'ANPRwv5.pt')  # custom/local model
model = torch.hub.load('yolov5', 'custom', path='ANPRwv5.pt', source='local')
model.conf = 0.50  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference
model.cpu()  # CPU
# model.cuda()  # GPU

results = model(imgs, size=640)

# Inference

# while True:
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             cv2.rectangle(imgs, (x1, y1), (x2, y2), (255, 0, 255), 3)
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#     cv2.imshow("Image", imgs)
#     cv2.waitKey(1)


i = 0
for img in imgs:
    labels, cord_thres = results.xyxyn[i][:, -1].numpy(), results.xyxyn[i][:, :-1].numpy()
    print('Labels: ')
    print(labels)
    print('Cord_Thres: ')
    print('xmin \t\t ymin \t\t xmax \t ymax \t\t conf ')
    print(cord_thres)
    i = i + 1


#EasyOCR
reader = easyocr.Reader(['en'], gpu=False)
lisenceplate = reader.readtext(imgs)
print(lisenceplate)

# Results
# results.crop()  # or .show(), .save(), .crop(), .pandas(), .print() etc.


# print(results.pandas().xyxy[0])
