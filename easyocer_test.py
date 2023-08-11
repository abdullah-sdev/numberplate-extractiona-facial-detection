import cv2
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'D:\FYP_Projects\Old\ANPR_practisev5\resx\PyTesseract\tesseract.exe'

# Load the image
image = cv2.imread('resx/pics/ef_00 (12).jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding or other preprocessing techniques if needed
# ...

# Perform OCR using PyTesseract
text = tess.image_to_string(gray)

# Print the extracted text
print(text)
