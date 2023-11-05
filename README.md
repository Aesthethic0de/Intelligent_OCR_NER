# Intelligent_OCR_NER

This is a Python script that extracts information from business cards using OCR and NER.

## Dependencies

- Python 3.x
- Libraries: numpy, pandas, cv2 (OpenCV), pytesseract, spacy

## How to Use

1. Install the necessary dependencies.
2. Run the script with an image file as input (e.g., `python main.py input_image.jpg`).
3. The script will process the image and display the bounding boxes around recognized entities.

## Functionality

- `main.py`: The main script that performs OCR, NER, and visualization of bounding boxes.
- `utils.py`: Contains helper functions for text cleaning, entity parsing, etc.
- `output/model-best/`: Pre-trained NER model.

## Usage Example

```python
import cv2
from business_card_parser import getPredictions

image = cv2.imread("input_image.jpg")
img_results, entities = getPredictions(image)
print(entities)
cv2.namedWindow("Predictions", cv2.WINDOW_NORMAL)
cv2.imshow("original", img_results)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Result

![Before](242.jpeg)

![After](output_image.jpeg)


