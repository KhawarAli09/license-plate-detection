from ultralytics import YOLO
import easyocr
import cv2
import matplotlib.pyplot as plt
import argparse
import os

def detect_license_plate(image_path, weights_path='weights/best.pt'):
    # Load YOLO model
    model = YOLO(weights_path)

    # Load EasyOCR
    reader = easyocr.Reader(['en'])

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Detect license plate
    results = model(image)

    print(f"\nProcessing: {image_path}")
    print(f"Plates detected: {len(results[0].boxes)}")

    # Process each detected plate
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop license plate
        plate = image[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        # Preprocess
        plate = cv2.resize(plate, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # OCR
        result = reader.readtext(
            thresh,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        )

        print(f"\n--- Plate {i+1} ---")
        if result:
            for detection in result:
                print(f"  Text:       {detection[1]}")
                print(f"  Confidence: {round(detection[2], 2)}")
        else:
            print("  No text found.")

        # Show processed plate
        plt.figure(figsize=(8, 3))
        plt.title(f"Plate {i+1}")
        plt.imshow(thresh, cmap='gray')
        plt.axis("off")
        plt.show()

    # Show annotated full image
    annotated = results[0].plot(font_size=10, line_width=1)
    plt.figure(figsize=(10, 6))
    plt.title("License Plate Detection")
    plt.imshow(annotated)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='License Plate Detection + OCR')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='Path to YOLO weights')
    args = parser.parse_args()

    detect_license_plate(args.image, args.weights)