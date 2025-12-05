import os

import cv2
import easyocr

from src import detect_corners, perspective_crop, quantize_image, perform_ocr


RAW_IMAGES_DIR = "data/raw_images"
PROCESSED_DIR = "data/processed"
OCR_OUTPUT = "data/final_document.pdf"

reader = easyocr.Reader(['vi'])

def process_images():
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for file_name in sorted(os.listdir(RAW_IMAGES_DIR)):

        input_path = os.path.join(RAW_IMAGES_DIR, file_name)
        output_path = os.path.join(PROCESSED_DIR, file_name)

        # Load raw image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Skipping unreadable file: {file_name}")
            continue

        # 1. Detect corners
        points = detect_corners(image)

        # 2. Apply perspective correction
        cropped_img = perspective_crop(image, points)

        # 3. Apply color_quantization
        quantized_img = quantize_image(cropped_img, 4)
        cv2.imwrite(output_path, quantized_img)
        

        print(f"Processed {file_name}")


def main():
    print("Starting image preprocessing...")
    process_images()

    print("Running OCR on processed images...")
    perform_ocr(PROCESSED_DIR, OCR_OUTPUT, reader)

    print(f"OCR complete. Output saved to: {OCR_OUTPUT}")


if __name__ == "__main__":
    main()
