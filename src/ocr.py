import os
import io
import cv2
import pytesseract
from PyPDF2 import PdfMerger


def perform_ocr(images_dir: str, output_path: str) -> None:
    """
    Runs OCR on every image in a directory and merges all results into a single PDF.

    Parameters:
        images_dir (str): Directory containing image files.
        output_path (str): Where the final merged PDF should be saved.
    """

    merger = PdfMerger()

    # Sort files to maintain predictable order (optional but recommended)
    image_files = sorted(os.listdir(images_dir))

    for file_name in image_files:
        image_path = os.path.join(images_dir, file_name)

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping non-image or unreadable file: {file_name}")
            continue

        # Run OCR: Tesseract returns a PDF as bytes
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')
        pdf_stream = io.BytesIO(pdf_bytes)

        # Append to the PDF merger
        merger.append(pdf_stream)

    # Write final merged PDF
    merger.write(output_path)
    merger.close()
