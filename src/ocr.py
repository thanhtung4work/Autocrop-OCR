import os

import cv2
from easyocr import Reader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


def perform_ocr(images_dir: str, output_path: str, ocr_reader: Reader) -> None:
    """
    Create a PDF with the original image + invisible text layer
    (similar to Tesseract PDF output).
    """

    
    c = canvas.Canvas(output_path, pagesize=A4)
    page_w, page_h = A4

    for file_name in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, file_name)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping unreadable file: {file_name}")
            continue

        h, w = img.shape[:2]

        # Fit image to A4 page while keeping aspect ratio
        scale_w, scale_h = page_w / w, page_h / h
        new_w, new_h = int(w * scale_w), int(h * scale_h)

        # Add the image
        c.drawImage(
            image_path,
            0, page_h - new_h,
            width=new_w,
            height=new_h
        )

        # Run OCR
        results = ocr_reader.readtext(img, detail=1)

        # Draw invisible text layer
        c.setFillColorRGB(1, 1, 1, alpha=0.01)  # Invisible text

        for (bbox, text, conf) in results:
            pt1, pt2, pt3, pt4 = bbox
            x = pt1[0] * scale_w
            y = page_h - (pt1[1] * scale_h)

            # Draw invisible text so PDF is searchable
            c.drawString(x, y, text)

        c.showPage()

    c.save()
