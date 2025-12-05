import cv2
import numpy as np


def perspective_crop(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies a perspective transform to crop a quadrilateral region from the image.
    Expects `points` in the order:
        top-left, top-right, bottom-right, bottom-left.
    Saves the warped image and also returns it.
    """

    # Ensure points are float32
    pts = points.astype("float32")

    # Compute output dimensions by measuring opposite sides
    width_top = np.linalg.norm(pts[0] - pts[1])
    width_bottom = np.linalg.norm(pts[3] - pts[2])
    height_left = np.linalg.norm(pts[0] - pts[3])
    height_right = np.linalg.norm(pts[1] - pts[2])

    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    # Destination rectangle coordinates
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Compute perspective transform
    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped


def scale_to_a4(image):
    """
    Scales an image to fit within an A4 page (595x842 px)
    while preserving aspect ratio.
    """

    A4_W, A4_H = 595, 842  # A4 in points/pixels at 72 DPI

    h, w = image.shape[:2]

    # Calculate best uniform scale
    scale = min(A4_W / w, A4_H / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized
