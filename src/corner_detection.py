import cv2
import numpy as np


def detect_corners(image: np.ndarray) -> np.ndarray:
    """
    Detects the 4 corners of the largest rectangular contour in an image.
    Returns corner points ordered as: top-left, top-right, bottom-right, bottom-left.
    """

    # 1. Resize image (optional step for consistency)
    scale_percent = 100
    h, w = image.shape[:2]
    new_w = int(w * scale_percent / 100)
    new_h = int(h * scale_percent / 100)
    image_resized = cv2.resize(image, (new_w, new_h))

    # 2. Preprocessing: grayscale, blur, threshold
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 180)

    # 3. Find external contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 4. Detect the largest contour with 4 edges (the “page”)
    page_contour = None
    largest_area = -np.inf
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > largest_area:
            page_contour = approx
            largest_area = area

    # Fallback: if no 4-sided contour found, return whole image rectangle
    if page_contour is None:
        print("Could not detect page corners.")
        return np.array([
            [0, 0],
            [new_w, 0],
            [new_w, new_h],
            [0, new_h]
        ], dtype=np.float32)

    # 5. Extract corner points
    corner_pts = page_contour[:, 0, :].astype(np.float32)

    # 6. Sort corners clockwise using angle from center
    center = np.mean(corner_pts, axis=0)

    # Sort by angle relative to center (atan2 gives counterclockwise ordering)
    ordered = sorted(
        corner_pts,
        key=lambda pt: np.arctan2(pt[1] - center[1], pt[0] - center[0])
    )

    # Format: TL, TR, BR, BL
    ordered = np.array(ordered, dtype=np.float32)

    top_left, top_right, bottom_right, bottom_left = ordered

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
