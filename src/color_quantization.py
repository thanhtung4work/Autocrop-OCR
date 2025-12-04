import cv2
import numpy as np

def quantize_image(image, k=8):
    """
    Reduce the number of colors in the image using K-means clustering.
    k = number of colors to keep.
    """
    # Convert to float32 and reshape to (num_pixels, 3)
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    # KMeans criteria: stop after 10 iterations or accuracy < 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Apply KMeans
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert colors back to uint8
    centers = np.uint8(centers)

    # Apply clustered colors to pixels
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(image.shape)

    return quantized
