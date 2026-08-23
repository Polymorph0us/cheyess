import cv2
import numpy as np
from PIL import Image

def detect_board(image_input) -> np.ndarray:
    """
    Stage 0 Board Detection:
    Validates and normalizes input image into an OpenCV BGR numpy array.
    Future Stage 1: Classical contour/Hough corner detection & corner regressor CNN
    will be integrated here to output the 4 corner points of a physical board photo.
    """
    if isinstance(image_input, Image.Image):
        # Convert PIL Image to OpenCV RGB numpy array
        img_np = np.array(image_input.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Invalid or empty image provided.")

    return img_bgr
