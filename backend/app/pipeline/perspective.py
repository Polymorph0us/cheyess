import cv2
import numpy as np

TARGET_SIZE = 800

def normalize_perspective(image_bgr: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """
    Stage 0 Perspective Transform:
    Normalizes board image to a square RGB array of target_size x target_size (default 800x800).
    In Stage 1, cv2.warpPerspective with detected 4 corner points will transform angled photos.
    """
    h, w, _ = image_bgr.shape
    if h != target_size or w != target_size:
        resized_bgr = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)
    else:
        resized_bgr = image_bgr

    # Convert BGR to RGB for PIL / PyTorch compatibility
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return resized_rgb
