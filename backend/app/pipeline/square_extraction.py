from typing import List, Dict, Any
import numpy as np
from PIL import Image

FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['8', '7', '6', '5', '4', '3', '2', '1']

def extract_squares(board_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """
    Slices normalized 800x800 board image into 64 square crops (100x100 px).
    Returns list of 64 crop dictionaries in rank-major order (a8..h8, a7..h7, ..., a1..h1).
    """
    h, w, _ = board_rgb.shape
    square_h = h // 8
    square_w = w // 8

    squares = []
    for r_idx, rank_char in enumerate(RANKS):
        for f_idx, file_char in enumerate(FILES):
            y_start = r_idx * square_h
            y_end = (r_idx + 1) * square_h
            x_start = f_idx * square_w
            x_end = (f_idx + 1) * square_w

            crop_np = board_rgb[y_start:y_end, x_start:x_end]
            crop_pil = Image.fromarray(crop_np)

            squares.append({
                "rank_idx": r_idx,
                "file_idx": f_idx,
                "square_name": f"{file_char}{rank_char}",
                "image_np": crop_np,
                "image_pil": crop_pil
            })

    return squares
