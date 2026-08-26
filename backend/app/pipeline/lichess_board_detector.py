"""
lichess_board_detector.py
--------------------------
Detects and crops the actual 8x8 chessboard region from Lichess (and chess.com)
screenshot images.

Strategy
--------
Lichess screenshots have the board filling most of the frame with thin rank/file
label strips on the right (~rank numbers 1-8) and bottom (~file letters a-h).
Some themes also have a thin border on all sides.

Detection approach:
  1. Scan columns from the RIGHT edge inward - the label strip has low variance
     compared to the alternating square colors on the board. Stop when variance
     crosses a threshold => right edge of board.
  2. Scan rows from the BOTTOM edge upward - same logic for the bottom label strip.
  3. Scan from LEFT and TOP for any thin padding/border.
  4. Crop to those boundaries.
  5. Fall back to a conservative percentage-trim if heuristic detection fails.

The cropped region is returned as a BGR numpy array ready for
perspective.normalize_perspective().
"""

import cv2
import numpy as np


# ── tunables ──────────────────────────────────────────────────────────────────
# Minimum colour variance (per-channel std-dev summed) a column/row must have
# to be considered "inside the board" (chessboard squares alternate ⇒ high var).
VARIANCE_THRESHOLD = 8.0

# Safety margin: never trim more than this fraction of width/height.
MAX_TRIM_FRACTION = 0.12

# Fallback: trim this fraction from right & bottom if heuristics fail.
FALLBACK_TRIM = 0.04
# ──────────────────────────────────────────────────────────────────────────────


def _col_variance(img_bgr: np.ndarray, col: int) -> float:
    """Sum of per-channel std-dev for a single column."""
    col_pixels = img_bgr[:, col, :].astype(np.float32)
    return float(np.sum(np.std(col_pixels, axis=0)))


def _row_variance(img_bgr: np.ndarray, row: int) -> float:
    """Sum of per-channel std-dev for a single row."""
    row_pixels = img_bgr[row, :, :].astype(np.float32)
    return float(np.sum(np.std(row_pixels, axis=0)))


def _find_board_right(img_bgr: np.ndarray) -> int:
    """Scan from the right edge inward to find the first high-variance column."""
    h, w = img_bgr.shape[:2]
    max_trim = int(w * MAX_TRIM_FRACTION)
    for offset in range(max_trim):
        col = w - 1 - offset
        if _col_variance(img_bgr, col) > VARIANCE_THRESHOLD:
            return col + 1          # exclusive right boundary
    return w                        # fallback: no trim


def _find_board_bottom(img_bgr: np.ndarray) -> int:
    """Scan from the bottom edge upward to find the first high-variance row."""
    h, w = img_bgr.shape[:2]
    max_trim = int(h * MAX_TRIM_FRACTION)
    for offset in range(max_trim):
        row = h - 1 - offset
        if _row_variance(img_bgr, row) > VARIANCE_THRESHOLD:
            return row + 1          # exclusive bottom boundary
    return h


def _find_board_left(img_bgr: np.ndarray) -> int:
    """Scan from the left edge inward to find the first high-variance column."""
    h, w = img_bgr.shape[:2]
    max_trim = int(w * MAX_TRIM_FRACTION)
    for offset in range(max_trim):
        if _col_variance(img_bgr, offset) > VARIANCE_THRESHOLD:
            return offset
    return 0


def _find_board_top(img_bgr: np.ndarray) -> int:
    """Scan from the top edge downward to find the first high-variance row."""
    h, w = img_bgr.shape[:2]
    max_trim = int(h * MAX_TRIM_FRACTION)
    for offset in range(max_trim):
        if _row_variance(img_bgr, offset) > VARIANCE_THRESHOLD:
            return offset
    return 0


def detect_and_crop_board(image_bgr: np.ndarray) -> np.ndarray:
    """
    Detects the 8x8 chessboard region inside a Lichess screenshot and returns
    it as a cropped BGR numpy array.

    Parameters
    ----------
    image_bgr : np.ndarray
        Full screenshot in OpenCV BGR format.

    Returns
    -------
    np.ndarray
        Cropped board region (BGR). May not yet be square — caller should pass
        this into perspective.normalize_perspective() to get a clean 800x800 RGB.
    """
    h, w = image_bgr.shape[:2]

    x0 = _find_board_left(image_bgr)
    y0 = _find_board_top(image_bgr)
    x1 = _find_board_right(image_bgr)
    y1 = _find_board_bottom(image_bgr)

    # Sanity check: crop must be at least 60% of original dimensions.
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w < w * 0.60 or crop_h < h * 0.60:
        # Heuristic failed — fall back to a fixed-percentage trim.
        fx = int(w * FALLBACK_TRIM)
        fy = int(h * FALLBACK_TRIM)
        x0, y0 = 0, 0
        x1 = w - fx
        y1 = h - fy

    cropped = image_bgr[y0:y1, x0:x1]
    return cropped
