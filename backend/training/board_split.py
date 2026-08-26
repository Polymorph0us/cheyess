"""
board_split.py - creates (or incrementally updates) a board-level train/val/test
split with MD5 crop hashes.

Supports multiple source directories. If board_split.json already exists, only
new boards are added — existing assignments are NEVER changed.

Run from Chess-Digital-Thing/ root:
    python backend/training/board_split.py

With extra source directories:
    python backend/training/board_split.py \
        --source-dirs tempimagestore/lichess tempimagestore/chess.com tempimagestore/mygames
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline.lichess_board_detector import detect_and_crop_board
from app.pipeline.perspective import normalize_perspective
from app.pipeline.square_extraction import extract_squares

# Default source directories (all are combined into one flat board list)
DEFAULT_SOURCE_DIRS = [
    PROJECT_ROOT / "tempimagestore" / "lichess",
]
OUTPUT_PATH  = BACKEND_ROOT / "dataset" / "board_split.json"

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20


def md5_of_array(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


def get_board_crop_hashes(img_path: Path) -> list:
    """Run detection+extraction on one board image, return list of 64 MD5 hashes."""
    img_bytes = img_path.read_bytes()
    img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"  [WARN] Could not load {img_path.name}")
        return []
    board_bgr = detect_and_crop_board(img_bgr)
    board_rgb = normalize_perspective(board_bgr, target_size=800)
    squares   = extract_squares(board_rgb)
    return [md5_of_array(sq["image_np"]) for sq in squares]


def collect_images(source_dir: Path) -> list:
    if not source_dir.exists():
        return []
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def load_existing_split(output_path: Path) -> dict:
    """Load existing board_split.json if it exists, else return empty structure."""
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded existing split: {output_path}")
        existing_boards = (
            {e["name"] for e in data.get("train", [])}
            | {e["name"] for e in data.get("val", [])}
            | {e["name"] for e in data.get("test", [])}
        )
        print(f"  Already tracked: {len(existing_boards)} boards, "
              f"{len(data.get('hash_to_board', {}))} hashes")
        return data, existing_boards
    return {
        "train": [], "val": [], "test": [], "hash_to_board": {},
    }, set()


def assign_split(board_idx: int, n_train: int, n_val: int) -> str:
    """Deterministically assign a board to train/val/test by index."""
    if board_idx < n_train:
        return "train"
    elif board_idx < n_train + n_val:
        return "val"
    else:
        return "test"


def main():
    parser = argparse.ArgumentParser(
        description="Create or incrementally update the board-level train/val/test split."
    )
    parser.add_argument(
        "--source-dirs", nargs="+", type=Path,
        default=DEFAULT_SOURCE_DIRS,
        help="One or more directories containing board screenshots. "
             f"(default: {[str(d) for d in DEFAULT_SOURCE_DIRS]})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Output JSON path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=TRAIN_RATIO,
    )
    parser.add_argument(
        "--val-ratio", type=float, default=VAL_RATIO,
    )
    opts = parser.parse_args()

    # ── Collect ALL board images from all source directories ──────────────────
    all_board_paths = []
    for src_dir in opts.source_dirs:
        imgs = collect_images(src_dir)
        if imgs:
            print(f"Found {len(imgs)} images in {src_dir}")
            all_board_paths.extend(imgs)
        else:
            print(f"[INFO] No images in {src_dir} (skipping)")

    if not all_board_paths:
        print("[ERROR] No board images found in any source directory.")
        sys.exit(1)

    # Sort for deterministic ordering
    all_board_paths = sorted(all_board_paths, key=lambda p: p.name)
    n = len(all_board_paths)
    print(f"\nTotal boards found: {n}")

    # ── Load existing split (preserve existing assignments) ───────────────────
    split_data, already_tracked = load_existing_split(opts.output)

    # New boards are those not already in the split
    new_paths = [p for p in all_board_paths if p.name not in already_tracked]
    print(f"New boards to add:  {len(new_paths)}")

    if not new_paths:
        print("\nNothing to add — all boards already tracked.")
        print(f"Current split: train={len(split_data['train'])}  "
              f"val={len(split_data['val'])}  test={len(split_data['test'])}")
        return

    # Compute split boundaries for the NEW batch only
    n_new  = len(new_paths)
    n_train = round(n_new * opts.train_ratio)
    n_val   = round(n_new * opts.val_ratio)

    print(f"Assigning new boards: train={n_train}  val={n_val}  "
          f"test={n_new - n_train - n_val}")

    # ── Process new boards ────────────────────────────────────────────────────
    for idx, img_path in enumerate(new_paths):
        split_name = assign_split(idx, n_train, n_val)
        entry = {"name": img_path.name, "path": str(img_path), "hashes": []}
        print(f"  [{split_name}] {img_path.name}", end=" ... ", flush=True)

        hashes = get_board_crop_hashes(img_path)
        entry["hashes"] = hashes
        split_data[split_name].append(entry)

        for h in hashes:
            split_data["hash_to_board"][h] = {
                "board_name": img_path.name,
                "split":      split_name,
            }
        print(f"{len(hashes)} hashes")

    # ── Save ──────────────────────────────────────────────────────────────────
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    with open(opts.output, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    total_hashes = len(split_data["hash_to_board"])
    print(f"\nTotal hashes recorded: {total_hashes}")
    print(f"Saved board split to:  {opts.output}")
    print(f"\nFull split summary:")
    print(f"  Train: {len(split_data['train'])} boards")
    print(f"  Val:   {len(split_data['val'])} boards")
    print(f"  Test:  {len(split_data['test'])} boards")


if __name__ == "__main__":
    main()
