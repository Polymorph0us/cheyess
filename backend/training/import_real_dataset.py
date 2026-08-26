"""
import_real_dataset.py
----------------------
Automated pipeline to import real Lichess (and chess.com) chessboard screenshots
into the labelled square-crop dataset used for classifier training.

Usage
-----
Run from the repository root (Chess-Digital-Thing/):

    python backend/training/import_real_dataset.py

Modes
-----
Mode A — Manifest (per-image FENs, recommended for mixed positions):

    python backend/training/import_real_dataset.py --manifest tempimagestore/lichess/manifest.csv

  The manifest is a CSV or JSON file next to your screenshots.
  CSV format (one row per image):

      filename,fen
      my_game1.png,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
      my_game2.png,r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R

  JSON format:

      [{"file": "my_game1.png", "fen": "rnbqkbnr/..."},
       {"file": "my_game2.png", "fen": "r1bqkb1r/..."}]

  Images are resolved relative to the manifest file's directory by default.
  Use --source-dir to override the image search directory.

Mode B — Legacy (single FEN for a whole directory, existing behaviour):

    python backend/training/import_real_dataset.py --source-lichess DIR --fen FEN

Optional flags (all modes):
    --output           DIR   Root output directory
                             (default: backend/dataset/real)
    --debug-count      N     How many debug preview grids to generate (default: 5)
    --no-debug               Skip debug preview generation

Mode A additional flags:
    --manifest FILE          CSV or JSON file mapping filename -> FEN
    --source-dir DIR         Directory containing the images listed in the manifest
                             (default: same directory as the manifest file)

Mode B additional flags:
    --source-lichess   DIR   Directory of Lichess screenshots
                             (default: tempimagestore/lichess)
    --source-chesscom  DIR   Directory of chess.com screenshots
                             (default: tempimagestore/chess.com — skipped if empty)
    --fen              FEN   Ground-truth FEN for ALL images in Mode B
                             (default: starting position)

Idempotency / safety
--------------------
Each crop is saved with a filename derived from the MD5 hash of its pixel data.
Re-running the script will NOT create duplicates — existing files are skipped.

Output structure
----------------
backend/dataset/real/
├── empty/
├── P/ N/ B/ R/ Q/ K/
├── p/ n/ b/ r/ q/ k/
└── debug_previews/
    └── board_NNNNN.png   (annotated 8x8 grid for visual verification)
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── resolve project root so we can import from backend/app/pipeline ──────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Chess-Digital-Thing/
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline.lichess_board_detector import detect_and_crop_board
from app.pipeline.perspective import normalize_perspective
from app.pipeline.square_extraction import extract_squares
from app.pipeline.fen_builder import fen_to_matrix

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_LICHESS_SRC = PROJECT_ROOT / "tempimagestore" / "lichess"
DEFAULT_CHESSCOM_SRC = PROJECT_ROOT / "tempimagestore" / "chess.com"
DEFAULT_OUTPUT = BACKEND_ROOT / "dataset" / "real"
DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# All 13 classes (uses 'empty' for '.'; 'black_X' prefix for lowercase pieces
# to avoid Windows case-insensitive filesystem collisions between e.g. 'P' and 'p').
CLASSES = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
CLASS_TO_FOLDER = {
    ".": "empty",
    "P": "P",  "N": "N",  "B": "B",  "R": "R",  "Q": "Q",  "K": "K",
    "p": "black_p", "n": "black_n", "b": "black_b",
    "r": "black_r", "q": "black_q", "k": "black_k",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def md5_of_array(arr: np.ndarray) -> str:
    """Stable MD5 hash of a numpy array's raw bytes."""
    return hashlib.md5(arr.tobytes()).hexdigest()


def collect_images(source_dir: Path) -> List[Path]:
    """Return sorted list of image paths from a directory (non-recursive)."""
    if not source_dir.exists():
        return []
    paths = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    return paths


def load_manifest(manifest_path: Path) -> Dict[str, str]:
    """
    Load a manifest CSV or JSON file and return a dict mapping
    image filename (basename only) -> FEN string.

    CSV format:  filename,fen   (header row required)
    JSON format: [{"file": "...", "fen": "..."}, ...]
    """
    suffix = manifest_path.suffix.lower()
    mapping: Dict[str, str] = {}

    if suffix == ".json":
        with open(manifest_path, encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError("JSON manifest must be a list of {file, fen} objects.")
        for entry in entries:
            fname = Path(entry["file"]).name  # accept full paths or bare filenames
            mapping[fname] = entry["fen"]

    else:  # treat as CSV (default, including .csv and anything else)
        with open(manifest_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not {"filename", "fen"}.issubset(
                {h.strip().lower() for h in reader.fieldnames}
            ):
                raise ValueError(
                    "CSV manifest must have 'filename' and 'fen' columns. "
                    f"Found columns: {reader.fieldnames}"
                )
            for row in reader:
                # Case-insensitive column lookup
                fname_key = next(k for k in row if k.strip().lower() == "filename")
                fen_key   = next(k for k in row if k.strip().lower() == "fen")
                fname = Path(row[fname_key].strip()).name
                fen   = row[fen_key].strip()
                if fname and fen:
                    mapping[fname] = fen

    return mapping


def resolve_sources_from_manifest(
    manifest_path: Path,
    source_dir: Optional[Path],
    fallback_fen: str,
) -> List[Tuple[Path, str]]:
    """
    Given a manifest file, return a list of (image_path, fen) tuples.
    Images are resolved from source_dir (or the manifest's directory if not given).
    Images listed in the manifest but not found on disk are skipped with a warning.
    Images in source_dir NOT listed in the manifest get fallback_fen.
    """
    image_dir = source_dir if source_dir is not None else manifest_path.parent
    fen_map   = load_manifest(manifest_path)

    print(f"Loaded manifest: {manifest_path.name}  ({len(fen_map)} entries)")
    print(f"Image directory: {image_dir}")

    # All images in the directory
    all_images = collect_images(image_dir)
    sources: List[Tuple[Path, str]] = []

    unlisted = 0
    for img_path in all_images:
        fen = fen_map.get(img_path.name)
        if fen is None:
            # Image exists on disk but not in manifest: skip with warning
            unlisted += 1
            print(f"  [SKIP] {img_path.name} not in manifest — use --fen to process unlisted images")
            continue
        sources.append((img_path, fen))

    # Warn about manifest entries that have no matching file
    missing_files = set(fen_map.keys()) - {p.name for p in all_images}
    for fname in sorted(missing_files):
        print(f"  [WARN] Manifest entry '{fname}' not found in {image_dir}")

    return sources


def fen_to_square_labels(fen: str) -> dict:
    """
    Parse a FEN and return a dict mapping square_name -> piece_symbol.
    e.g. {'a8': 'r', 'b8': 'n', ..., 'e1': 'K', ...}
    """
    matrix = fen_to_matrix(fen)   # 8x8 list[list[str]], row 0 = rank 8
    files = list("abcdefgh")
    ranks = list("87654321")

    labels = {}
    for r_idx, rank_char in enumerate(ranks):
        for f_idx, file_char in enumerate(files):
            piece = matrix[r_idx][f_idx]
            labels[f"{file_char}{rank_char}"] = piece
    return labels


def make_debug_preview(
    board_rgb: np.ndarray,
    square_labels: dict,
    source_name: str,
) -> Image.Image:
    """
    Creates an annotated 8x8 grid image showing square names overlaid on the
    board for visual verification that detection and orientation are correct.
    """
    h, w = board_rgb.shape[:2]
    sq_h = h // 8
    sq_w = w // 8

    pil_board = Image.fromarray(board_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_board.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    files = list("abcdefgh")
    ranks = list("87654321")

    # Try to load a small font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", size=max(10, sq_h // 6))
    except Exception:
        font = ImageFont.load_default()

    for r_idx, rank_char in enumerate(ranks):
        for f_idx, file_char in enumerate(files):
            sq_name = f"{file_char}{rank_char}"
            piece = square_labels.get(sq_name, "?")
            label = f"{sq_name}\n{piece}"

            x0 = f_idx * sq_w
            y0 = r_idx * sq_h
            x1 = x0 + sq_w
            y1 = y0 + sq_h

            # Draw grid lines
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 80, 80, 200), width=2)

            # Draw text
            tx, ty = x0 + 4, y0 + 4
            draw.text((tx + 1, ty + 1), label, font=font, fill=(0, 0, 0, 200))
            draw.text((tx, ty), label, font=font, fill=(255, 255, 80, 230))

    # Title bar
    title_h = max(30, sq_h // 4)
    title_img = Image.new("RGBA", (w, title_h), (30, 30, 30, 240))
    title_draw = ImageDraw.Draw(title_img)
    try:
        title_font = ImageFont.truetype("arial.ttf", size=max(12, title_h // 2))
    except Exception:
        title_font = ImageFont.load_default()
    title_draw.text((8, 4), f"Board: {source_name}", font=title_font, fill=(200, 255, 200, 255))

    composite = Image.alpha_composite(pil_board, overlay).convert("RGB")
    final = Image.new("RGB", (w, h + title_h), (30, 30, 30))
    final.paste(title_img.convert("RGB"), (0, 0))
    final.paste(composite, (0, title_h))
    return final


def process_image(
    img_path: Path,
    square_labels: dict,
    output_dir: Path,
    debug_dir: Path | None,
    generate_debug: bool,
) -> Tuple[dict, bool]:
    """
    Process a single screenshot:
      1. Load & detect board region
      2. Normalize to 800x800 RGB
      3. Extract 64 square crops
      4. Save each crop to the appropriate class folder (skip if hash exists)
      5. Optionally save a debug preview

    Returns (counts_dict, debug_was_saved).
    counts_dict maps class_folder -> number of NEW crops saved.
    """
    counts = {folder: 0 for folder in CLASS_TO_FOLDER.values()}
    skipped = 0

    # ── 1. Load image ────────────────────────────────────────────────────────
    img_bgr = cv2.imdecode(
        np.frombuffer(img_path.read_bytes(), np.uint8),
        cv2.IMREAD_COLOR,
    )
    if img_bgr is None:
        print(f"  [WARN] Could not load {img_path.name}, skipping.")
        return counts, False

    # ── 2. Detect & crop board region ────────────────────────────────────────
    board_bgr = detect_and_crop_board(img_bgr)

    # ── 3. Normalize to 800x800 RGB ──────────────────────────────────────────
    board_rgb = normalize_perspective(board_bgr, target_size=800)

    # ── 4. Extract 64 squares ────────────────────────────────────────────────
    squares = extract_squares(board_rgb)

    # ── 5. Save crops ────────────────────────────────────────────────────────
    for sq in squares:
        sq_name = sq["square_name"]
        piece = square_labels.get(sq_name, ".")
        folder_name = CLASS_TO_FOLDER[piece]

        dest_dir = output_dir / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        crop_np = sq["image_np"]  # H x W x 3 RGB uint8
        file_hash = md5_of_array(crop_np)
        dest_path = dest_dir / f"{file_hash}.png"

        if dest_path.exists():
            skipped += 1
            continue

        pil_crop = Image.fromarray(crop_np)
        pil_crop.save(str(dest_path))
        counts[folder_name] += 1

    # ── 6. Debug preview ─────────────────────────────────────────────────────
    debug_saved = False
    if generate_debug and debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        preview = make_debug_preview(board_rgb, square_labels, img_path.name)
        stem = img_path.stem.replace(" ", "_")
        preview_path = debug_dir / f"debug_{stem}.png"
        preview.save(str(preview_path))
        debug_saved = True

    return counts, debug_saved


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import real Lichess/chess.com screenshots into the training dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Mode A: manifest with per-image FENs (recommended for mixed positions):
          python backend/training/import_real_dataset.py --manifest tempimagestore/lichess/manifest.csv

          # Mode B: single FEN for a whole directory (legacy, unchanged):
          python backend/training/import_real_dataset.py --source-lichess tempimagestore/lichess --fen STARTPOS_FEN

          # Default run (starting position, all lichess images):
          python backend/training/import_real_dataset.py
        """),
    )
    # ── Mode A: manifest ──────────────────────────────────────────────────────
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="CSV or JSON file mapping image filename -> FEN (enables per-image FEN mode)",
    )
    parser.add_argument(
        "--source-dir", type=Path, default=None,
        help="Directory containing images listed in the manifest (default: manifest's directory)",
    )
    # ── Mode B: legacy directory + single FEN ────────────────────────────────
    parser.add_argument("--source-lichess", type=Path, default=DEFAULT_LICHESS_SRC,
                        help=f"Directory of Lichess screenshots (default: {DEFAULT_LICHESS_SRC})")
    parser.add_argument("--source-chesscom", type=Path, default=DEFAULT_CHESSCOM_SRC,
                        help=f"Directory of chess.com screenshots (default: {DEFAULT_CHESSCOM_SRC})")
    parser.add_argument("--fen", type=str, default=DEFAULT_FEN,
                        help="Ground-truth FEN for all images (Mode B) or fallback FEN (Mode A)")
    # ── Shared ────────────────────────────────────────────────────────────────
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output root directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--debug-count", type=int, default=5,
                        help="Number of debug preview images to generate (default: 5)")
    parser.add_argument("--no-debug", action="store_true",
                        help="Skip generating debug preview images")
    args = parser.parse_args()

    # ── Build source list ─────────────────────────────────────────────────────
    # Each entry is (image_path, fen_string)
    sources: List[Tuple[Path, str]] = []

    if args.manifest is not None:
        # ── Mode A: per-image FENs from manifest ─────────────────────────────
        if not args.manifest.exists():
            print(f"[ERROR] Manifest not found: {args.manifest}")
            sys.exit(1)
        sources = resolve_sources_from_manifest(
            manifest_path=args.manifest,
            source_dir=args.source_dir,
            fallback_fen=args.fen,
        )
    else:
        # ── Mode B: legacy single-FEN directory scan ──────────────────────────
        lichess_imgs = collect_images(args.source_lichess)
        if lichess_imgs:
            sources.extend((p, args.fen) for p in lichess_imgs)
            print(f"Found {len(lichess_imgs)} Lichess screenshots in {args.source_lichess}")
        else:
            print(f"[INFO] No Lichess images found in {args.source_lichess}")

        chesscom_imgs = collect_images(args.source_chesscom)
        if chesscom_imgs:
            sources.extend((p, args.fen) for p in chesscom_imgs)
            print(f"Found {len(chesscom_imgs)} chess.com screenshots in {args.source_chesscom}")
        else:
            print(f"[INFO] No chess.com images found in {args.source_chesscom} (skipping)")

    if not sources:
        print("\n[ERROR] No source images found. Nothing to do.")
        sys.exit(1)

    total_images = len(sources)
    print(f"\nTotal images to process: {total_images}")
    print(f"Output directory       : {args.output}")
    if args.manifest is None:
        print(f"Ground-truth FEN       : {args.fen}")
    else:
        print(f"Manifest               : {args.manifest.name}  (per-image FENs)")
    print(f"Debug previews         : {'disabled' if args.no_debug else args.debug_count}")
    print("-" * 60)

    # ── Ensure all class output dirs exist ───────────────────────────────────
    for folder in CLASS_TO_FOLDER.values():
        (args.output / folder).mkdir(parents=True, exist_ok=True)

    debug_dir = args.output / "debug_previews" if not args.no_debug else None

    # ── Process each image ───────────────────────────────────────────────────
    total_counts = {folder: 0 for folder in CLASS_TO_FOLDER.values()}
    debug_generated = 0

    for img_idx, (img_path, img_fen) in enumerate(sources, start=1):
        want_debug = (
            not args.no_debug
            and debug_generated < args.debug_count
        )
        print(f"  [{img_idx:03d}/{total_images:03d}] {img_path.name}", end=" ... ")

        square_labels = fen_to_square_labels(img_fen)

        try:
            counts, did_debug = process_image(
                img_path=img_path,
                square_labels=square_labels,
                output_dir=args.output,
                debug_dir=debug_dir,
                generate_debug=want_debug,
            )
            new_crops = sum(counts.values())
            print(f"{new_crops} new crops saved" + (" | debug preview saved" if did_debug else ""))
            for folder, n in counts.items():
                total_counts[folder] += n
            if did_debug:
                debug_generated += 1
        except Exception as exc:
            print(f"ERROR: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    grand_total = sum(total_counts.values())
    print("\n" + "=" * 60)
    print("DATASET IMPORT COMPLETE")
    print("=" * 60)
    print(f"{'Class':<10}  {'Folder':<10}  {'New Crops':>10}")
    print("-" * 35)

    ordered_display = [
        (".",  "empty"),
        ("P",  "P"),       ("N",  "N"),       ("B",  "B"),
        ("R",  "R"),       ("Q",  "Q"),       ("K",  "K"),
        ("p",  "black_p"), ("n",  "black_n"), ("b",  "black_b"),
        ("r",  "black_r"), ("q",  "black_q"), ("k",  "black_k"),
    ]
    for piece, folder in ordered_display:
        n = total_counts[folder]
        print(f"  {piece:<8}  {folder:<12}  {n:>10}")

    print("-" * 35)
    print(f"  {'TOTAL':<8}  {'':10}  {grand_total:>10}")
    print("=" * 60)

    if debug_generated:
        print(f"\nDebug previews saved to: {debug_dir}")
    print(f"\nDataset root: {args.output}")
    print("\nTo train on synthetic + real data combined, run:")
    print("  python backend/training/train_classifier.py")


if __name__ == "__main__":
    main()
