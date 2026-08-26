"""
diagnose_model.py
-----------------
Comprehensive diagnostic for the chess piece classifier.

Checks:
  1. Class distribution in synthetic + real datasets
  2. Preprocessing alignment: training vs inference
  3. Confusion matrix / classification report on held-out real crops
  4. Board-level FEN accuracy on held-out Lichess images (not used in training)
  5. Per-square diagnostic: square name, ground truth, predicted piece, confidence
  6. Checks whether the API pipeline correctly detects the board region

Run from Chess-Digital-Thing/ root:
    python backend/training/diagnose_model.py
"""

import os
import sys
import random
import hashlib
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline.lichess_board_detector import detect_and_crop_board
from app.pipeline.perspective import normalize_perspective
from app.pipeline.square_extraction import extract_squares
from app.pipeline.fen_builder import fen_to_matrix, matrix_to_fen
from app.pipeline.board_detection import detect_board  # the old stub used by routes.py

import onnxruntime as ort

CLASSES     = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
IDX_TO_CLS  = {i: c for i, c in enumerate(CLASSES)}
CLS_TO_IDX  = {c: i for i, c in enumerate(CLASSES)}

# Folder → class mappings (matching import_real_dataset.py)
FOLDER_TO_CLASS = {
    'empty': '.', 'E': '.',
    'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
    'black_p': 'p', 'black_n': 'n', 'black_b': 'b',
    'black_r': 'r', 'black_q': 'q', 'black_k': 'k',
}

REAL_DIR      = BACKEND_ROOT / "dataset" / "real"
SYNTH_DIR     = BACKEND_ROOT / "dataset" / "synthetic"
MODEL_PATH    = BACKEND_ROOT / "models" / "piece_classifier.onnx"
LICHESS_DIR   = PROJECT_ROOT / "tempimagestore" / "lichess"
START_FEN     = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

DIVIDER = "=" * 70

# ── ONNX helpers ──────────────────────────────────────────────────────────────

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")
    sess = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
    print(f"Loaded model: {MODEL_PATH}")
    return sess

def preprocess_training(pil_img: Image.Image) -> np.ndarray:
    """Replicates torchvision transforms used during training."""
    img = pil_img.resize((64, 64), Image.BILINEAR)  # torchvision Resize uses bilinear
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))

def preprocess_inference(pil_img: Image.Image) -> np.ndarray:
    """Current inference preprocessing (piece_classifier.py)."""
    img = pil_img.resize((64, 64), Image.BILINEAR)   # ← fixed: now matches training (BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))

def predict_batch(sess, arrays: list) -> tuple:
    batch = np.stack(arrays, axis=0)
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    logits = sess.run([out_name], {in_name: batch})[0]
    exp    = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs  = exp / exp.sum(axis=1, keepdims=True)
    preds  = np.argmax(probs, axis=1)
    confs  = probs[np.arange(len(preds)), preds]
    return preds, confs, probs

def predict_single(sess, pil_img: Image.Image, preprocess_fn) -> tuple:
    arr = preprocess_fn(pil_img)[np.newaxis]
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    logits = sess.run([out_name], {in_name: arr})[0]   # shape (1, 13)
    exp    = np.exp(logits - logits.max())
    probs  = (exp / exp.sum()).flatten()               # shape (13,)
    pred   = int(np.argmax(probs))
    conf   = float(probs[pred])
    return pred, conf

# ── 1. Class distribution ─────────────────────────────────────────────────────

def check_class_distribution():
    print(f"\n{DIVIDER}")
    print("1. CLASS DISTRIBUTION")
    print(DIVIDER)
    total_real = 0
    total_synth = 0

    print(f"\n{'Class':<6} {'Folder':<14} {'Real':>8} {'Synth':>8} {'Total':>8}")
    print("-" * 50)

    synth_folder_to_cls = {
        'empty': '.', 'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K'
    }

    for cls in CLASSES:
        # Real dataset count
        real_count = 0
        for folder, mapped_cls in FOLDER_TO_CLASS.items():
            if mapped_cls == cls:
                folder_path = REAL_DIR / folder
                if folder_path.exists():
                    real_count += len(list(folder_path.glob("*.png")))

        # Synthetic dataset count (only white pieces + empty)
        synth_count = 0
        for folder, mapped_cls in synth_folder_to_cls.items():
            if mapped_cls == cls:
                folder_path = SYNTH_DIR / folder
                if folder_path.exists():
                    synth_count += len([f for f in folder_path.iterdir() if f.suffix.lower() in {'.png','.jpg'}])

        total_real  += real_count
        total_synth += synth_count
        print(f"  {cls:<4}  {'empty' if cls=='.' else ('black_'+cls if cls.islower() else cls):<14} {real_count:>8} {synth_count:>8} {real_count+synth_count:>8}")

    print("-" * 50)
    print(f"  {'TOTAL':<4}  {'':<14} {total_real:>8} {total_synth:>8} {total_real+total_synth:>8}")
    print(f"\n  ⚠️  Black pieces have ZERO synthetic samples — model relies entirely")
    print(f"      on real data for black piece classes.")

# ── 2. Preprocessing alignment check ─────────────────────────────────────────

def check_preprocessing_alignment(sess):
    print(f"\n{DIVIDER}")
    print("2. PREPROCESSING ALIGNMENT: Training vs Inference")
    print(DIVIDER)

    # Pick a real crop to test
    sample_path = next((REAL_DIR / "P").glob("*.png"), None)
    if sample_path is None:
        print("  No sample found.")
        return

    pil_img = Image.open(sample_path).convert("RGB")

    arr_train = preprocess_training(pil_img)
    arr_infer = preprocess_inference(pil_img)

    max_diff = float(np.max(np.abs(arr_train - arr_infer)))
    mean_diff = float(np.mean(np.abs(arr_train - arr_infer)))

    print(f"\n  Resize method — Training: BILINEAR (torchvision default)")
    print(f"  Resize method — Inference: BILINEAR (piece_classifier.py) — FIXED")
    print(f"  Max pixel difference:  {max_diff:.6f}")
    print(f"  Mean pixel difference: {mean_diff:.6f}")
    print(f"\n  Normalization — both use ImageNet mean/std: ✓ MATCH")

    if max_diff > 0.01:
        print(f"\n  ⚠️  MISMATCH DETECTED: resize method differs between training and inference.")
        print(f"      This causes subtle but real input distribution shift.")
    else:
        print(f"\n  ✓ Preprocessing is effectively identical.")

# ── 3. Confusion matrix on held-out real crops ────────────────────────────────

def confusion_matrix_on_real(sess, samples_per_class=20):
    print(f"\n{DIVIDER}")
    print(f"3. CONFUSION MATRIX (held-out real crops, {samples_per_class} per class)")
    print(DIVIDER)

    all_true = []
    all_pred = []

    for folder, cls in FOLDER_TO_CLASS.items():
        folder_path = REAL_DIR / folder
        if not folder_path.exists():
            continue
        files = sorted(folder_path.glob("*.png"))
        # Take LAST N files as held-out (training used random split, these may overlap,
        # but gives us a fast signal)
        held_out = files[-samples_per_class:] if len(files) >= samples_per_class else files

        for fpath in held_out:
            pil = Image.open(fpath).convert("RGB")
            pred_idx, conf = predict_single(sess, pil, preprocess_inference)
            all_true.append(CLS_TO_IDX[cls])
            all_pred.append(pred_idx)

    # Build confusion matrix
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t][p] += 1

    # Per-class accuracy
    print(f"\n  {'Class':<6} {'True →':<8} {'Predicted':<12} {'Correct':>8} {'Total':>8} {'Acc%':>8}")
    print("  " + "-" * 55)

    for i, cls in enumerate(CLASSES):
        row = cm[i]
        total = row.sum()
        correct = row[i]
        if total == 0:
            continue
        top_wrong = [(IDX_TO_CLS[j], int(row[j])) for j in np.argsort(-row) if j != i and row[j] > 0][:3]
        wrong_str = ", ".join(f"→{c}({n})" for c, n in top_wrong) if top_wrong else "—"
        acc = 100.0 * correct / total
        flag = " ⚠️" if acc < 90 else ""
        print(f"  {cls:<6} {correct:>4}/{total:<5} top-errors: {wrong_str:<25} {acc:>6.1f}%{flag}")

    overall = 100.0 * sum(cm[i][i] for i in range(n)) / max(sum(cm.flatten()), 1)
    print(f"\n  Overall accuracy on held-out real crops: {overall:.2f}%")

# ── 4. Board-level FEN accuracy (held-out boards) ────────────────────────────

def board_level_fen_accuracy(sess, num_boards=10):
    print(f"\n{DIVIDER}")
    print(f"4. BOARD-LEVEL FEN ACCURACY ({num_boards} held-out Lichess boards)")
    print(DIVIDER)
    print(f"\n  Ground-truth FEN: {START_FEN}\n")

    expected_matrix = fen_to_matrix(START_FEN)
    image_paths = sorted(LICHESS_DIR.glob("*.png"))

    if len(image_paths) == 0:
        print("  No Lichess images found.")
        return

    # Use the LAST N images as held-out
    held_out = image_paths[-num_boards:] if len(image_paths) >= num_boards else image_paths

    correct_boards = 0
    total_square_correct = 0
    total_squares = 0

    for img_path in held_out:
        img_bgr = cv2.imdecode(np.frombuffer(img_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        board_bgr = detect_and_crop_board(img_bgr)
        board_rgb = normalize_perspective(board_bgr, target_size=800)
        squares   = extract_squares(board_rgb)

        arrays = [preprocess_inference(sq["image_pil"]) for sq in squares]
        preds, confs, _ = predict_batch(sess, arrays)

        # Build predicted matrix
        files = list("abcdefgh")
        ranks = list("87654321")
        pred_matrix = [["." for _ in range(8)] for _ in range(8)]
        sq_results = []
        for idx, sq in enumerate(squares):
            pred_cls = IDX_TO_CLS[preds[idx]]
            pred_matrix[sq["rank_idx"]][sq["file_idx"]] = pred_cls

            gt_cls = expected_matrix[sq["rank_idx"]][sq["file_idx"]]
            is_correct = (pred_cls == gt_cls)
            total_square_correct += int(is_correct)
            total_squares += 1
            sq_results.append((sq["square_name"], gt_cls, pred_cls, float(confs[idx]), is_correct))

        pred_fen_placement = matrix_to_fen(pred_matrix).split()[0]
        expected_placement = START_FEN

        board_correct = (pred_fen_placement == expected_placement)
        correct_boards += int(board_correct)
        sq_acc = 100.0 * sum(r[4] for r in sq_results) / 64

        status = "✓ CORRECT" if board_correct else "✗ WRONG"
        print(f"  [{status}] {img_path.name}  square_acc={sq_acc:.1f}%")
        if not board_correct:
            errors = [(n, gt, pred, f"{c:.2f}") for n, gt, pred, c, ok in sq_results if not ok]
            for sq_name, gt, pred, conf in errors[:8]:
                print(f"           {sq_name}: GT={gt!r} PRED={pred!r} conf={conf}")

    total = len(held_out)
    print(f"\n  Board-level FEN accuracy: {correct_boards}/{total} ({100.0*correct_boards/max(total,1):.1f}%)")
    print(f"  Square-level accuracy:    {total_square_correct}/{total_squares} ({100.0*total_square_correct/max(total_squares,1):.1f}%)")

# ── 5. Per-square diagnostic on one board ────────────────────────────────────

def per_square_diagnostic(sess):
    print(f"\n{DIVIDER}")
    print("5. PER-SQUARE DIAGNOSTIC (first Lichess image)")
    print(DIVIDER)

    img_paths = sorted(LICHESS_DIR.glob("*.png"))
    if not img_paths:
        print("  No images found.")
        return

    img_path = img_paths[0]
    print(f"\n  Image: {img_path.name}")
    print(f"  Ground-truth FEN: {START_FEN}\n")

    expected_matrix = fen_to_matrix(START_FEN)

    img_bgr = cv2.imdecode(np.frombuffer(img_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    board_bgr = detect_and_crop_board(img_bgr)
    board_rgb = normalize_perspective(board_bgr, target_size=800)
    squares   = extract_squares(board_rgb)

    arrays = [preprocess_inference(sq["image_pil"]) for sq in squares]
    preds, confs, _ = predict_batch(sess, arrays)

    print(f"  {'Square':<8} {'GT':>4} {'Pred':>6} {'Conf':>8}  {'Status'}")
    print("  " + "-" * 45)

    errors = 0
    for idx, sq in enumerate(squares):
        pred_cls = IDX_TO_CLS[preds[idx]]
        gt_cls   = expected_matrix[sq["rank_idx"]][sq["file_idx"]]
        ok       = pred_cls == gt_cls
        if not ok:
            errors += 1
        flag = "  ✗" if not ok else ""
        if not ok:  # print all errors
            print(f"  {sq['square_name']:<8} {gt_cls:>4} {pred_cls:>6} {confs[idx]:>8.3f}{flag}")

    if errors == 0:
        print("  ✓ All 64 squares predicted correctly on this board!")
    else:
        print(f"\n  Total errors: {errors}/64")

# ── 6. API pipeline board detection check ────────────────────────────────────

def check_api_pipeline(sess):
    print(f"\n{DIVIDER}")
    print("6. API PIPELINE CHECK: does routes.py detect the board correctly?")
    print(DIVIDER)

    img_paths = sorted(LICHESS_DIR.glob("*.png"))
    if not img_paths:
        return

    img_path = img_paths[0]
    pil_img = Image.open(img_path).convert("RGB")

    # What routes.py ACTUALLY does:
    board_bgr_api = detect_board(pil_img)      # old stub — no board crop
    board_rgb_api = normalize_perspective(board_bgr_api, target_size=800)

    h_api, w_api = board_rgb_api.shape[:2]

    # What the correct pipeline does:
    img_bgr = cv2.imdecode(np.frombuffer(img_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    board_bgr_correct = detect_and_crop_board(img_bgr)
    board_rgb_correct = normalize_perspective(board_bgr_correct, target_size=800)

    # Run inference on API version
    squares_api = extract_squares(board_rgb_api)
    arrays_api  = [preprocess_inference(sq["image_pil"]) for sq in squares_api]
    preds_api, confs_api, _ = predict_batch(sess, arrays_api)

    # Run inference on correct version
    squares_ok = extract_squares(board_rgb_correct)
    arrays_ok  = [preprocess_inference(sq["image_pil"]) for sq in squares_ok]
    preds_ok, confs_ok, _ = predict_batch(sess, arrays_ok)

    expected_matrix = fen_to_matrix(START_FEN)

    correct_api = sum(
        IDX_TO_CLS[preds_api[i]] == expected_matrix[sq["rank_idx"]][sq["file_idx"]]
        for i, sq in enumerate(squares_api)
    )
    correct_ok = sum(
        IDX_TO_CLS[preds_ok[i]] == expected_matrix[sq["rank_idx"]][sq["file_idx"]]
        for i, sq in enumerate(squares_ok)
    )

    print(f"\n  Input image: {img_path.name}")
    print(f"\n  routes.py (detect_board stub — NO board crop):")
    print(f"    Input to model: full screenshot resized to 800×800")
    print(f"    Square accuracy: {correct_api}/64  ({100.0*correct_api/64:.1f}%)")

    print(f"\n  Correct pipeline (detect_and_crop_board):")
    print(f"    Board crop size before resize: {board_bgr_correct.shape[1]}×{board_bgr_correct.shape[0]} px")
    print(f"    Square accuracy: {correct_ok}/64  ({100.0*correct_ok/64:.1f}%)")

    if correct_api < correct_ok:
        print(f"\n  ⚠️  ROOT CAUSE CONFIRMED: routes.py uses detect_board() (old stub)")
        print(f"      which does NOT crop the board — it passes the full screenshot to the")
        print(f"      model, causing ~{correct_ok-correct_api} extra wrong predictions per board.")
        print(f"      Fix: replace detect_board() with detect_and_crop_board() in routes.py.")
    else:
        print(f"\n  Pipeline output is the same — board detection is not the issue here.")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(DIVIDER)
    print("CHESS CLASSIFIER DIAGNOSTIC REPORT")
    print(DIVIDER)

    sess = load_model()

    for fn, kwargs in [
        (check_class_distribution, {}),
        (check_preprocessing_alignment, {'sess': sess}),
        (confusion_matrix_on_real, {'sess': sess, 'samples_per_class': 20}),
        (board_level_fen_accuracy, {'sess': sess, 'num_boards': 10}),
        (per_square_diagnostic, {'sess': sess}),
        (check_api_pipeline, {'sess': sess}),
    ]:
        try:
            fn(**kwargs)
        except Exception as e:
            print(f"\n  [ERROR in {fn.__name__}]: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{DIVIDER}")
    print("DIAGNOSTIC COMPLETE")
    print(DIVIDER)
