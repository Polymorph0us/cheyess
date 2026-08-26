"""
evaluate_classifier.py
----------------------
Proper evaluation of the chess piece classifier on the held-out TEST boards.

Metrics reported:
  - Exact FEN accuracy   (board must be 100% correct)
  - Square-level accuracy
  - Per-class accuracy   (all 13 classes)
  - Full 13x13 confusion matrix
  - Q->q and k->q error investigation
  - Misclassified crops saved to backend/dataset/misclassified_crops/

Run from Chess-Digital-Thing/ root:
    python backend/training/evaluate_classifier.py
"""

import sys
import json
from pathlib import Path

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

import onnxruntime as ort

CLASSES    = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
IDX_TO_CLS = {i: c for i, c in enumerate(CLASSES)}
CLS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

MODEL_PATH       = BACKEND_ROOT / "models" / "piece_classifier.onnx"
SPLIT_PATH       = BACKEND_ROOT / "dataset" / "board_split.json"
MISCLASSIFIED_DIR = BACKEND_ROOT / "dataset" / "misclassified_crops"
START_FEN        = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

DIVIDER = "=" * 70


# ── Preprocessing (BILINEAR — matches training) ───────────────────────────────

def preprocess(pil_img: Image.Image) -> np.ndarray:
    """Resize BILINEAR to match torchvision.transforms.Resize default in training."""
    img = pil_img.resize((64, 64), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))


def predict_batch(sess, pil_images: list) -> tuple:
    arrays = np.stack([preprocess(img) for img in pil_images], axis=0)
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    logits = sess.run([out_name], {in_name: arrays})[0]
    exp    = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs  = exp / exp.sum(axis=1, keepdims=True)
    preds  = np.argmax(probs, axis=1)
    confs  = probs[np.arange(len(preds)), preds]
    return preds, confs, probs


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    print(DIVIDER)
    print("CHESS CLASSIFIER EVALUATION REPORT")
    print(DIVIDER)

    # Load model
    if not MODEL_PATH.exists():
        print(f"[ERROR] ONNX model not found: {MODEL_PATH}")
        sys.exit(1)
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    print(f"Loaded model: {MODEL_PATH}")

    # Load split
    if not SPLIT_PATH.exists():
        print(f"[ERROR] board_split.json not found: {SPLIT_PATH}")
        print("       Run: python backend/training/board_split.py")
        sys.exit(1)
    with open(SPLIT_PATH, encoding="utf-8") as f:
        split_data = json.load(f)

    test_boards = split_data["test"]
    print(f"Test set: {len(test_boards)} boards\n")

    expected_matrix = fen_to_matrix(START_FEN)
    MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

    # Accumulators
    correct_boards = 0
    total_sq_correct = 0
    total_sq = 0
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)  # confusion matrix

    # Per-board results for the FEN section
    board_results = []

    print(f"{DIVIDER}")
    print("BOARD-LEVEL FEN RESULTS")
    print(DIVIDER)
    print(f"  Ground-truth FEN: {START_FEN}\n")

    for board_entry in test_boards:
        img_path = Path(board_entry["path"])
        if not img_path.exists():
            print(f"  [SKIP] {img_path.name} — file not found")
            continue

        img_bgr  = cv2.imdecode(np.frombuffer(img_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        board_bgr = detect_and_crop_board(img_bgr)
        board_rgb = normalize_perspective(board_bgr, target_size=800)
        squares   = extract_squares(board_rgb)

        pil_imgs = [sq["image_pil"] for sq in squares]
        preds, confs, probs = predict_batch(sess, pil_imgs)

        pred_matrix = [["." for _ in range(8)] for _ in range(8)]
        sq_errors = []
        board_sq_correct = 0

        for idx, sq in enumerate(squares):
            pred_cls = IDX_TO_CLS[preds[idx]]
            gt_cls   = expected_matrix[sq["rank_idx"]][sq["file_idx"]]
            pred_matrix[sq["rank_idx"]][sq["file_idx"]] = pred_cls

            gt_idx   = CLS_TO_IDX[gt_cls]
            pred_idx = int(preds[idx])
            cm[gt_idx][pred_idx] += 1

            is_correct = (pred_cls == gt_cls)
            board_sq_correct += int(is_correct)
            total_sq_correct  += int(is_correct)
            total_sq          += 1

            if not is_correct:
                sq_errors.append({
                    "square": sq["square_name"],
                    "gt":     gt_cls,
                    "pred":   pred_cls,
                    "conf":   float(confs[idx]),
                    "pil":    sq["image_pil"],
                })

        pred_fen_placement = matrix_to_fen(pred_matrix).split()[0]
        board_ok = (pred_fen_placement == START_FEN)
        correct_boards += int(board_ok)
        sq_acc = 100.0 * board_sq_correct / 64

        status = "[CORRECT]" if board_ok else "[WRONG]  "
        print(f"  {status} {img_path.name}  sq_acc={sq_acc:.1f}%")
        for err in sq_errors[:8]:
            print(f"           {err['square']}: GT={err['gt']!r} PRED={err['pred']!r} conf={err['conf']:.2f}")

        # Save misclassified crops
        for err in sq_errors:
            label = f"{err['gt']}_predicted_as_{err['pred']}"
            out_name = f"{label}_{img_path.stem.replace(' ','_')}_{err['square']}.png"
            err["pil"].save(str(MISCLASSIFIED_DIR / out_name))

        board_results.append({
            "name": img_path.name,
            "ok":   board_ok,
            "sq_acc": sq_acc,
            "errors": [{k: v for k, v in e.items() if k != "pil"} for e in sq_errors],
        })

    total_boards = len(test_boards)
    print(f"\n  Board-level FEN accuracy: {correct_boards}/{total_boards}  ({100.0*correct_boards/max(total_boards,1):.1f}%)")
    print(f"  Square-level accuracy:    {total_sq_correct}/{total_sq}  ({100.0*total_sq_correct/max(total_sq,1):.2f}%)")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("PER-CLASS ACCURACY")
    print(DIVIDER)
    print(f"\n  {'Class':<6} {'Correct':>8} {'Total':>8} {'Acc%':>8}  Top errors")
    print("  " + "-" * 60)

    for i, cls in enumerate(CLASSES):
        row   = cm[i]
        total = int(row.sum())
        correct = int(row[i])
        if total == 0:
            continue
        acc = 100.0 * correct / total
        top_wrong = [
            (IDX_TO_CLS[j], int(row[j]))
            for j in np.argsort(-row)
            if j != i and row[j] > 0
        ][:3]
        wrong_str = ", ".join(f"->{c}({cnt})" for c, cnt in top_wrong) if top_wrong else "—"
        flag = "  !!!" if acc < 90 else ("  !" if acc < 99 else "")
        print(f"  {cls:<6} {correct:>8} {total:>8} {acc:>7.1f}%  {wrong_str}{flag}")

    # ── Full confusion matrix ─────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("13x13 CONFUSION MATRIX  (rows=GT, cols=Predicted)")
    print(DIVIDER)
    header = "       " + "".join(f"{c:>5}" for c in CLASSES)
    print(f"\n{header}")
    print("  " + "-" * (5 * n + 5))
    for i, cls in enumerate(CLASSES):
        row_str = "".join(
            f"{'[' + str(cm[i][j]) + ']':>5}" if cm[i][j] > 0 and i == j
            else f"{'*' + str(cm[i][j]) + '*':>5}" if cm[i][j] > 0 and i != j
            else f"{'0':>5}"
            for j in range(n)
        )
        print(f"  {cls:<4} {row_str}")

    # ── Q->q and k->q investigation ───────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("SPECIFIC ERROR INVESTIGATION: Q->q and k->q")
    print(DIVIDER)

    q_idx  = CLS_TO_IDX["q"]
    Q_idx  = CLS_TO_IDX["Q"]
    k_idx  = CLS_TO_IDX["k"]

    q_to_q   = cm[Q_idx][q_idx]   # White Queen predicted as black queen
    k_to_q   = cm[k_idx][q_idx]   # Black King predicted as black queen
    Q_total  = int(cm[Q_idx].sum())
    k_total  = int(cm[k_idx].sum())

    print(f"\n  White Queen (Q) -> black queen (q) errors: {q_to_q}/{Q_total}")
    if Q_total:
        print(f"    Q accuracy: {100.0*(Q_total-q_to_q)/Q_total:.1f}%")

    print(f"\n  Black King  (k) -> black queen (q) errors: {k_to_q}/{k_total}")
    if k_total:
        print(f"    k accuracy: {100.0*(k_total-k_to_q)/k_total:.1f}%")

    misclass_files = list(MISCLASSIFIED_DIR.glob("*.png"))
    print(f"\n  Misclassified crops saved: {len(misclass_files)} files")
    print(f"  Directory: {MISCLASSIFIED_DIR}")

    # List the Q and k errors specifically
    q_k_files = [f for f in misclass_files if f.name.startswith(("Q_predicted", "k_predicted"))]
    if q_k_files:
        print(f"\n  Q/k error crops:")
        for f in sorted(q_k_files):
            print(f"    {f.name}")

    print(f"\n{DIVIDER}")
    print("EVALUATION COMPLETE")
    print(DIVIDER)


if __name__ == "__main__":
    main()
