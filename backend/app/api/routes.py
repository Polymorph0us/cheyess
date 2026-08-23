import time
import io
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from PIL import Image

from app.pipeline.board_detection import detect_board
from app.pipeline.perspective import normalize_perspective
from app.pipeline.square_extraction import extract_squares
from app.pipeline.piece_classifier import PieceClassifier
from app.pipeline.board_reconstruction import reconstruct_board_matrix
from app.pipeline.validation import validate_board_position
from app.pipeline.fen_builder import matrix_to_fen
from app.engine.stockfish_wrapper import analyze_position

router = APIRouter()
classifier = PieceClassifier()

class AnalyzeRequest(BaseModel):
    fen: str
    depth: Optional[int] = 15

@router.get("/health")
def health_check():
    return {
        "status": "online",
        "onnx_model_loaded": classifier.session is not None,
        "model_path": classifier.model_path
    }

@router.post("/predict")
async def predict_chess_position(
    file: UploadFile = File(...),
    turn: str = Form("w"),
    castling: str = Form("KQkq"),
    en_passant: str = Form("-")
):
    start_time = time.time()
    timing_breakdown = {}

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    # Stage 1: Board Detection
    t0 = time.time()
    board_bgr = detect_board(pil_image)
    timing_breakdown["board_detection_ms"] = round((time.time() - t0) * 1000, 2)

    # Stage 2: Perspective Correction
    t0 = time.time()
    board_rgb_800 = normalize_perspective(board_bgr)
    timing_breakdown["perspective_ms"] = round((time.time() - t0) * 1000, 2)

    # Stage 3: Square Extraction
    t0 = time.time()
    squares = extract_squares(board_rgb_800)
    timing_breakdown["square_extraction_ms"] = round((time.time() - t0) * 1000, 2)

    # Stage 4: Piece Classifier (64 batch ONNX forward pass)
    t0 = time.time()
    classified_squares = classifier.classify_batch(squares)
    timing_breakdown["classifier_ms"] = round((time.time() - t0) * 1000, 2)

    # Stage 5: Board Reconstruction
    t0 = time.time()
    recon = reconstruct_board_matrix(classified_squares)
    matrix = recon["matrix"]
    low_conf = recon["low_confidence_squares"]
    timing_breakdown["reconstruction_ms"] = round((time.time() - t0) * 1000, 2)

    # Stage 6: Validation
    t0 = time.time()
    val_res = validate_board_position(matrix)
    timing_breakdown["validation_ms"] = round((time.time() - t0) * 1000, 2)

    # Stage 7: FEN Builder
    t0 = time.time()
    fen_str = matrix_to_fen(matrix, turn=turn, castling=castling, en_passant=en_passant)
    timing_breakdown["fen_builder_ms"] = round((time.time() - t0) * 1000, 2)

    total_time_ms = round((time.time() - start_time) * 1000, 2)

    # Simplified list of square predictions for UI square confidence overlays
    square_predictions = [
        {
            "square": sq["square_name"],
            "piece": sq["piece"],
            "confidence": sq["confidence"]
        } for sq in classified_squares
    ]

    return {
        "success": True,
        "fen": fen_str,
        "matrix": matrix,
        "square_predictions": square_predictions,
        "low_confidence_squares": low_conf,
        "validation": val_res,
        "timing": timing_breakdown,
        "total_time_ms": total_time_ms
    }

@router.post("/analyze")
def analyze_fen(request: AnalyzeRequest):
    res = analyze_position(request.fen, depth=request.depth or 15)
    return res
