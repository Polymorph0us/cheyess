import os
import shutil
import chess
import chess.engine
from typing import Dict, Any

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def analyze_position(fen: str, depth: int = 15, time_limit: float = 0.5) -> Dict[str, Any]:
    """
    Analyzes position using Stockfish UCI engine if available, or static board evaluation fallback.
    Returns: score (float/str), eval_type ('cp' or 'mate'), best_move (str), pv (list of moves).
    """
    try:
        board = chess.Board(fen)
    except Exception as e:
        return {
            "success": False,
            "error": f"Invalid FEN string: {str(e)}",
            "score": 0.0,
            "eval_type": "cp",
            "best_move": None,
            "pv": []
        }

    # Search for Stockfish executable in system PATH or common installation locations
    stockfish_path = shutil.which("stockfish") or shutil.which("stockfish.exe")

    if stockfish_path and os.path.exists(stockfish_path):
        try:
            with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
                info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit))
                score_obj = info.get("score").white()

                if score_obj.is_mate():
                    eval_type = "mate"
                    score_val = score_obj.mate()
                else:
                    eval_type = "cp"
                    score_val = score_obj.score() / 100.0 # Convert centipawns to pawns

                pv_moves = [move.uci() for move in info.get("pv", [])]
                best_move = pv_moves[0] if pv_moves else None

                return {
                    "success": True,
                    "engine": "Stockfish",
                    "score": score_val,
                    "eval_type": eval_type,
                    "best_move": best_move,
                    "pv": pv_moves[:5],
                    "depth": depth
                }
        except Exception as e:
            print(f"Stockfish execution failed ({e}), falling back to static evaluator.")

    # Static Evaluation Fallback (when Stockfish binary is not present)
    score_cp = 0
    for sq, piece in board.piece_map().items():
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            score_cp += val
        else:
            score_cp -= val

    score_pawns = score_cp / 100.0

    # Pick first legal move if available
    legal_moves = list(board.legal_moves)
    best_move = legal_moves[0].uci() if legal_moves else None

    return {
        "success": True,
        "engine": "StaticEvaluator (Fallback)",
        "score": score_pawns,
        "eval_type": "cp",
        "best_move": best_move,
        "pv": [m.uci() for m in legal_moves[:3]],
        "depth": 1
    }
