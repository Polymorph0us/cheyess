from typing import List, Dict, Any

def reconstruct_board_matrix(classified_squares: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reconstructs 8x8 grid matrix from 64 square predictions.
    Returns matrix representation (8 rows x 8 columns) and metadata (low confidence squares).
    """
    matrix = []
    low_confidence_squares = []

    # 64 squares are ordered rank 8 to 1, file a to h
    for r in range(8):
        row = []
        for f in range(8):
            idx = r * 8 + f
            sq = classified_squares[idx]
            row.append(sq["piece"])

            if sq.get("confidence", 1.0) < 0.70:
                low_confidence_squares.append({
                    "square": sq["square_name"],
                    "piece": sq["piece"],
                    "confidence": sq["confidence"]
                })
        matrix.append(row)

    return {
        "matrix": matrix,
        "low_confidence_squares": low_confidence_squares
    }
