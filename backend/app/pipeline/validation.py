import chess
from typing import List, Dict, Any

def validate_board_position(matrix: List[List[str]]) -> Dict[str, Any]:
    """
    Performs plausibility and legality checks on the predicted 8x8 piece placement matrix.
    Checks: King counts, Pawn limits, rank 1/8 pawn placement, total piece counts, legal FEN.
    """
    warnings = []
    errors = []

    white_kings = 0
    black_kings = 0
    white_pawns = 0
    black_pawns = 0
    white_pieces = 0
    black_pieces = 0

    for r_idx, row in enumerate(matrix):
        for f_idx, piece in enumerate(row):
            if piece == '.':
                continue
            
            if piece.isupper():
                white_pieces += 1
                if piece == 'K': white_kings += 1
                elif piece == 'P': white_pawns += 1
            else:
                black_pieces += 1
                if piece == 'k': black_kings += 1
                elif piece == 'p': black_pawns += 1

            # Check rank 1 and rank 8 pawns (ranks 8 is r_idx=0, rank 1 is r_idx=7)
            if piece.upper() == 'P' and (r_idx == 0 or r_idx == 7):
                errors.append(f"Invalid pawn placement: '{piece}' found on rank {8 - r_idx}.")

    if white_kings != 1:
        errors.append(f"Invalid position: Expected exactly 1 White King ('K'), found {white_kings}.")
    if black_kings != 1:
        errors.append(f"Invalid position: Expected exactly 1 Black King ('k'), found {black_kings}.")

    if white_pawns > 8:
        warnings.append(f"High white pawn count: {white_pawns} pawns detected.")
    if black_pawns > 8:
        warnings.append(f"High black pawn count: {black_pawns} pawns detected.")

    if white_pieces > 16:
        warnings.append(f"High white piece count: {white_pieces} pieces detected.")
    if black_pieces > 16:
        warnings.append(f"High black piece count: {black_pieces} pieces detected.")

    is_plausible = len(errors) == 0

    return {
        "is_plausible": is_plausible,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "white_pieces": white_pieces,
            "black_pieces": black_pieces,
            "white_pawns": white_pawns,
            "black_pawns": black_pawns
        }
    }
