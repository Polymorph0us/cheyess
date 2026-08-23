from typing import List

def matrix_to_fen(
    matrix: List[List[str]],
    turn: str = 'w',
    castling: str = 'KQkq',
    en_passant: str = '-',
    halfmove: int = 0,
    fullmove: int = 1
) -> str:
    """
    Converts 8x8 piece matrix into a complete valid Forsyth-Edwards Notation (FEN) string.
    """
    fen_rows = []
    for row in matrix:
        empty_count = 0
        row_str = ""
        for piece in row:
            if piece == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += piece
        if empty_count > 0:
            row_str += str(empty_count)
        fen_rows.append(row_str)

    piece_placement = "/".join(fen_rows)
    return f"{piece_placement} {turn} {castling} {en_passant} {halfmove} {fullmove}"

def fen_to_matrix(fen: str) -> List[List[str]]:
    """
    Parses FEN string back into an 8x8 piece placement matrix.
    """
    parts = fen.strip().split()
    piece_placement = parts[0]
    rows = piece_placement.split('/')

    matrix = []
    for r in rows:
        matrix_row = []
        for char in r:
            if char.isdigit():
                matrix_row.extend(['.'] * int(char))
            else:
                matrix_row.append(char)
        matrix.append(matrix_row)
    return matrix
