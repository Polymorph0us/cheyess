import numpy as np
from app.pipeline.square_extraction import extract_squares

def test_extract_squares():
    dummy_board = np.zeros((800, 800, 3), dtype=np.uint8)
    squares = extract_squares(dummy_board)

    assert len(squares) == 64
    assert squares[0]["square_name"] == "a8"
    assert squares[7]["square_name"] == "h8"
    assert squares[56]["square_name"] == "a1"
    assert squares[63]["square_name"] == "h1"

    for sq in squares:
        assert sq["image_np"].shape == (100, 100, 3)
