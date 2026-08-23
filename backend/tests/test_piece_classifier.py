from PIL import Image
from app.pipeline.piece_classifier import PieceClassifier

def test_piece_classifier_onnx_inference():
    classifier = PieceClassifier()
    assert classifier.session is not None, "ONNX model session should be loaded"

    # Create dummy 100x100 square crops (64 squares)
    dummy_squares = []
    for i in range(64):
        img = Image.new("RGB", (100, 100), (200, 200, 200))
        dummy_squares.append({
            "square_name": f"sq_{i}",
            "image_pil": img
        })

    results = classifier.classify_batch(dummy_squares)
    assert len(results) == 64
    for res in results:
        assert "piece" in res
        assert "confidence" in res
        assert 0.0 <= res["confidence"] <= 1.0
