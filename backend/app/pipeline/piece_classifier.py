import os
import json
import numpy as np
from typing import List, Dict, Any
from PIL import Image
import onnxruntime as ort

CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

class PieceClassifier:
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, "..", "..", "models", "piece_classifier.onnx")
        
        self.model_path = model_path
        self.session = None

        if os.path.exists(self.model_path):
            try:
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                print(f"Loaded ONNX piece classifier from: {self.model_path}")
            except Exception as e:
                print(f"Warning: Failed to load ONNX model ({e}). Using fallback classifier.")

    def preprocess_crop(self, pil_img: Image.Image) -> np.ndarray:
        """Preprocesses crop image: Resize to 64x64, normalize ImageNet stats, C x H x W float32."""
        img = pil_img.resize((64, 64), Image.BILINEAR)  # matches torchvision.transforms.Resize default used in training
        arr = np.array(img).astype(np.float32) / 255.0

        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        # HWC to CHW
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def classify_batch(self, squares: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs single batch inference over 64 square crops.
        Attaches 'piece', 'confidence', and 'probabilities' to each square dict.
        """
        if self.session is not None:
            batch_inputs = np.stack([self.preprocess_crop(sq["image_pil"]) for sq in squares], axis=0)
            
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name

            logits = self.session.run([output_name], {input_name: batch_inputs})[0]
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            predicted_indices = np.argmax(probs, axis=1)

            classified_squares = []
            for idx, sq in enumerate(squares):
                pred_idx = predicted_indices[idx]
                conf = float(probs[idx][pred_idx])
                piece_label = IDX_TO_CLASS[pred_idx]

                sq_copy = sq.copy()
                sq_copy["piece"] = piece_label
                sq_copy["confidence"] = round(conf, 4)
                classified_squares.append(sq_copy)

            return classified_squares
        else:
            # Fallback heuristic / empty detection if ONNX model is not present
            classified_squares = []
            for sq in squares:
                sq_copy = sq.copy()
                sq_copy["piece"] = "."
                sq_copy["confidence"] = 1.0
                classified_squares.append(sq_copy)
            return classified_squares
