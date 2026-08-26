# Cheyess

Chess position recognition from screenshots. Upload a board image, get a FEN string, run Stockfish analysis.

Built with FastAPI + ONNX Runtime (backend) and React + Vite (frontend).

---

## What it does

1. Detects the board region in a screenshot
2. Splits it into 64 squares
3. Classifies each square (empty, or one of 12 piece types) via a CNN exported to ONNX
4. Builds a FEN string with validation (king count, pawn placement, etc.)
5. Optionally runs Stockfish UCI analysis on the resulting position

The frontend lets you drag-drop an image, see the recognized position on an interactive board, fix any misidentified squares with a piece palette, and get engine evaluation.

---

## Quick start

### Backend

```bash
cd backend
pip install -r requirements.txt

# Start the API server
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [localhost:5173](http://localhost:5173). The backend runs on port 8000.

---

## Training the classifier

A pre-trained `piece_classifier.onnx` is included. To retrain from scratch:

```bash
cd backend

# Generate synthetic training data
python training/generate_synthetic_data.py

# Import real board screenshots (starting position)
python training/import_real_dataset.py

# Create board-level train/val/test split
python training/board_split.py

# Train and export to ONNX
python training/train_classifier.py

# Evaluate on held-out test boards
python training/evaluate_classifier.py
```

### Adding images with different positions

Create a `manifest.csv` next to your screenshots:

```csv
filename,fen
game_screenshot_1.png,r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R
game_screenshot_2.png,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
```

Then import:

```bash
python training/import_real_dataset.py --manifest path/to/manifest.csv
python training/board_split.py
python training/train_classifier.py
```

Crops from the same board are never split across train/val/test sets.

---

## API

### `POST /api/predict`

Upload a board image → get FEN + per-stage timing.

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@board.png" -F "turn=w" -F "castling=KQkq"
```

```json
{
  "success": true,
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "matrix": [["r","n","b","q","k","b","n","r"], "..."],
  "timing": { "classifier_ms": 12.8, "total_time_ms": 20.1 }
}
```

### `POST /api/analyze`

Submit a FEN → get Stockfish evaluation.

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "depth": 15}'
```

```json
{
  "success": true,
  "score": -0.30,
  "eval_type": "cp",
  "best_move": "e7e5",
  "pv": ["e7e5", "g1f3", "b8c6"]
}
```

### `GET /api/health`

Pipeline status and model loading check.

---

<details>
<summary><strong>Pipeline stages</strong></summary>

```
Image → Board Detection → Perspective Correction (800×800)
      → Square Extraction (64 × 100×100 crops)
      → Piece Classification (ONNX batch inference, 13 classes)
      → Board Reconstruction (8×8 matrix)
      → Validation (king count, pawn ranks, python-chess)
      → FEN Builder
      → Stockfish Analysis (optional)
```

</details>

<details>
<summary><strong>Project structure</strong></summary>

```
backend/
├── app/
│   ├── main.py                      # FastAPI entry point
│   ├── api/routes.py                # /predict, /analyze, /health
│   ├── pipeline/
│   │   ├── board_detection.py       # Board region detection
│   │   ├── lichess_board_detector.py# Lichess-specific detector
│   │   ├── perspective.py           # 800×800 normalization
│   │   ├── square_extraction.py     # 64-square cropping
│   │   ├── piece_classifier.py      # ONNX inference
│   │   ├── board_reconstruction.py  # 8×8 matrix assembly
│   │   ├── validation.py            # Position legality checks
│   │   └── fen_builder.py           # FEN string generation
│   └── engine/
│       └── stockfish_wrapper.py     # Stockfish UCI interface
├── models/
│   ├── piece_classifier.onnx        # Trained model
│   └── classes.json                 # Class index map
├── training/
│   ├── generate_synthetic_data.py   # Procedural data generator
│   ├── import_real_dataset.py       # Real screenshot importer
│   ├── board_split.py               # Board-level train/val/test split
│   ├── train_classifier.py          # Training + ONNX export
│   ├── evaluate_classifier.py       # Test set evaluation
│   └── diagnose_model.py            # Full diagnostic report
├── dataset/
│   ├── synthetic/                   # Generated training crops
│   ├── real/                        # Imported real crops
│   └── board_split.json             # Split assignments + hash map
└── tests/

frontend/
├── src/
│   ├── App.tsx                      # Main app state
│   ├── components/
│   │   ├── ImageUpload.tsx          # Drag-drop upload
│   │   ├── ChessBoard.tsx           # Interactive board
│   │   ├── PiecePalette.tsx         # Piece brush for corrections
│   │   ├── FenBar.tsx               # FEN display/edit
│   │   ├── EngineAnalysis.tsx       # Stockfish eval display
│   │   └── PipelineStatus.tsx       # Stage timing breakdown
│   └── styles/main.css
├── index.html
└── package.json
```

</details>

---

## Tech stack

| Layer | Stack |
|-------|-------|
| Backend | Python, FastAPI, PyTorch, ONNX Runtime, OpenCV, python-chess |
| Frontend | React 18, TypeScript, Vite, react-chessboard, chess.js |
| Engine | Stockfish (UCI protocol) |

---

## License

MIT

