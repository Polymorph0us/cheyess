# ♟ Cheyess — Chess Position Recognition & Analysis

A modular computer vision and machine learning platform for extracting chess positions from images, building valid FEN (Forsyth-Edwards Notation) strings, and evaluating positions using the Stockfish UCI engine.

---

## 🚀 Key Features

- **Per-Square CNN Classifier**: 13-class deep CNN model (`[empty, P, N, B, R, Q, K, p, n, b, r, q, k]`) trained on rendered piece shapes and color themes.
- **Low-Latency ONNX CPU Inference**: Runs a single 64-crop batch forward pass in milliseconds without requiring GPU hardware in production.
- **7-Stage Decoupled Computer Vision Pipeline**: Each stage (`board_detection` → `perspective` → `square_extraction` → `piece_classifier` → `board_reconstruction` → `validation` → `fen_builder`) is independently testable and modular.
- **Plausibility & Legality Validation**: Built-in rules via `python-chess` checking King counts, Pawn limits, rank bounds, and position sanity.
- **Interactive Drag-and-Drop Board**: React frontend with `react-chessboard` and a piece brush palette for manual single-click square edits.
- **Stockfish UCI Engine Analysis**: Real-time position evaluation (+/- score gauge bar, mate counter, top engine move, and principal variation line).
- **Future Fine-Tuning Support**: Structured data loading supporting synthetic procedural data and user-provided real photo datasets (`backend/dataset/real/`).

---

## 🛠️ Architecture & Pipeline Overview

```
Upload Image
   │
   ▼
[1] Board Detection ──────── Standardize image input (Stage 1: contours / corner regressor)
   │
   ▼
[2] Perspective Correction ── Normalize board image to 800×800 top-down square
   │
   ▼
[3] Square Extraction ────── Slice 800×800 board into 64 uniform 100×100 square crops (a8..h1)
   │
   ▼
[4] Piece Classification ── Single batch ONNX forward pass (64 crops ➔ 13 classes + confidence)
   │
   ▼
[5] Board Reconstruction ── Map piece predictions to 8×8 rank/file matrix + flag low-confidence squares
   │
   ▼
[6] Plausibility Validation ── Check King & Pawn bounds, illegal pawn ranks via python-chess
   │
   ▼
[7] FEN Builder ─────────── Generate valid FEN string with turn, castling rights, and en-passant
   │
   ▼
[8] Interactive UI & Engine ── React Board manual correction ➔ UCI Stockfish position evaluation
```

---

## 📂 Project Structure

```
cheyess/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app entry point & CORS
│   │   ├── api/
│   │   │   └── routes.py           # /api/predict, /api/analyze, /api/health
│   │   ├── pipeline/
│   │   │   ├── board_detection.py  # Stage 1: Input image normalization
│   │   │   ├── perspective.py      # Stage 2: 800x800 square normalization
│   │   │   ├── square_extraction.py# Stage 3: 64-square cropping (100x100)
│   │   │   ├── piece_classifier.py # Stage 4: ONNX Runtime 64-batch inference
│   │   │   ├── board_reconstruction.py # Stage 5: 8x8 matrix reconstruction
│   │   │   ├── validation.py       # Stage 6: Plausibility checks
│   │   │   └── fen_builder.py      # Stage 7: FEN string builder
│   │   └── engine/
│   │       └── stockfish_wrapper.py# Stockfish UCI wrapper & static fallback
│   ├── models/
│   │   ├── piece_classifier.onnx   # Exported trained ONNX model
│   │   └── classes.json            # Class index metadata
│   ├── training/
│   │   ├── generate_synthetic_data.py # Procedural board & piece generator
│   │   └── train_classifier.py     # PyTorch training & ONNX export
│   ├── tests/                      # Pytest suite
│   │   ├── test_fen_builder.py
│   │   ├── test_piece_classifier.py
│   │   ├── test_square_extraction.py
│   │   └── test_validation.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx          # Navigation header & engine badges
│   │   │   ├── ImageUpload.tsx     # Drag & drop upload & preset boards
│   │   │   ├── ChessBoard.tsx      # Interactive react-chessboard
│   │   │   ├── PiecePalette.tsx    # Piece brush palette
│   │   │   ├── FenBar.tsx          # FEN view/edit, turn toggle, copy
│   │   │   ├── EngineAnalysis.tsx  # Stockfish evaluation score & best move
│   │   │   └── PipelineStatus.tsx  # Stage timing & diagnostic breakdown
│   │   ├── styles/
│   │   │   └── main.css            # Dark glassmorphic design system
│   │   ├── App.tsx                 # Main layout & state manager
│   │   └── main.tsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── README.md
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ & npm

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# (Optional) Generate synthetic data & train piece classifier model
python training/generate_synthetic_data.py
python training/train_classifier.py

# Run pytest suite
python -m pytest

# Start FastAPI server (runs on http://localhost:8000)
python -m uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install npm dependencies
npm install

# Start Vite development server (runs on http://localhost:5173)
npm run dev
```

Open `http://localhost:5173` in your browser to interact with the application.

---

## 📡 API Reference

### `POST /api/predict`
Uploads a chess board image and executes the full CV pipeline.
- **Request**: Multipart Form Data (`file: UploadFile`, `turn: "w"|"b"`, `castling: "KQkq"`)
- **Response**:
  ```json
  {
    "success": true,
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "matrix": [["r","n","b","q","k","b","n","r"], ...],
    "timing": {
      "board_detection_ms": 1.2,
      "perspective_ms": 2.1,
      "square_extraction_ms": 3.4,
      "classifier_ms": 12.8,
      "validation_ms": 0.5,
      "fen_builder_ms": 0.1
    },
    "total_time_ms": 20.1
  }
  ```

### `POST /api/analyze`
Submits a FEN string to the Stockfish UCI engine.
- **Request**: `{ "fen": "r1bqkb1r/pp1ppppp/2n2n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5", "depth": 15 }`
- **Response**: `{ "success": true, "score": 0.45, "eval_type": "cp", "best_move": "d4e6", "pv": ["d4e6", "d4c6", "d4f5"] }`

### `GET /api/health`
Returns pipeline health and ONNX engine loading status.

---

## 🎯 Fine-Tuning with Real Photos

To fine-tune the piece classifier on real photographs:
1. Place image crops inside `backend/dataset/real/<class_name>/` (e.g. `backend/dataset/real/P/`, `backend/dataset/real/empty/`, etc.).
2. Run `python training/train_classifier.py`.
3. The dataset loader automatically combines synthetic and real datasets, training a unified model exported directly to `backend/models/piece_classifier.onnx`.

---

## 📜 License

MIT License — free to use and modify.
