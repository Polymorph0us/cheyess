import React, { useState, useEffect } from 'react';
import { Chess } from 'chess.js';
import { Header } from './components/Header';
import { ImageUpload } from './components/ImageUpload';
import { ChessBoard } from './components/ChessBoard';
import { FenBar } from './components/FenBar';
import { EngineAnalysis } from './components/EngineAnalysis';
import { PipelineStatus } from './components/PipelineStatus';

// Sample FEN presets
const PRESET_FENS: Record<string, string> = {
  start: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  sicilian: 'r1bqkb1r/pp1ppppp/2n2n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5',
  endgame: '8/2p5/4k3/1p2P3/pP2K3/8/8/8 b - - 0 42'
};

export const App: React.FC = () => {
  const [fen, setFen] = useState<string>(PRESET_FENS.start);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [boardOrientation, setBoardOrientation] = useState<'white' | 'black'>('white');
  const [selectedBrush, setSelectedBrush] = useState<string | null>(null);
  const [turn, setTurn] = useState<'w' | 'b'>('w');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [onnxLoaded, setOnnxLoaded] = useState<boolean>(false);

  // Analysis state
  const [analysis, setAnalysis] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);

  // Pipeline diagnostics
  const [timing, setTiming] = useState<any>(null);
  const [totalTimeMs, setTotalTimeMs] = useState<number | undefined>(undefined);
  const [validation, setValidation] = useState<any>(null);
  const [lowConfidenceSquares, setLowConfidenceSquares] = useState<any[]>([]);

  // Fetch API Health status on mount
  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => {
        setOnnxLoaded(data.onnx_model_loaded);
      })
      .catch(() => setOnnxLoaded(false));
  }, []);

  // Sync turn state from FEN string
  useEffect(() => {
    const parts = fen.split(' ');
    if (parts.length > 1) {
      setTurn(parts[1] === 'b' ? 'b' : 'w');
    }
  }, [fen]);

  // Handle uploading board image to backend pipeline
  const handleImageUpload = async (file: File) => {
    setIsLoading(true);
    setPreviewUrl(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('file', file);
    formData.append('turn', turn);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (data.success) {
        setFen(data.fen);
        setTiming(data.timing);
        setTotalTimeMs(data.total_time_ms);
        setValidation(data.validation);
        setLowConfidenceSquares(data.low_confidence_squares || []);
      }
    } catch (err) {
      console.error("Prediction error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  // Select Preset Sample Board
  const handlePresetSelect = (presetKey: string) => {
    const presetFen = PRESET_FENS[presetKey] || PRESET_FENS.start;
    setFen(presetFen);
    setAnalysis(null);
    setLowConfidenceSquares([]);
  };

  // Edit Square via Piece Brush Selection
  const handleSquareClick = (square: string) => {
    if (!selectedBrush) return;

    // Convert FEN string to 8x8 matrix, update target square, rebuild FEN
    const parts = fen.split(' ');
    const placement = parts[0];
    const rows = placement.split('/');

    const matrix: string[][] = rows.map((r) => {
      const rowArr: string[] = [];
      for (const char of r) {
        if (!isNaN(Number(char))) {
          rowArr.push(...Array(Number(char)).fill('.'));
        } else {
          rowArr.push(char);
        }
      }
      return rowArr;
    });

    const fileMap: Record<string, number> = { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7 };
    const rankMap: Record<string, number> = { '8': 0, '7': 1, '6': 2, '5': 3, '4': 4, '3': 5, '2': 6, '1': 7 };

    const col = fileMap[square[0]];
    const row = rankMap[square[1]];

    if (row !== undefined && col !== undefined) {
      matrix[row][col] = selectedBrush;

      // Reconstruct FEN placement
      const newRows = matrix.map((rowArr) => {
        let empty = 0;
        let str = '';
        for (const cell of rowArr) {
          if (cell === '.') {
            empty++;
          } else {
            if (empty > 0) {
              str += empty;
              empty = 0;
            }
            str += cell;
          }
        }
        if (empty > 0) str += empty;
        return str;
      });

      const newPlacement = newRows.join('/');
      parts[0] = newPlacement;
      const newFen = parts.join(' ');
      setFen(newFen);
    }
  };

  // Toggle Turn ('w' / 'b')
  const handleToggleTurn = () => {
    const parts = fen.split(' ');
    const newTurn = parts[1] === 'w' ? 'b' : 'w';
    parts[1] = newTurn;
    setFen(parts.join(' '));
  };

  // Trigger Stockfish analysis
  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fen, depth: 15 })
      });
      const data = await res.json();
      setAnalysis(data);
    } catch (err) {
      console.error("Analysis failed:", err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app-container">
      <Header onnxLoaded={onnxLoaded} />

      <div className="main-grid">
        {/* Left Column: Image Upload & Interactive Board */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <ImageUpload
            onImageSelected={handleImageUpload}
            onPresetSelected={handlePresetSelect}
            previewUrl={previewUrl}
            isLoading={isLoading}
          />

          <ChessBoard
            fen={fen}
            onFenChange={setFen}
            boardOrientation={boardOrientation}
            onFlipBoard={() => setBoardOrientation(boardOrientation === 'white' ? 'black' : 'white')}
            selectedBrush={selectedBrush}
            onSelectBrush={setSelectedBrush}
            onSquareClick={handleSquareClick}
            lowConfidenceSquares={lowConfidenceSquares}
          />
        </div>

        {/* Right Column: FEN Bar, Engine Evaluation, Pipeline Diagnostics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <FenBar
            fen={fen}
            onApplyFen={setFen}
            turn={turn}
            onToggleTurn={handleToggleTurn}
          />

          <EngineAnalysis
            analysis={analysis}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
          />

          <PipelineStatus
            timing={timing}
            totalTimeMs={totalTimeMs}
            validation={validation}
            lowConfidenceCount={lowConfidenceSquares.length}
          />
        </div>
      </div>
    </div>
  );
};

export default App;
