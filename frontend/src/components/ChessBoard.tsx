import React from 'react';
import { Chessboard } from 'react-chessboard';
import { RotateCw, Edit3 } from 'lucide-react';
import { PiecePalette } from './PiecePalette';

interface ChessBoardProps {
  fen: string;
  onFenChange: (newFen: str) => void;
  boardOrientation: 'white' | 'black';
  onFlipBoard: () => void;
  selectedBrush: string | null;
  onSelectBrush: (piece: string | null) => void;
  onSquareClick: (square: string) => void;
  lowConfidenceSquares?: Array<{ square: string; piece: string; confidence: number }>;
}

export const ChessBoard: React.FC<ChessBoardProps> = ({
  fen,
  onFenChange,
  boardOrientation,
  onFlipBoard,
  selectedBrush,
  onSelectBrush,
  onSquareClick,
  lowConfidenceSquares = []
}) => {
  // Custom square styling for low confidence predictions
  const customSquareStyles: Record<string, React.CSSProperties> = {};

  lowConfidenceSquares.forEach((item) => {
    customSquareStyles[item.square] = {
      backgroundColor: 'rgba(239, 68, 68, 0.45)',
      boxShadow: 'inset 0 0 0 2px #ef4444'
    };
  });

  return (
    <div className="glass-panel" style={{ padding: '1.25rem' }}>
      <div className="panel-header">
        <div className="panel-title">
          <Edit3 size={20} color="#34d399" /> Interactive Board & Palette
        </div>

        <button className="btn btn-secondary btn-sm" onClick={onFlipBoard}>
          <RotateCw size={14} /> Flip ({boardOrientation})
        </button>
      </div>

      <div style={{ maxWidth: '440px', margin: '0 auto', position: 'relative' }}>
        <Chessboard
          position={fen}
          boardOrientation={boardOrientation}
          customSquareStyles={customSquareStyles}
          onSquareClick={(square) => onSquareClick(square)}
          customBoardStyle={{
            borderRadius: '12px',
            boxShadow: '0 8px 30px rgba(0,0,0,0.6)'
          }}
          customDarkSquareStyle={{ backgroundColor: '#769656' }}
          customLightSquareStyle={{ backgroundColor: '#eeeed2' }}
        />
      </div>

      <div style={{ marginTop: '1.25rem' }}>
        <PiecePalette selectedPiece={selectedBrush} onSelectPiece={onSelectBrush} />
      </div>
    </div>
  );
};
