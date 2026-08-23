import React from 'react';
import { Eraser } from 'lucide-react';

interface PiecePaletteProps {
  selectedPiece: string | null;
  onSelectPiece: (piece: string | null) => void;
}

const PIECE_ITEMS = [
  { symbol: 'P', glyph: '♙', title: 'White Pawn' },
  { symbol: 'N', glyph: '♘', title: 'White Knight' },
  { symbol: 'B', glyph: '♗', title: 'White Bishop' },
  { symbol: 'R', glyph: '♖', title: 'White Rook' },
  { symbol: 'Q', glyph: '♕', title: 'White Queen' },
  { symbol: 'K', glyph: '♔', title: 'White King' },
  { symbol: 'p', glyph: '♟', title: 'Black Pawn' },
  { symbol: 'n', glyph: '♞', title: 'Black Knight' },
  { symbol: 'b', glyph: '♝', title: 'Black Bishop' },
  { symbol: 'r', glyph: '♜', title: 'Black Rook' },
  { symbol: 'q', glyph: '♛', title: 'Black Queen' },
  { symbol: 'k', glyph: '♚', title: 'Black King' },
];

export const PiecePalette: React.FC<PiecePaletteProps> = ({ selectedPiece, onSelectPiece }) => {
  return (
    <div>
      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.4rem', textAlign: 'center' }}>
        {selectedPiece ? `Selected Brush: '${selectedPiece}' (click square on board to apply)` : 'Click a piece brush below, then click any square to edit:'}
      </div>

      <div className="piece-palette">
        {PIECE_ITEMS.map((item) => (
          <button
            key={item.symbol}
            className={`palette-btn ${selectedPiece === item.symbol ? 'selected' : ''}`}
            onClick={() => onSelectPiece(selectedPiece === item.symbol ? null : item.symbol)}
            title={item.title}
          >
            {item.glyph}
          </button>
        ))}

        <button
          className={`palette-btn ${selectedPiece === '.' ? 'selected' : ''}`}
          onClick={() => onSelectPiece(selectedPiece === '.' ? null : '.')}
          title="Clear Square (Empty)"
        >
          <Eraser size={18} color="#9ca3af" />
        </button>
      </div>
    </div>
  );
};
