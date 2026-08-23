import React, { useState } from 'react';
import { Copy, Check, Hash, ArrowRightLeft } from 'lucide-react';

interface FenBarProps {
  fen: string;
  onApplyFen: (newFen: string) => void;
  turn: 'w' | 'b';
  onToggleTurn: () => void;
}

export const FenBar: React.FC<FenBarProps> = ({
  fen,
  onApplyFen,
  turn,
  onToggleTurn
}) => {
  const [copied, setCopied] = useState(false);
  const [inputFen, setInputFen] = useState(fen);

  React.useEffect(() => {
    setInputFen(fen);
  }, [fen]);

  const handleCopy = () => {
    navigator.clipboard.writeText(fen);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="glass-panel" style={{ padding: '1rem 1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
        <Hash size={18} color="#818cf8" />
        <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#f3f4f6' }}>Position FEN Notation</span>
        
        <button
          className="btn btn-secondary btn-sm"
          style={{ marginLeft: 'auto', gap: '0.35rem' }}
          onClick={onToggleTurn}
        >
          <ArrowRightLeft size={13} /> Move: <strong>{turn === 'w' ? 'White to move' : 'Black to move'}</strong>
        </button>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <input
          type="text"
          value={inputFen}
          onChange={(e) => setInputFen(e.target.value)}
          className="mono-font"
          style={{
            flex: 1,
            background: 'rgba(0,0,0,0.3)',
            border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-sm)',
            padding: '0.5rem 0.75rem',
            color: '#f3f4f6',
            fontSize: '0.85rem'
          }}
        />

        <button className="btn btn-secondary btn-sm" onClick={() => onApplyFen(inputFen)}>
          Apply
        </button>

        <button className="btn btn-secondary btn-sm" onClick={handleCopy}>
          {copied ? <Check size={14} color="#34d399" /> : <Copy size={14} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
    </div>
  );
};
