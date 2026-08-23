import React from 'react';
import { Cpu, ShieldCheck, Zap } from 'lucide-react';

interface HeaderProps {
  onnxLoaded: boolean;
}

export const Header: React.FC<HeaderProps> = ({ onnxLoaded }) => {
  return (
    <header>
      <div className="header-title">
        <div style={{
          background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
          width: 42,
          height: 42,
          borderRadius: 12,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '1.5rem',
          boxShadow: '0 4px 14px rgba(99, 102, 241, 0.4)'
        }}>
          ♟
        </div>
        <div>
          <h1>Chess Vision AI</h1>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Stage 0 MVP — CNN Per-Square Piece Classification & UCI Engine Analysis
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
        <span className="badge badge-primary">
          <Zap size={13} /> Stage 0 CNN
        </span>
        <span className={onnxLoaded ? "badge badge-success" : "badge badge-warning"}>
          <Cpu size={13} /> {onnxLoaded ? "ONNX Engine Active" : "Fallback Engine"}
        </span>
        <span className="badge badge-success">
          <ShieldCheck size={13} /> python-chess UCI
        </span>
      </div>
    </header>
  );
};
