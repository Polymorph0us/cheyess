import React from 'react';
import { Cpu, Play, TrendingUp, Layers } from 'lucide-react';

interface EngineAnalysisProps {
  analysis: {
    success: boolean;
    engine?: string;
    score: number;
    eval_type: string;
    best_move: string | null;
    pv: string[];
    depth?: number;
    error?: string;
  } | null;
  onAnalyze: () => void;
  isAnalyzing: boolean;
}

export const EngineAnalysis: React.FC<EngineAnalysisProps> = ({
  analysis,
  onAnalyze,
  isAnalyzing
}) => {
  const getScoreDisplay = () => {
    if (!analysis || !analysis.success) return '0.0';
    if (analysis.eval_type === 'mate') {
      return `M${analysis.score}`;
    }
    const val = analysis.score;
    return val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2);
  };

  const getScorePercentage = () => {
    if (!analysis || !analysis.success) return 50;
    if (analysis.eval_type === 'mate') {
      return analysis.score > 0 ? 100 : 0;
    }
    // Clamp evaluation score between -5.0 and +5.0 pawns for visual gauge
    const clamped = Math.max(-5.0, Math.min(5.0, analysis.score));
    return ((clamped + 5.0) / 10.0) * 100;
  };

  return (
    <div className="glass-panel" style={{ padding: '1.25rem' }}>
      <div className="panel-header">
        <div className="panel-title">
          <Cpu size={20} color="#818cf8" /> Engine Evaluation & Best Move
        </div>

        <button
          className="btn btn-primary btn-sm"
          onClick={onAnalyze}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? (
            'Analyzing...'
          ) : (
            <>
              <Play size={14} /> Evaluate Position
            </>
          )}
        </button>
      </div>

      {analysis && analysis.success ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Score Track Gauge */}
          <div className="score-bar-container">
            <div
              className="score-badge"
              style={{
                color: analysis.score >= 0 ? '#34d399' : '#f87171'
              }}
            >
              {getScoreDisplay()}
            </div>

            <div className="score-track">
              <div
                className="score-fill"
                style={{
                  width: `${getScorePercentage()}%`,
                  background: analysis.score >= 0 
                    ? 'linear-gradient(90deg, #10b981 0%, #34d399 100%)' 
                    : 'linear-gradient(90deg, #ef4444 0%, #f87171 100%)'
                }}
              />
            </div>

            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', minWidth: 50 }}>
              {analysis.engine || 'Engine'}
            </span>
          </div>

          {/* Best Move & Continuation */}
          <div style={{
            background: 'rgba(0,0,0,0.25)',
            borderRadius: 'var(--radius-md)',
            padding: '0.85rem 1rem',
            border: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Top Engine Move</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f3f4f6' }} className="mono-font">
                {analysis.best_move ? analysis.best_move : 'None (Game Over)'}
              </div>
            </div>

            {analysis.pv && analysis.pv.length > 0 && (
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Principal Variation</div>
                <div className="mono-font" style={{ fontSize: '0.85rem', color: '#818cf8' }}>
                  {analysis.pv.join('  ➔  ')}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div style={{
          textAlign: 'center',
          padding: '1.5rem',
          color: 'var(--text-muted)',
          fontSize: '0.85rem'
        }}>
          Click <strong>Evaluate Position</strong> above to calculate Stockfish score & principal variation.
        </div>
      )}
    </div>
  );
};
