import React from 'react';
import { Activity, AlertTriangle, CheckCircle, Clock } from 'lucide-react';

interface PipelineStatusProps {
  timing?: Record<string, number>;
  totalTimeMs?: number;
  validation?: {
    is_plausible: boolean;
    errors: string[];
    warnings: string[];
    stats: Record<string, number>;
  };
  lowConfidenceCount?: number;
}

export const PipelineStatus: React.FC<PipelineStatusProps> = ({
  timing,
  totalTimeMs,
  validation,
  lowConfidenceCount = 0
}) => {
  return (
    <div className="glass-panel" style={{ padding: '1.25rem' }}>
      <div className="panel-header">
        <div className="panel-title">
          <Activity size={20} color="#f59e0b" /> Pipeline Diagnostics & Plausibility
        </div>

        {totalTimeMs !== undefined && (
          <span className="badge badge-primary mono-font">
            <Clock size={12} /> {totalTimeMs} ms
          </span>
        )}
      </div>

      {/* Validation Status */}
      {validation && (
        <div style={{ marginBottom: '1rem' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.6rem 0.85rem',
            borderRadius: 'var(--radius-sm)',
            background: validation.is_plausible ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            border: `1px solid ${validation.is_plausible ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`
          }}>
            {validation.is_plausible ? (
              <CheckCircle size={18} color="#34d399" />
            ) : (
              <AlertTriangle size={18} color="#f87171" />
            )}

            <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>
              {validation.is_plausible ? 'Position Plausible & Legal' : 'Position Validation Warnings'}
            </div>
          </div>

          {validation.errors.map((err, idx) => (
            <div key={idx} style={{ fontSize: '0.8rem', color: '#f87171', marginTop: '0.35rem', marginLeft: '0.5rem' }}>
              • {err}
            </div>
          ))}

          {validation.warnings.map((warn, idx) => (
            <div key={idx} style={{ fontSize: '0.8rem', color: '#fbbf24', marginTop: '0.35rem', marginLeft: '0.5rem' }}>
              • {warn}
            </div>
          ))}
        </div>
      )}

      {/* Timing Stages Breakdown */}
      {timing && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.5rem' }}>
          <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px', textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>1. Detection</div>
            <div className="mono-font" style={{ fontWeight: 600, fontSize: '0.85rem' }}>{timing.board_detection_ms || 0} ms</div>
          </div>
          <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px', textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>2. Perspective</div>
            <div className="mono-font" style={{ fontWeight: 600, fontSize: '0.85rem' }}>{timing.perspective_ms || 0} ms</div>
          </div>
          <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px', textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>3. Crop (64 sq)</div>
            <div className="mono-font" style={{ fontWeight: 600, fontSize: '0.85rem' }}>{timing.square_extraction_ms || 0} ms</div>
          </div>
          <div style={{ background: 'rgba(0,0,0,0.2)', padding: '0.5rem', borderRadius: '6px', textAlign: 'center' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>4. CNN ONNX</div>
            <div className="mono-font" style={{ fontWeight: 600, fontSize: '0.85rem', color: '#818cf8' }}>{timing.classifier_ms || 0} ms</div>
          </div>
        </div>
      )}
    </div>
  );
};
