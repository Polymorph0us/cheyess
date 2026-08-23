import React, { useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Sparkles } from 'lucide-react';

interface ImageUploadProps {
  onImageSelected: (file: File) => void;
  onPresetSelected: (presetKey: string) => void;
  previewUrl: string | null;
  isLoading: boolean;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageSelected,
  onPresetSelected,
  previewUrl,
  isLoading
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onImageSelected(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '1.25rem' }}>
      <div className="panel-header">
        <div className="panel-title">
          <ImageIcon size={20} color="#818cf8" /> Upload Board Image
        </div>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            onImageSelected(e.target.files[0]);
          }
        }}
      />

      <div
        className="dropzone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        {previewUrl ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
            <img
              src={previewUrl}
              alt="Uploaded Board Crop"
              style={{
                maxWidth: '100%',
                maxHeight: '220px',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                boxShadow: '0 4px 16px rgba(0,0,0,0.5)'
              }}
            />
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Click or drag another image to replace
            </span>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
            <UploadCloud size={44} color="#818cf8" />
            <div>
              <div style={{ fontWeight: 600, fontSize: '0.95rem', color: '#f3f4f6' }}>
                Drag & drop a chess board screenshot or image
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                Supports PNG, JPG, WebP top-down board crops
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Preset Test Boards */}
      <div style={{ marginTop: '1.25rem' }}>
        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <Sparkles size={14} color="#fbbf24" /> Preset Sample Boards:
        </div>
        <div className="preset-grid">
          <div className="preset-card" onClick={() => onPresetSelected('start')}>
            <strong>Starting Position</strong>
            <span>Standard setup</span>
          </div>
          <div className="preset-card" onClick={() => onPresetSelected('sicilian')}>
            <strong>Sicilian Najdorf</strong>
            <span>Midgame position</span>
          </div>
          <div className="preset-card" onClick={() => onPresetSelected('endgame')}>
            <strong>Rook Endgame</strong>
            <span>Tactical endgame</span>
          </div>
        </div>
      </div>
    </div>
  );
};
