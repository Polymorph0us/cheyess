import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# Unicode chess symbols for visual rendering of piece glyphs
UNICODE_PIECES = {
    'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

COLOR_THEMES = [
    # (light square, dark square)
    ((238, 238, 210), (118, 150, 86)),  # Green / White (Chess.com default)
    ((240, 217, 181), (181, 136, 99)),  # Wood / Tan (Lichess wood)
    ((222, 227, 230), (140, 162, 173)), # Blue / Gray
    ((240, 240, 240), (100, 100, 100)), # Dark / Light Gray
    ((245, 245, 220), (139, 69, 19)),   # Brown / Beige
]

def create_piece_image(piece_symbol: str, bg_color: tuple, square_size: int = 100) -> Image.Image:
    """Generates a single square crop (100x100) with or without a piece symbol."""
    img = Image.new("RGB", (square_size, square_size), bg_color)
    draw = ImageDraw.Draw(img)

    if piece_symbol == '.':
        return img

    is_white_piece = piece_symbol.isupper()
    fg_color = (255, 255, 255) if is_white_piece else (20, 20, 20)
    outline_color = (0, 0, 0) if is_white_piece else (220, 220, 220)

    # Draw geometric piece representations if font rendering is unavailable
    center_x, center_y = square_size // 2, square_size // 2
    r = square_size // 3

    # Add shadow/outline for visual realism
    offset = random.randint(-2, 2)
    cx, cy = center_x + offset, center_y + offset

    # Draw piece shapes distinctly based on piece type
    upper = piece_symbol.upper()
    if upper == 'P': # Pawn
        draw.ellipse([cx - r * 0.5, cy - r * 0.6, cx + r * 0.5, cy + r * 0.4], fill=fg_color, outline=outline_color, width=3)
        draw.rectangle([cx - r * 0.6, cy + r * 0.3, cx + r * 0.6, cy + r * 0.6], fill=fg_color, outline=outline_color, width=2)
    elif upper == 'R': # Rook
        draw.rectangle([cx - r * 0.6, cy - r * 0.6, cx + r * 0.6, cy + r * 0.5], fill=fg_color, outline=outline_color, width=3)
        # Crenellations
        draw.rectangle([cx - r * 0.6, cy - r * 0.7, cx - r * 0.2, cy - r * 0.5], fill=fg_color, outline=outline_color, width=1)
        draw.rectangle([cx + r * 0.2, cy - r * 0.7, cx + r * 0.6, cy - r * 0.5], fill=fg_color, outline=outline_color, width=1)
    elif upper == 'N': # Knight
        pts = [(cx - r * 0.5, cy + r * 0.5), (cx - r * 0.5, cy - r * 0.3), (cx, cy - r * 0.7), (cx + r * 0.5, cy - r * 0.2), (cx + r * 0.2, cy + r * 0.5)]
        draw.polygon(pts, fill=fg_color, outline=outline_color)
    elif upper == 'B': # Bishop
        draw.ellipse([cx - r * 0.5, cy - r * 0.7, cx + r * 0.5, cy + r * 0.3], fill=fg_color, outline=outline_color, width=3)
        draw.polygon([(cx, cy - r * 0.9), (cx - r * 0.2, cy - r * 0.6), (cx + r * 0.2, cy - r * 0.6)], fill=fg_color)
        draw.rectangle([cx - r * 0.6, cy + r * 0.3, cx + r * 0.6, cy + r * 0.6], fill=fg_color, outline=outline_color, width=2)
    elif upper == 'Q': # Queen
        draw.polygon([(cx - r * 0.7, cy - r * 0.5), (cx - r * 0.4, cy + r * 0.5), (cx + r * 0.4, cy + r * 0.5), (cx + r * 0.7, cy - r * 0.5), (cx, cy)], fill=fg_color, outline=outline_color)
        draw.ellipse([cx - r * 0.2, cy - r * 0.8, cx + r * 0.2, cy - r * 0.4], fill=fg_color, outline=outline_color, width=2)
    elif upper == 'K': # King
        draw.rectangle([cx - r * 0.6, cy - r * 0.4, cx + r * 0.6, cy + r * 0.5], fill=fg_color, outline=outline_color, width=3)
        # Cross on top
        draw.rectangle([cx - r * 0.15, cy - r * 0.85, cx + r * 0.15, cy - r * 0.4], fill=fg_color, outline=outline_color, width=1)
        draw.rectangle([cx - r * 0.4, cy - r * 0.7, cx + r * 0.4, cy - r * 0.55], fill=fg_color, outline=outline_color, width=1)

    return img

def apply_augmentations(img: Image.Image) -> Image.Image:
    """Applies brightness, contrast, noise, rotation, and slight blur to simulate camera/screenshot variety."""
    # Slight rotation
    if random.random() < 0.4:
        angle = random.uniform(-4, 4)
        img = img.rotate(angle, resample=Image.BICUBIC)

    # Slight blur or noise
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    # Color shift / Brightness variation
    arr = np.array(img).astype(np.float32)
    brightness = random.uniform(0.85, 1.15)
    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)

    # Add Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(2, 8), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)

def generate_dataset(output_dir: str, num_samples_per_class: int = 400):
    """Generates synthetic dataset organized by class subfolders."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating synthetic dataset at {output_dir} ({num_samples_per_class} samples/class)...")

    for cls in CLASSES:
        cls_dir = os.path.join(output_dir, cls if cls != '.' else 'empty')
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(num_samples_per_class):
            theme = random.choice(COLOR_THEMES)
            bg_color = random.choice(theme) # Light or dark square
            
            img = create_piece_image(cls, bg_color)
            img = apply_augmentations(img)
            
            img.save(os.path.join(cls_dir, f"{i:04d}.png"))

    print("Synthetic dataset generation complete!")

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "synthetic")
    generate_dataset(dataset_path, num_samples_per_class=300)
