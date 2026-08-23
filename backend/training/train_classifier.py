import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import onnx
import onnxruntime as ort
import numpy as np

CLASSES = ['.', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
FOLDER_TO_CLASS = {cls if cls != '.' else 'empty': cls for cls in CLASSES}

class ChessPieceCNN(nn.Module):
    """Lightweight 4-layer Convolutional Neural Network for 13-class chess piece classification."""
    def __init__(self, num_classes=13):
        super(ChessPieceCNN, self).__init__()
        self.features = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 32 x 32 x 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 128 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ChessSquareDataset(Dataset):
    """Dataset loader supporting synthetic and real user-provided image subdirectories."""
    def __init__(self, root_dirs, transform=None):
        self.samples = []
        self.transform = transform

        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                continue
            for folder in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder)
                if not os.path.isdir(folder_path):
                    continue
                
                cls = FOLDER_TO_CLASS.get(folder, folder)
                if cls not in CLASS_TO_IDX:
                    continue
                label_idx = CLASS_TO_IDX[cls]

                for fname in os.listdir(folder_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(folder_path, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_and_export(data_dirs, output_onnx_path, epochs=5, batch_size=64, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using training device: {device}")

    dataset = ChessSquareDataset(data_dirs, transform=get_transforms())
    if len(dataset) == 0:
        raise ValueError(f"No image samples found in directories: {data_dirs}")

    print(f"Total dataset size: {len(dataset)} images across {len(CLASSES)} classes.")
    
    # Train / Val split
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ChessPieceCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # Export to ONNX
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"Exported trained ONNX model to: {output_onnx_path}")

    # Save classes JSON metadata
    idx_to_class_dict = {str(i): c for i, c in enumerate(CLASSES)}
    meta_path = os.path.join(os.path.dirname(output_onnx_path), "classes.json")
    with open(meta_path, 'w') as f:
        json.dump({"classes": CLASSES, "idx_to_class": idx_to_class_dict}, f, indent=2)
    print(f"Saved class metadata to: {meta_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    synthetic_dir = os.path.join(base_dir, "..", "dataset", "synthetic")
    real_dir = os.path.join(base_dir, "..", "dataset", "real") # Placeholder for future user real data
    output_onnx = os.path.join(base_dir, "..", "models", "piece_classifier.onnx")

    train_and_export([synthetic_dir, real_dir], output_onnx, epochs=6)
