"""
Species Classifier Training Pipeline.
Trains ResNet-18 backbone on real multi-species livestock captures (Cow, Hen, Goat, Sheep)
using strict animal-isolated grouped validation, and saves verified PyTorch checkpoint (.pth).
"""

from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PARENT_DIR = PROJECT_ROOT.parent
for p in [str(PARENT_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from LivestockVisionAI.data.dataset_loader import SingleViewLivestockDataset, livestock_collate_fn
from LivestockVisionAI.data.schemas.animal_record import Species

SPECIES_MAP = {
    "COW": 0,
    "HEN": 1,
    "GOAT": 2,
    "SHEEP": 3,
}


class SpeciesClassifierModel(nn.Module):
    """ResNet-18 4-class species classifier."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feats, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def train_species_classifier(
    manifest_path: Path,
    output_checkpoint_path: Path,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> dict:
    manifest_path = Path(manifest_path)
    output_checkpoint_path = Path(output_checkpoint_path)
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Folds 0, 1, 2, 3 for train; fold 4 for held-out validation
    train_ds = SingleViewLivestockDataset(
        manifest_path=manifest_path,
        folds=[0, 1, 2, 3],
        transform=train_transform,
        strict=True,
    )
    val_ds = SingleViewLivestockDataset(
        manifest_path=manifest_path,
        folds=[4],
        transform=val_transform,
        strict=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=livestock_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=livestock_collate_fn)

    print(f"Dataset partitioned: {len(train_ds)} train captures, {len(val_ds)} validation captures.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeciesClassifierModel(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = torch.tensor([SPECIES_MAP[s] for s in batch["species"]], dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = torch.tensor([SPECIES_MAP[s] for s in batch["species"]], dtype=torch.long, device=device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * len(labels)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss/total:.4f}, Train Acc: {train_acc*100:.1f}% | Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()

    # Save checkpoint
    torch.save(best_weights or model.state_dict(), str(output_checkpoint_path))
    print(f"Saved trained species checkpoint to: {output_checkpoint_path} (Best Val Acc: {best_val_acc*100:.1f}%)")

    return {
        "best_val_accuracy": float(best_val_acc),
        "checkpoint_path": str(output_checkpoint_path),
        "total_train_samples": len(train_ds),
        "total_val_samples": len(val_ds),
    }


if __name__ == "__main__":
    man_path = PROJECT_ROOT / "data" / "manifests" / "master_livestock_manifest.csv"
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "species_classifier_best.pth"
    res = train_species_classifier(man_path, ckpt_path, epochs=4, batch_size=32)
    print("Training finished:", res)
