"""
Cattle Multi-View Weight Regression Training Pipeline.
Trains MultiViewWeightRegressionModel on verified scale weights (510 Bengal Meat animals)
with animal-isolated fold splitting, saving the best trained model checkpoint (.pth).
"""

from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PARENT_DIR = PROJECT_ROOT.parent
for p in [str(PARENT_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from LivestockVisionAI.models.cattle.weight_estimator import MultiViewWeightRegressionModel
from LivestockVisionAI.data.dataset_loader import MultiViewLivestockDataset, livestock_collate_fn


def train_cattle_weight_model(
    manifest_path: Path,
    output_checkpoint_path: Path,
    target_mode: str = "linear",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> dict:
    manifest_path = Path(manifest_path)
    output_checkpoint_path = Path(output_checkpoint_path)
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    use_log = (target_mode.lower() == "log")

    train_ds = MultiViewLivestockDataset(
        manifest_path=manifest_path,
        species="COW",
        folds=[0, 1, 2, 3],
        use_log_target=use_log,
        strict=True,
    )
    val_ds = MultiViewLivestockDataset(
        manifest_path=manifest_path,
        species="COW",
        folds=[4],
        use_log_target=use_log,
        strict=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=livestock_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=livestock_collate_fn)

    print(f"Cattle Dataset partitioned: {len(train_ds)} train animals, {len(val_ds)} validation animals.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiViewWeightRegressionModel(target_mode=target_mode).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_mae = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        total_train = 0

        for batch in train_loader:
            images = batch["images"].to(device)  # [B, 4, 3, 224, 224]
            targets = batch["weight_target"].to(device).unsqueeze(1)  # [B, 1]

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            total_train += len(targets)

        # Validation
        model.eval()
        val_errors = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                raw_weights = batch["raw_weight_kg"]
                preds = model(images).squeeze(1).cpu().numpy()

                for pred_val, true_val in zip(preds, raw_weights):
                    val_errors.append(abs(float(pred_val) - float(true_val)))

        val_mae = float(np.mean(val_errors))
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss/total_train:.4f} | Held-out Val MAE: {val_mae:.2f} kg")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = model.state_dict().copy()

    # Save checkpoint
    torch.save(best_weights or model.state_dict(), str(output_checkpoint_path))
    print(f"Saved trained cattle weight checkpoint to: {output_checkpoint_path} (Best Val MAE: {best_val_mae:.2f} kg)")

    return {
        "best_val_mae_kg": float(best_val_mae),
        "checkpoint_path": str(output_checkpoint_path),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }


if __name__ == "__main__":
    man_path = PROJECT_ROOT / "data" / "manifests" / "cattle_manifest.csv"
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "cattle_weight_multiview.pth"
    res = train_cattle_weight_model(man_path, ckpt_path, epochs=3, batch_size=16)
    print("Training finished:", res)
