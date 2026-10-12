"""
Species Recognition Benchmark & Confusion Matrix Evaluator.
Evaluates 4-species classification accuracy, confusion matrix, and OOD rejection rates.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LivestockVisionAI.models.shared.species_detector import SpeciesDetector
from LivestockVisionAI.data.schemas.animal_record import Species
from LivestockVisionAI.data.dataset_path_resolver import resolve_image_path


def evaluate_species_classifier_on_manifest(
    manifest_path: Path,
    output_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)

    detector = SpeciesDetector(confidence_threshold=0.65)
    species_order = ["COW", "HEN", "GOAT", "SHEEP"]

    y_true = []
    y_pred = []
    rejected_count = 0

    # Test each record
    for _, row in df.iterrows():
        true_spec = str(row["species"]).upper()
        y_true.append(true_spec)

        # Real computer vision model inference on actual image capture
        img_path = resolve_image_path(str(row["view_0_path"]), species=true_spec)
        res = detector.predict(img_path)
        if res.is_out_of_distribution or res.predicted_species == Species.UNKNOWN:
            y_pred.append("REJECTED_OOD")
            rejected_count += 1
        else:
            y_pred.append(res.predicted_species.value)

    # Compute confusion matrix for non-rejected
    valid_mask = [p != "REJECTED_OOD" for p in y_pred]
    y_true_valid = [t for t, m in zip(y_true, valid_mask) if m]
    y_pred_valid = [p for p, m in zip(y_pred, valid_mask) if m]

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=species_order)
    report = classification_report(y_true_valid, y_pred_valid, labels=species_order, output_dict=True)

    results = {
        "total_samples": len(df),
        "rejected_ood": rejected_count,
        "rejection_rate": round(rejected_count / len(df), 4),
        "confusion_matrix": cm.tolist(),
        "species_order": species_order,
        "classification_report": report,
    }

    if output_report_path:
        with open(output_report_path, "w") as f:
            f.write("# Batch B3: Species Recognition MVP Benchmark Report\n\n")
            f.write(f"- Total Evaluated Records: {len(df)}\n")
            f.write(f"- OOD Rejection Count: {rejected_count} ({results['rejection_rate']*100:.1f}%)\n")
            f.write(f"- Overall Accuracy (on Accepted Captures): {report['accuracy']*100:.2f}%\n\n")
            f.write("### Confusion Matrix (Rows: True, Cols: Predicted)\n\n")
            f.write("| Species | " + " | ".join(species_order) + " |\n")
            f.write("| :--- | " + " | ".join([":---:"] * 4) + " |\n")
            for idx, spec in enumerate(species_order):
                row_str = " | ".join(str(cm[idx][c]) for c in range(4))
                f.write(f"| **{spec}** | {row_str} |\n")
            f.write("\n### Per-Species Metrics\n\n")
            f.write("| Species | Precision | Recall | F1-Score | Support |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for spec in species_order:
                m = report[spec]
                f.write(f"| **{spec}** | {m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['f1-score']*100:.1f}% | {m['support']} |\n")

    return results


if __name__ == "__main__":
    man_path = Path(__file__).resolve().parent.parent.parent / "data" / "manifests" / "master_livestock_manifest.csv"
    rep_path = Path(__file__).resolve().parent.parent.parent / "docs" / "B3_species_recognition_report.md"
    evaluate_species_classifier_on_manifest(man_path, rep_path)
    print(f"Generated species benchmark report at {rep_path}")
