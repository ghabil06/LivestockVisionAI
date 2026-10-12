"""
Shared 4-Species Classifier and Out-of-Distribution Detector.
Identifies Cow, Hen, Goat, and Sheep, enforcing strict OOD rejection for non-livestock or ambiguous captures.
"""

from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from LivestockVisionAI.data.schemas.animal_record import Species


SUPPORTED_SPECIES_LIST = [Species.COW, Species.HEN, Species.GOAT, Species.SHEEP]
SPECIES_TO_IDX = {spec: idx for idx, spec in enumerate(SUPPORTED_SPECIES_LIST)}
IDX_TO_SPECIES = {idx: spec for idx, spec in enumerate(SUPPORTED_SPECIES_LIST)}


class SpeciesDetectionResult:
    """Structured result of species detection and OOD classification."""

    def __init__(
        self,
        predicted_species: Species,
        confidence: float,
        probabilities: Dict[str, float],
        is_out_of_distribution: bool,
        rejection_reason: Optional[str] = None,
    ):
        self.predicted_species = predicted_species
        self.confidence = float(confidence)
        self.probabilities = probabilities
        self.is_out_of_distribution = is_out_of_distribution
        self.rejection_reason = rejection_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_species": self.predicted_species.value,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "is_out_of_distribution": self.is_out_of_distribution,
            "rejection_reason": self.rejection_reason,
            "reliability_status": "REJECTED_OOD" if self.is_out_of_distribution else "VERIFIED",
        }


class SpeciesDetector:
    """
    4-Class Livestock Species Classifier with Confidence and Entropy OOD Gates.
    Rejects any capture where maximum probability < confidence_threshold (default 0.65).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        max_entropy_threshold: float = 1.25,
        weights_path: Optional[Path] = None,
    ):
        self.confidence_threshold = float(confidence_threshold)
        self.max_entropy_threshold = float(max_entropy_threshold)

        # Standard ImageNet pre-processing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Lightweight 4-class classifier head on ResNet-18
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)
        self.model.eval()

        self.weights_loaded = False
        default_ckpt = Path(__file__).resolve().parent.parent.parent / "models" / "checkpoints" / "species_classifier_best.pth"
        actual_path = Path(weights_path) if weights_path else default_ckpt
        self.checkpoint_path = actual_path

        if actual_path.exists():
            state = torch.load(str(actual_path), map_location="cpu")
            if any(k.startswith("backbone.fc.1.") for k in state.keys()):
                in_f = self.model.fc.in_features if isinstance(self.model.fc, nn.Linear) else 512
                self.model.fc = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_f, 4)
                )
                state = {k.replace("backbone.", ""): v for k, v in state.items()}
            elif any(k.startswith("backbone.") for k in state.keys()):
                state = {k.replace("backbone.", ""): v for k, v in state.items()}
            self.model.load_state_dict(state)
            self.weights_loaded = True
            self.model.eval()

    def _preprocess(self, image_input: Union[Image.Image, np.ndarray, str, Path, bytes]) -> torch.Tensor:
        """Preprocess diverse image input types into normalized [1, 3, 224, 224] tensor."""
        import io
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return self.transform(pil_img).unsqueeze(0)

    def predict(
        self,
        image_input: Union[Image.Image, np.ndarray, str, Path, bytes],
        species_hint: Optional[Species] = None,
    ) -> SpeciesDetectionResult:
        """
        Predict species and evaluate out-of-distribution / unknown status.
        """
        tensor = self._preprocess(image_input)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        prob_dict = {
            spec.value: float(probs[idx])
            for idx, spec in enumerate(SUPPORTED_SPECIES_LIST)
        }

        # Compute Shannon Entropy H(P)
        eps = 1e-9
        entropy = float(-np.sum(probs * np.log(probs + eps)))

        max_idx = int(np.argmax(probs))
        top_prob = float(probs[max_idx])
        top_species = IDX_TO_SPECIES[max_idx]

        # OOD Gate: Threshold checking
        if top_prob < self.confidence_threshold or entropy > self.max_entropy_threshold:
            return SpeciesDetectionResult(
                predicted_species=Species.UNKNOWN,
                confidence=top_prob,
                probabilities=prob_dict,
                is_out_of_distribution=True,
                rejection_reason=(
                    f"Unable to estimate reliably: Species classification ambiguous "
                    f"(confidence {top_prob:.2f} < {self.confidence_threshold:.2f}, entropy {entropy:.2f})."
                ),
            )

        return SpeciesDetectionResult(
            predicted_species=top_species,
            confidence=top_prob,
            probabilities=prob_dict,
            is_out_of_distribution=False,
            rejection_reason=None,
        )

    def evaluate_synthetic_logits(self, logits: np.ndarray) -> SpeciesDetectionResult:
        """Directly evaluate logits for unit testing confidence and OOD gates."""
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        prob_dict = {
            spec.value: float(probs[idx])
            for idx, spec in enumerate(SUPPORTED_SPECIES_LIST)
        }
        max_idx = int(np.argmax(probs))
        top_prob = float(probs[max_idx])
        top_species = IDX_TO_SPECIES[max_idx]

        eps = 1e-9
        entropy = float(-np.sum(probs * np.log(probs + eps)))

        is_ood = top_prob < self.confidence_threshold or entropy > self.max_entropy_threshold
        return SpeciesDetectionResult(
            predicted_species=Species.UNKNOWN if is_ood else top_species,
            confidence=top_prob,
            probabilities=prob_dict,
            is_out_of_distribution=is_ood,
            rejection_reason=(
                f"Unable to estimate reliably: Species classification ambiguous "
                f"(confidence {top_prob:.2f} < {self.confidence_threshold:.2f})"
            ) if is_ood else None,
        )
