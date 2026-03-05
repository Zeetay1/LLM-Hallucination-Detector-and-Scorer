from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

CALIBRATOR_PATH = Path("data") / "calibration" / "calibrator.pkl"


class CalibrationExample(BaseModel):
    """Single calibration data point: raw probability and binary label."""

    raw_score: float
    is_supported: bool


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for binary probabilities.
    """
    assert probs.shape == labels.shape
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = probs.shape[0]
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < num_bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        accuracy = bin_labels.mean()
        confidence = bin_probs.mean()
        weight = bin_probs.size / n
        ece += weight * abs(confidence - accuracy)
    return float(ece)


@dataclass
class ProbabilityCalibrator:
    """
    Logistic calibration wrapper around scikit-learn's LogisticRegression.
    """

    model: LogisticRegression

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        scores_2d = scores.reshape(-1, 1)
        calibrated = self.model.predict_proba(scores_2d)[:, 1]
        return calibrated


def fit_calibrator(examples: Iterable[CalibrationExample]) -> Tuple[ProbabilityCalibrator, float, float]:
    """
    Fit a probability calibrator on raw scores and binary labels.

    Returns the calibrator and ECE before and after calibration.
    """
    data = list(examples)
    if not data:
        raise ValueError("No calibration examples provided.")

    raw_scores = np.array([e.raw_score for e in data], dtype=float)
    labels = np.array([1.0 if e.is_supported else 0.0 for e in data], dtype=float)

    before_ece = expected_calibration_error(raw_scores, labels)

    model = LogisticRegression()
    model.fit(raw_scores.reshape(-1, 1), labels)
    calibrator = ProbabilityCalibrator(model=model)
    calibrated_scores = calibrator.calibrate(raw_scores)
    after_ece = expected_calibration_error(calibrated_scores, labels)

    return calibrator, before_ece, after_ece


def save_calibrator(calibrator: ProbabilityCalibrator, path: Path = CALIBRATOR_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)


def load_calibrator(path: Path = CALIBRATOR_PATH) -> ProbabilityCalibrator:
    calibrator: ProbabilityCalibrator = joblib.load(path)
    return calibrator

