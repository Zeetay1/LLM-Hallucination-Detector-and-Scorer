import numpy as np

from hallucination_scorer.calibration import (
    CalibrationExample,
    expected_calibration_error,
    fit_calibrator,
    load_calibrator,
    save_calibrator,
)


def test_ece_decreases_after_calibration(tmp_path, monkeypatch):
    # Synthetic data: slightly miscalibrated scores
    raw_scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    labels = np.array([0, 0, 1, 1], dtype=float)

    before_ece = expected_calibration_error(raw_scores, labels)

    examples = [
        CalibrationExample(raw_score=float(s), is_supported=bool(l))
        for s, l in zip(raw_scores, labels)
    ]
    calibrator, _, after_ece = fit_calibrator(examples)

    # Save and reload calibrator to verify persistence.
    path = tmp_path / "calibrator.pkl"

    # Override default path for this test only.
    from hallucination_scorer import calibration as calibration_module

    monkeypatch.setattr(calibration_module, "CALIBRATOR_PATH", path)

    save_calibrator(calibrator, path=path)
    loaded = load_calibrator(path=path)

    calibrated_scores = loaded.calibrate(raw_scores)
    final_ece = expected_calibration_error(calibrated_scores, labels)

    assert final_ece <= after_ece + 1e-6
    assert final_ece < before_ece

