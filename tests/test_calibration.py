import numpy as np

from hallucination_scorer.calibration import (
    CalibrationExample,
    expected_calibration_error,
    fit_calibrator,
    load_calibrator,
    save_calibrator,
)


def test_ece_decreases_after_calibration(tmp_path, monkeypatch):
    # Synthetic data: clearly separable so calibration improves ECE
    raw_scores = np.array(
        [0.1, 0.15, 0.2, 0.25, 0.3] * 20 + [0.7, 0.75, 0.8, 0.85, 0.9] * 20,
        dtype=float,
    )
    labels = np.array([0] * 100 + [1] * 100, dtype=float)

    before_ece = expected_calibration_error(raw_scores, labels)

    examples = [
        CalibrationExample(raw_score=float(s), is_supported=bool(l))
        for s, l in zip(raw_scores, labels)
    ]
    calibrator, train_after_ece, _ = fit_calibrator(examples)

    path = tmp_path / "calibrator.pkl"
    from hallucination_scorer import calibration as calibration_module
    monkeypatch.setattr(calibration_module, "CALIBRATOR_PATH", path)

    save_calibrator(calibrator, path=path)
    loaded = load_calibrator(path=path)

    calibrated_scores = loaded.calibrate(raw_scores)
    final_ece = expected_calibration_error(calibrated_scores, labels)

    assert np.all((calibrated_scores >= 0) & (calibrated_scores <= 1))
    assert final_ece <= train_after_ece + 1e-6
    assert final_ece < before_ece

