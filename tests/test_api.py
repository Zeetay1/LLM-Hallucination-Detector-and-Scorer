from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from hallucination_scorer.api import app

client = TestClient(app)


def test_score_endpoint_works_with_context():
    """POST /score returns 200 and GroundingResponse shape. NLI mocked for no live model load."""
    payload = {
        "claim": "The Eiffel Tower is in Paris.",
        "context_chunks": [
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
        ],
    }
    with patch("hallucination_scorer.scoring.get_nli_model") as mock_nli:
        mock_model = mock_nli.return_value
        mock_model.predict_entailment_matrix.return_value = np.array([[0.9]])
        response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_grounding_score" in data
    assert "per_sentence" in data
    assert "cost_metadata" in data
    assert isinstance(data["per_sentence"], list)

