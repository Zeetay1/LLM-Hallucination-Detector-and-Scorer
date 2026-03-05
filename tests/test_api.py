from fastapi.testclient import TestClient

from hallucination_scorer.api import app


client = TestClient(app)


def test_score_endpoint_works_with_context():
  payload = {
      "claim": "The Eiffel Tower is in Paris.",
      "context_chunks": [
          "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
      ],
  }
  response = client.post("/score", json=payload)
  assert response.status_code == 200
  data = response.json()
  assert "overall_grounding_score" in data
  assert "per_sentence" in data
  assert isinstance(data["per_sentence"], list)

