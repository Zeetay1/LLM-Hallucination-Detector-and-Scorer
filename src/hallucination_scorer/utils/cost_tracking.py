"""Cost tracking for external model API calls. Metadata attached to every response and persisted with results."""

from __future__ import annotations

from pathlib import Path

from ..config import CONFIG
from ..schemas import CostMetadata, GroundingResponse


def zero_cost_metadata() -> CostMetadata:
    """Return cost metadata for local-only pipeline (no external API call)."""
    return CostMetadata(tokens_in=0, tokens_out=0, estimated_cost=0.0)


def compute_cost_metadata(tokens_in: int, tokens_out: int) -> CostMetadata:
    """
    Compute CostMetadata for an external model API call using config pricing.

    Every call to an external model API must log tokens_in, tokens_out and
    estimated_cost via this function (or zero_cost_metadata when no external call).
    """
    cfg = CONFIG.cost_tracking
    cost = (tokens_in / 1000.0) * cfg.external_api_input_cost_per_1k + (
        tokens_out / 1000.0
    ) * cfg.external_api_output_cost_per_1k
    return CostMetadata(tokens_in=tokens_in, tokens_out=tokens_out, estimated_cost=cost)


def persist_response(path: Path | str, response: GroundingResponse) -> None:
    """
    Persist a GroundingResponse (including cost_metadata) to JSON.

    Ensures cost metadata is stored alongside results as required.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
