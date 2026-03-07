from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NLIConfig:
    model_name: str = "distilroberta-base-mnli"
    entailment_label_id: int = 2  # for distilroberta-base-mnli: 0=contradiction,1=neutral,2=entailment
    max_length: int = 256


@dataclass(frozen=True)
class ScoringConfig:
    entailment_threshold: float = 0.6
    default_retrieval_k: int = 5


@dataclass(frozen=True)
class CostTrackingConfig:
    """Per-token pricing for external model APIs (e.g. OpenAI). Used by cost_tracking."""

    # Placeholder / example: input cost per 1K tokens, output cost per 1K tokens
    external_api_input_cost_per_1k: float = 0.001
    external_api_output_cost_per_1k: float = 0.002


@dataclass(frozen=True)
class GlobalConfig:
    random_seed: int = 42
    nli: NLIConfig = NLIConfig()
    scoring: ScoringConfig = ScoringConfig()
    cost_tracking: CostTrackingConfig = CostTrackingConfig()


CONFIG = GlobalConfig()

