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
class GlobalConfig:
    random_seed: int = 42
    nli: NLIConfig = NLIConfig()
    scoring: ScoringConfig = ScoringConfig()


CONFIG = GlobalConfig()

