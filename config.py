"""
config.py — HDFN-Span Project Configuration
============================================
Centralised configuration for the Hybrid Deep Fusion Network for
Token-Level AI Attribution project.  All hyperparameters, model
identifiers, file paths, and ablation flags live here so that every
other module can import a single shared ``Config`` object.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    #  Model identifiers (HuggingFace Hub)                                #
    # ------------------------------------------------------------------ #
    deberta_model_name: str = "microsoft/deberta-v3-base"
    roberta_model_name: str = "roberta-base"
    gpt2_model_name: str = "gpt2"
    sbert_model_name: str = "all-MiniLM-L6-v2"

    # ------------------------------------------------------------------ #
    #  Training hyperparameters                                            #
    # ------------------------------------------------------------------ #
    batch_size: int = 8
    learning_rate: float = 1e-5
    epochs: int = 10
    max_length: int = 512
    warmup_ratio: float = 0.1          # fraction of total steps for lr warmup
    weight_decay: float = 0.01
    gradient_clip_max_norm: float = 0.5

    # ------------------------------------------------------------------ #
    #  Loss weights                                                        #
    # ------------------------------------------------------------------ #
    lambda_smoothness: float = 0.1     # weight for span-consistency loss
    lambda_adversarial: float = 0.5    # weight for adversarial robustness term

    # ------------------------------------------------------------------ #
    #  Ablation flags                                                      #
    # ------------------------------------------------------------------ #
    use_features: bool = True          # include SSS / SDS feature module
    use_attention: bool = True         # include deep-fusion MultiheadAttention
    use_gpt2: bool = True              # include GPT-2 tertiary encoder

    # ------------------------------------------------------------------ #
    #  Architecture dimensions                                             #
    # ------------------------------------------------------------------ #
    # DeBERTa-v3-base hidden size
    deberta_hidden: int = 768
    # RoBERTa-base hidden size
    roberta_hidden: int = 768
    # GPT-2 hidden size
    gpt2_hidden: int = 768
    # Projected feature embedding dim (SSS + SDS → feature_dim)
    feature_hidden: int = 64
    # Fusion projection output dim
    fusion_hidden: int = 512
    # Number of attention heads in deep-fusion layer
    num_attention_heads: int = 8
    # Dropout probability applied after fusion projection
    dropout: float = 0.1

    # ------------------------------------------------------------------ #
    #  Data paths                                                          #
    # ------------------------------------------------------------------ #
    data_path: str = "ai_human_content_detection_dataset.csv"
    model_save_path: str = "hdfn_span_best.pt"
    results_dir: str = "results"

    # ------------------------------------------------------------------ #
    #  Dataset splits                                                      #
    # ------------------------------------------------------------------ #
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10

    # ------------------------------------------------------------------ #
    #  Reproducibility                                                     #
    # ------------------------------------------------------------------ #
    seed: int = 42

    # ------------------------------------------------------------------ #
    #  Ablation variant definitions (used by main.py)                     #
    # ------------------------------------------------------------------ #
    ablation_variants: List[str] = field(default_factory=lambda: [
        "Full",
        "No_SSS_SDS",
        "No_Attention",
        "No_GPT2",
    ])


# Module-level singleton so every import gets the same object.
cfg = Config()


def get_config() -> Config:
    """Return the global singleton :class:`Config` instance."""
    return cfg


def print_config(config: Config) -> None:
    """Pretty-print all configuration values."""
    print("\n" + "=" * 60)
    print("  HDFN-Span Configuration")
    print("=" * 60)
    for key, value in config.__dict__.items():
        print(f"  {key:<30} = {value}")
    print("=" * 60 + "\n")

