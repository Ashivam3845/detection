"""
utils.py — Shared Utility Functions for HDFN-Span
===================================================
Provides helpers for:
  • Deterministic reproducibility (seed setting across Python / NumPy / PyTorch).
  • Human-readable token-level AI attribution highlighting.
  • Device auto-detection (CUDA → MPS → CPU fallback).
  • Logging and directory management.
"""

import os
import random
import logging
from typing import List, Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Build and return a logger that writes to stdout with a timestamped format.

    Args:
        name:  Name for the logger (typically ``__name__`` of the caller).
        level: Python logging level (default ``INFO``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds across the entire software stack for reproducibility.

    Sets:
      * Python's built-in ``random`` module.
      * NumPy global RNG.
      * PyTorch CPU and all CUDA GPUs.
      * ``PYTHONHASHSEED`` env-var (affects Python hash randomisation).
      * ``torch.backends.cudnn`` flags for deterministic cuDNN kernels.

    Args:
        seed: Integer seed value (default 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic cuDNN behaviour at a small performance cost.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d.", seed)


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Auto-detect the best available compute device.

    Priority order: CUDA GPU → Apple MPS → CPU.

    Returns:
        A :class:`torch.device` pointing to the chosen backend.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using device: Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Token-level AI attribution highlighting
# ─────────────────────────────────────────────────────────────────────────────

def highlight_text(
    tokens: List[str],
    probabilities: List[float],
    threshold: float = 0.5,
) -> str:
    """
    Wrap AI-attributed tokens with ``[AI: <token>]`` markup.

    Given a list of string tokens and their corresponding per-token AI
    probability scores, returns a reconstructed string where every token
    whose probability exceeds *threshold* is wrapped in an ``[AI: ...]``
    tag.  Consecutive AI tokens are merged into a single span tag for
    readability.

    Args:
        tokens:        List of string tokens (e.g. from a tokenizer's
                       ``convert_ids_to_tokens`` call).
        probabilities: Per-token AI probability in [0, 1].  Must be the
                       same length as *tokens*.
        threshold:     Decision boundary above which a token is classified
                       as AI-generated (default 0.5).

    Returns:
        A single string with AI-generated spans wrapped in ``[AI: ...]``
        tags and human-generated tokens left as-is.

    Example::

        >>> tokens = ["The", "quick", "brown", "fox"]
        >>> probs  = [0.2,   0.8,    0.9,    0.1  ]
        >>> highlight_text(tokens, probs, threshold=0.5)
        'The [AI: quick brown] fox'
    """
    if len(tokens) != len(probabilities):
        raise ValueError(
            f"tokens ({len(tokens)}) and probabilities ({len(probabilities)}) "
            "must have the same length."
        )

    output_parts: List[str] = []
    ai_span: List[str] = []

    def _flush_span() -> None:
        """Flush an accumulated AI span into output_parts."""
        if ai_span:
            output_parts.append(f"[AI: {' '.join(ai_span)}]")
            ai_span.clear()

    for token, prob in zip(tokens, probabilities):
        # Strip special tokeniser artefacts (e.g., "Ġ" from BPE, "##" from WP).
        clean_token = token.replace("Ġ", "").replace("▁", "").lstrip("##").strip()
        if not clean_token:
            continue
        if prob >= threshold:
            ai_span.append(clean_token)
        else:
            _flush_span()
            output_parts.append(clean_token)

    _flush_span()  # Flush any trailing AI span.
    return " ".join(output_parts)


# ─────────────────────────────────────────────────────────────────────────────
# File-system helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """
    Create *path* (and all intermediate directories) if it does not exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
    logger.debug("Directory ensured: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Metric formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics(metrics: dict, title: str = "Evaluation Metrics") -> str:
    """
    Format a flat dictionary of metric name → value into a readable string.

    Args:
        metrics: Dict mapping metric names to scalar values.
        title:   Header line for the formatted block.

    Returns:
        Multi-line formatted string.
    """
    lines = [f"\n{'─'*50}", f"  {title}", f"{'─'*50}"]
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {name:<25} {value:.4f}")
        else:
            lines.append(f"  {name:<25} {value}")
    lines.append(f"{'─'*50}\n")
    return "\n".join(lines)
