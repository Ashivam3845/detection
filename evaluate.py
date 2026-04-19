"""
evaluate.py — Evaluation Suite for HDFN-Span
==============================================
Implements a comprehensive evaluation suite covering:

  Token-level metrics
  ───────────────────
  • Accuracy, Precision, Recall, F1-score (binary).
  • Combined confusion matrix (saved as PNG).
  • ROC curve with AUC (saved as PNG).

  Document-level metric
  ─────────────────────
  • Document AI Score = mean(p_i over real tokens)     ∈ [0, 1].
    Represents the fraction of a document estimated to be AI-generated.

  Span-level metric
  ─────────────────
  • Intersection over Union (IoU) for predicted vs. ground-truth
    contiguous AI spans.  A "span" is a maximal run of consecutive
    positions classified as AI.

  Robustness evaluation
  ─────────────────────
  • Rerun inference on adversarially augmented versions of the test set.
  • Compare token-level accuracy on original vs. augmented inputs.
  • Robustness Drop = Acc_original - Acc_augmented.

All plots and JSON metric files are written to ``cfg.results_dir``.
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, get_config
from model import HDFNSpanModel, build_model
from utils import get_logger, get_device, ensure_dir, format_metrics

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Span extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def extract_spans(binary_sequence: List[int]) -> List[Tuple[int, int]]:
    """
    Extract maximal contiguous runs of 1s from a binary sequence.

    Args:
        binary_sequence: List of 0s and 1s (length L).

    Returns:
        List of (start, end) tuples (inclusive, 0-indexed) for each
        contiguous block of 1s.

    Example::

        >>> extract_spans([0, 0, 1, 1, 1, 0, 1, 0])
        [(2, 4), (6, 6)]
    """
    spans = []
    in_span = False
    start = 0
    for i, val in enumerate(binary_sequence):
        if val == 1 and not in_span:
            in_span = True
            start = i
        elif val != 1 and in_span:
            spans.append((start, i - 1))
            in_span = False
    if in_span:
        spans.append((start, len(binary_sequence) - 1))
    return spans


def span_iou(
    pred_spans: List[Tuple[int, int]],
    true_spans: List[Tuple[int, int]],
    seq_len: int,
) -> float:
    """
    Compute token-level Intersection over Union for two sets of spans.

    Converts each set of (start, end) spans into a binary mask of length
    *seq_len*, then calculates IoU = |intersection| / |union|.

    Args:
        pred_spans: Predicted AI spans.
        true_spans: Ground-truth AI spans.
        seq_len:    Total sequence length.

    Returns:
        IoU value in [0, 1].  Returns 1.0 if both masks are all-zeros.
    """
    pred_mask = np.zeros(seq_len, dtype=bool)
    true_mask = np.zeros(seq_len, dtype=bool)

    for (s, e) in pred_spans:
        pred_mask[s: e + 1] = True
    for (s, e) in true_spans:
        true_mask[s: e + 1] = True

    intersection = (pred_mask & true_mask).sum()
    union = (pred_mask | true_mask).sum()

    if union == 0:
        return 1.0  # Both empty — perfect match.
    return float(intersection / union)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: HDFNSpanModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference over *loader* and collect predictions.

    Args:
        model:     HDFN-Span model (put in eval mode internally).
        loader:    DataLoader to iterate.
        device:    Compute device.
        threshold: Binary classification threshold for probabilities.

    Returns:
        Tuple of four 1-D NumPy arrays, each of length = total non-padding
        token positions across the entire dataset:
          * ``all_probs``   — raw predicted probabilities.
          * ``all_preds``   — thresholded binary predictions.
          * ``all_labels``  — ground-truth labels.
          * ``doc_scores``  — per-document AI score (one value per sample).
    """
    model.eval()

    all_probs: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    doc_scores: List[float] = []

    for batch in loader:
        deberta_ids  = batch["deberta_input_ids"].to(device)
        deberta_mask = batch["deberta_attention_mask"].to(device)
        roberta_ids  = batch["roberta_input_ids"].to(device)
        roberta_mask = batch["roberta_attention_mask"].to(device)
        gpt2_ids     = batch["gpt2_input_ids"].to(device)
        gpt2_mask    = batch["gpt2_attention_mask"].to(device)
        token_labels = batch["token_labels"]  # keep on CPU for indexing

        _, probs = model(
            deberta_input_ids=deberta_ids,
            deberta_attention_mask=deberta_mask,
            roberta_input_ids=roberta_ids,
            roberta_attention_mask=roberta_mask,
            gpt2_input_ids=gpt2_ids,
            gpt2_attention_mask=gpt2_mask,
        )
        probs = probs.cpu()  # (B, L)

        B, L = probs.shape
        for b in range(B):
            valid_mask = token_labels[b] != -100  # non-padding
            valid_probs  = probs[b][valid_mask].numpy()
            valid_labels = token_labels[b][valid_mask].numpy()

            all_probs.extend(valid_probs.tolist())
            all_preds.extend((valid_probs >= threshold).astype(int).tolist())
            all_labels.extend(valid_labels.tolist())
            doc_scores.append(float(valid_probs.mean()))

    return (
        np.array(all_probs),
        np.array(all_preds),
        np.array(all_labels),
        np.array(doc_scores),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Token-level metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute token-level classification metrics.

    Args:
        all_preds:  Binary predictions array.
        all_labels: Ground-truth binary labels array.

    Returns:
        Dict with keys: accuracy, precision, recall, f1.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="binary", zero_division=0),
        "f1":        f1_score(all_labels, all_preds, average="binary", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Span-level IoU metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_span_iou_from_loader(
    model: HDFNSpanModel,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    """
    Compute the mean span-level IoU over all samples in *loader*.

    Each sample's predicted and ground-truth binary label sequences are
    converted to span lists, and their token-level IoU is accumulated.

    Args:
        model:     HDFN-Span model.
        loader:    DataLoader to evaluate.
        device:    Compute device.
        threshold: Decision threshold.

    Returns:
        Mean IoU across all samples.
    """
    model.eval()
    iou_scores: List[float] = []

    with torch.no_grad():
        for batch in loader:
            deberta_ids  = batch["deberta_input_ids"].to(device)
            deberta_mask = batch["deberta_attention_mask"].to(device)
            roberta_ids  = batch["roberta_input_ids"].to(device)
            roberta_mask = batch["roberta_attention_mask"].to(device)
            gpt2_ids     = batch["gpt2_input_ids"].to(device)
            gpt2_mask    = batch["gpt2_attention_mask"].to(device)
            token_labels = batch["token_labels"]  # CPU

            _, probs = model(
                deberta_input_ids=deberta_ids,
                deberta_attention_mask=deberta_mask,
                roberta_input_ids=roberta_ids,
                roberta_attention_mask=roberta_mask,
                gpt2_input_ids=gpt2_ids,
                gpt2_attention_mask=gpt2_mask,
            )
            probs = probs.cpu()  # (B, L)

            B, L = probs.shape
            for b in range(B):
                valid_mask = token_labels[b] != -100
                seq_len = valid_mask.sum().item()

                pred_seq  = (probs[b][valid_mask] >= threshold).long().tolist()
                true_seq  = token_labels[b][valid_mask].tolist()

                pred_spans = extract_spans(pred_seq)
                true_spans = extract_spans(true_seq)

                iou = span_iou(pred_spans, true_spans, seq_len=int(seq_len))
                iou_scores.append(iou)

    return float(np.mean(iou_scores)) if iou_scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Robustness evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_robustness(
    model: HDFNSpanModel,
    test_df,
    tokenizers: Tuple,
    device: torch.device,
    cfg: Config,
) -> Dict[str, float]:
    """
    Compare accuracy on original test text vs. adversarially augmented text.

    Creates a fresh ``HDFNSpanDataset`` with ``augment=True`` over the test
    split to simulate adversarial inputs, and measures the accuracy drop.

    Args:
        model:      HDFN-Span model.
        test_df:    Test split ``pd.DataFrame``.
        tokenizers: Tuple of (deberta_tok, roberta_tok, gpt2_tok).
        device:     Compute device.
        cfg:        Project configuration.

    Returns:
        Dict with keys: original_acc, augmented_acc, robustness_drop.
    """
    from dataset import HDFNSpanDataset
    from torch.utils.data import DataLoader

    deberta_tok, roberta_tok, gpt2_tok = tokenizers

    original_ds = HDFNSpanDataset(
        test_df, deberta_tok, roberta_tok, gpt2_tok,
        max_length=cfg.max_length, augment=False
    )
    augmented_ds = HDFNSpanDataset(
        test_df, deberta_tok, roberta_tok, gpt2_tok,
        max_length=cfg.max_length, augment=True
    )

    loader_kw = dict(batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    orig_loader = DataLoader(original_ds, **loader_kw)
    aug_loader  = DataLoader(augmented_ds, **loader_kw)

    _, orig_preds, orig_labels, _ = collect_predictions(model, orig_loader, device)
    _, aug_preds,  aug_labels,  _ = collect_predictions(model, aug_loader,  device)

    from sklearn.metrics import accuracy_score
    orig_acc = accuracy_score(orig_labels, orig_preds)
    aug_acc  = accuracy_score(aug_labels,  aug_preds)

    return {
        "original_acc":     orig_acc,
        "augmented_acc":    aug_acc,
        "robustness_drop":  orig_acc - aug_acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    save_path: str,
) -> None:
    """
    Generate and save a styled confusion matrix plot.

    Args:
        all_labels: Ground-truth binary labels.
        all_preds:  Binary predictions.
        save_path:  Output file path (PNG).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Human", "AI"],
            yticklabels=["Human", "AI"],
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("HDFN-Span — Token-Level Confusion Matrix", fontsize=13, pad=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", save_path)
    except ImportError as e:
        logger.warning("Could not save confusion matrix: %s", e)


def save_roc_curve(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    save_path: str,
) -> None:
    """
    Generate and save a ROC curve plot with AUC annotation.

    Args:
        all_labels: Ground-truth binary labels.
        all_probs:  Predicted probabilities.
        save_path:  Output file path (PNG).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#4A90D9", lw=2.5,
                label=f"ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.5, linestyle="--",
                label="Random baseline")
        ax.fill_between(fpr, tpr, alpha=0.12, color="#4A90D9")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("HDFN-Span — Token-Level ROC Curve", fontsize=13, pad=12)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("ROC curve saved to %s", save_path)
    except ImportError as e:
        logger.warning("Could not save ROC curve: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Document AI Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_document_ai_score(probs: np.ndarray) -> float:
    """
    Compute the Document AI Score: the mean of per-token AI probabilities.

    Represents the overall proportion of AI-generated content in a document.

    Args:
        probs: NumPy array of per-token probabilities for a single document.

    Returns:
        Scalar float in [0, 1].
    """
    if len(probs) == 0:
        return 0.0
    return float(np.mean(probs))


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    cfg: Optional[Config] = None,
    model: Optional[HDFNSpanModel] = None,
    test_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    test_df=None,
    tokenizers: Optional[Tuple] = None,
) -> Dict[str, float]:
    """
    Run the full HDFN-Span evaluation suite.

    Generates and saves:
      • ``results/metrics.json``         — all scalar metrics.
      • ``results/confusion_matrix.png`` — confusion matrix plot.
      • ``results/roc_curve.png``        — ROC curve with AUC.

    Args:
        cfg:         Project configuration (global singleton if None).
        model:       Trained model (loaded from checkpoint if None).
        test_loader: Test DataLoader (built from cfg if None).
        device:      Compute device (auto-detected if None).
        test_df:     Raw test DataFrame (needed for robustness eval).
        tokenizers:  Tuple of tokenisers (needed for robustness eval).

    Returns:
        Dict of all computed metrics.
    """
    if cfg is None:
        cfg = get_config()
    if device is None:
        device = get_device()

    ensure_dir(cfg.results_dir)

    # ── Load model from checkpoint if needed ─────────────────────────────── #
    if model is None:
        model = build_model(cfg).to(device)
        if os.path.exists(cfg.model_save_path):
            model.load_state_dict(
                torch.load(cfg.model_save_path, map_location=device)
            )
            logger.info("Model loaded from %s", cfg.model_save_path)
        else:
            logger.warning(
                "Checkpoint not found at %s — evaluating with random weights.",
                cfg.model_save_path,
            )

    # ── Build test loader if needed ───────────────────────────────────────── #
    if test_loader is None:
        from dataset import get_dataloaders
        _, _, test_loader, tokenizers = get_dataloaders(cfg)

    # ── Collect predictions ───────────────────────────────────────────────── #
    logger.info("Running inference on test set …")
    all_probs, all_preds, all_labels, doc_scores = collect_predictions(
        model, test_loader, device
    )

    # ── Token-level metrics ───────────────────────────────────────────────── #
    token_metrics = compute_token_metrics(all_preds, all_labels)
    logger.info(format_metrics(token_metrics, title="Token-Level Metrics"))

    # ── Span-level IoU ────────────────────────────────────────────────────── #
    logger.info("Computing span-level IoU …")
    mean_iou = compute_span_iou_from_loader(model, test_loader, device)
    logger.info("  Mean Span IoU: %.4f", mean_iou)

    # ── Document AI Score (mean over all test samples) ─────────────────────── #
    mean_doc_score = float(np.mean(doc_scores))
    logger.info("  Mean Document AI Score: %.4f", mean_doc_score)

    # ── Robustness evaluation ─────────────────────────────────────────────── #
    robustness_metrics: Dict[str, float] = {}
    if test_df is not None and tokenizers is not None:
        logger.info("Running robustness evaluation …")
        robustness_metrics = compute_robustness(
            model, test_df, tokenizers, device, cfg
        )
        logger.info(format_metrics(robustness_metrics, title="Robustness Metrics"))
    else:
        logger.info("Skipping robustness eval (test_df / tokenizers not provided).")

    # ── Plots ─────────────────────────────────────────────────────────────── #
    save_confusion_matrix(
        all_labels, all_preds,
        os.path.join(cfg.results_dir, "confusion_matrix.png"),
    )
    save_roc_curve(
        all_labels, all_probs,
        os.path.join(cfg.results_dir, "roc_curve.png"),
    )

    # ── Aggregate all metrics ─────────────────────────────────────────────── #
    all_metrics = {
        **token_metrics,
        "mean_span_iou":      mean_iou,
        "mean_doc_ai_score":  mean_doc_score,
        **robustness_metrics,
    }

    # ── Save metrics to JSON ──────────────────────────────────────────────── #
    metrics_path = os.path.join(cfg.results_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info("All metrics saved to %s", metrics_path)

    return all_metrics
