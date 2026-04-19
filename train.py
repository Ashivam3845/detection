"""
train.py — Training Loop for HDFN-Span
========================================
Implements:
  • Compound loss = Token-level BCE + λ · Span-Consistency (smoothness) loss.
  • AdamW optimiser with linear warm-up + linear decay scheduler.
  • Gradient clipping (max_norm = 1.0).
  • Per-epoch train / validation cycling with best-model checkpointing
    based on validation F1 score.
  • Detailed, timestamped logging of loss components and metrics.

Loss formulation
────────────────
  L_bce   = BCELoss(p_i, y_i)      over non-padding positions
  L_smooth = (1/T) * Σ_i (p_i - p_{i+1})²   — penalises abrupt probability jumps
  L_total  = L_bce + λ_smooth * L_smooth

The smoothness term encourages contiguous AI-span predictions rather than
noisy per-token fluctuations, which is the core inductive bias of the
HDFN-Span research contribution.
"""

import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import Config, get_config
from model import HDFNSpanModel, build_model
from utils import get_logger, get_device, format_metrics

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

def bce_loss_masked(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Token-level Binary Cross-Entropy Loss, ignoring padding positions.

    Uses ``binary_cross_entropy_with_logits`` which fuses sigmoid + BCE in a
    numerically stable log-sum-exp form, avoiding the CUDA assertion that fires
    when ``F.binary_cross_entropy`` receives values that land outside [0, 1]
    due to floating-point rounding.

    Padding positions in *labels* are marked with -100 (standard HuggingFace
    convention).

    Args:
        logits: Raw pre-sigmoid scores, shape ``(B, L)``.
        labels: Ground-truth token labels, shape ``(B, L)``, values in {0, 1, -100}.

    Returns:
        Scalar mean BCE loss over non-padding positions.
    """
    mask = labels != -100                          # (B, L) bool
    valid_logits = logits[mask]                    # (N_valid,)
    valid_labels = labels[mask].float()            # (N_valid,)
    if valid_logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.binary_cross_entropy_with_logits(valid_logits, valid_labels)


def span_smoothness_loss(
    probs: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Span Consistency (Smoothness) Loss.

    Penalises abrupt changes in consecutive token probabilities within the
    non-padding region.  Encourages the model to produce smoothly-varying,
    contiguous span predictions.

    Formula:
        L_smooth = (1 / (T - 1)) * Σ_{i=0}^{T-2} (p_i - p_{i+1})²

    where T is the number of real (non-padding) tokens.

    Args:
        probs:          Predicted probabilities ``(B, L)``.
        attention_mask: Binary mask ``(B, L)``, 1 = real token, 0 = padding.

    Returns:
        Scalar smoothness loss averaged over the batch.
    """
    # Compute squared differences between adjacent positions.
    diff = (probs[:, :-1] - probs[:, 1:]) ** 2   # (B, L-1)

    # Only consider positions where BOTH tokens are real.
    valid_pair = (attention_mask[:, :-1] == 1) & (attention_mask[:, 1:] == 1)  # (B, L-1)

    if valid_pair.sum() == 0:
        return torch.tensor(0.0, device=probs.device, requires_grad=True)

    return diff[valid_pair].mean()


def compute_total_loss(
    logits: torch.Tensor,
    probs: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    lambda_smoothness: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the compound HDFN-Span loss.

    Args:
        logits: Raw pre-sigmoid model outputs ``(B, L)`` — used for stable BCE.
        probs:  Sigmoid probabilities ``(B, L)`` — used for smoothness term.
        labels: Ground-truth token labels ``(B, L)``.
        attention_mask: Binary mask ``(B, L)`` from DeBERTa.
        lambda_smoothness: Weight for the smoothness loss term.

    Returns:
        Tuple of (total_loss, bce_loss, smooth_loss) — all scalar tensors.
    """
    # Force float32 for all loss computations to prevent NaN collapse.
    logits_f32 = logits.float()
    probs_f32 = probs.float()

    l_bce = bce_loss_masked(logits_f32, labels)
    l_smooth = span_smoothness_loss(probs_f32, attention_mask)
    l_total = l_bce + lambda_smoothness * l_smooth
    return l_total, l_bce, l_smooth


# ─────────────────────────────────────────────────────────────────────────────
# Training & Validation Steps
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: HDFNSpanModel,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg: Config,
    epoch: int,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Args:
        model:     HDFN-Span model.
        loader:    Training DataLoader.
        optimiser: Initialised AdamW optimiser.
        scheduler: HuggingFace linear LR scheduler.
        device:    Compute device.
        cfg:       Project configuration.
        epoch:     Current epoch index (1-indexed, for logging).

    Returns:
        Dict with keys ``loss``, ``bce``, ``smooth`` — all averaged over the
        epoch.
    """
    model.train()

    # Use bfloat16 autocast on CUDA (supported on Ampere+ like RTX 4060).
    # bf16 does not require a GradScaler unlike fp16.
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    total_loss = total_bce = total_smooth = 0.0
    valid_steps = 0
    n_batches = len(loader)
    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        # ── Move batch to device ─────────────────────────────────────────── #
        deberta_ids  = batch["deberta_input_ids"].to(device)
        deberta_mask = batch["deberta_attention_mask"].to(device)
        roberta_ids  = batch["roberta_input_ids"].to(device)
        roberta_mask = batch["roberta_attention_mask"].to(device)
        gpt2_ids     = batch["gpt2_input_ids"].to(device)
        gpt2_mask    = batch["gpt2_attention_mask"].to(device)
        token_labels = batch["token_labels"].to(device)

        texts = None

        # ── Forward pass (bfloat16 autocast) ─────────────────────────────── #
        optimiser.zero_grad()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits, probs = model(
                deberta_input_ids=deberta_ids,
                deberta_attention_mask=deberta_mask,
                roberta_input_ids=roberta_ids,
                roberta_attention_mask=roberta_mask,
                gpt2_input_ids=gpt2_ids,
                gpt2_attention_mask=gpt2_mask,
                texts=texts,
            )  # (B, L) each

            loss, l_bce, l_smooth = compute_total_loss(
                logits, probs, token_labels, deberta_mask, cfg.lambda_smoothness
            )

        # ── NaN guard — skip corrupt batches ────────────────────────────── #
        if not torch.isfinite(loss):
            logger.warning("Step %d: non-finite loss (%.4g) — skipping batch.",
                           step, loss.item())
            continue

        # ── Backward + gradient clipping (bf16 needs no scaler) ──────────── #
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip_max_norm)
        optimiser.step()
        scheduler.step()

        total_loss   += loss.item()
        total_bce    += l_bce.item()
        total_smooth += l_smooth.item()
        valid_steps  += 1

        if step % max(1, n_batches // 5) == 0 or step == n_batches:
            elapsed = time.time() - t0
            denom = max(valid_steps, 1)
            logger.info(
                "Epoch %d  [%d/%d]  loss=%.4f  bce=%.4f  smooth=%.4f  time=%.1fs",
                epoch, step, n_batches,
                total_loss / denom, total_bce / denom, total_smooth / denom,
                elapsed,
            )

    denom = max(valid_steps, 1)
    return {
        "loss":   total_loss   / denom,
        "bce":    total_bce    / denom,
        "smooth": total_smooth / denom,
    }


@torch.no_grad()
def validate(
    model: HDFNSpanModel,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> Dict[str, float]:
    """
    Run full validation pass and compute loss + F1 score.

    Token predictions are thresholded at 0.5 to obtain binary labels.

    Args:
        model:  HDFN-Span model (eval mode set internally).
        loader: Validation DataLoader.
        device: Compute device.
        cfg:    Project configuration.

    Returns:
        Dict with keys ``loss``, ``bce``, ``smooth``, ``f1``, ``acc``.
    """
    from sklearn.metrics import f1_score, accuracy_score

    model.eval()

    total_loss = total_bce = total_smooth = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        deberta_ids  = batch["deberta_input_ids"].to(device)
        deberta_mask = batch["deberta_attention_mask"].to(device)
        roberta_ids  = batch["roberta_input_ids"].to(device)
        roberta_mask = batch["roberta_attention_mask"].to(device)
        gpt2_ids     = batch["gpt2_input_ids"].to(device)
        gpt2_mask    = batch["gpt2_attention_mask"].to(device)
        token_labels = batch["token_labels"].to(device)

        logits, probs = model(
            deberta_input_ids=deberta_ids,
            deberta_attention_mask=deberta_mask,
            roberta_input_ids=roberta_ids,
            roberta_attention_mask=roberta_mask,
            gpt2_input_ids=gpt2_ids,
            gpt2_attention_mask=gpt2_mask,
        )  # (B, L) each

        loss, l_bce, l_smooth = compute_total_loss(
            logits, probs, token_labels, deberta_mask, cfg.lambda_smoothness
        )
        total_loss   += loss.item()
        total_bce    += l_bce.item()
        total_smooth += l_smooth.item()

        # Flatten and collect non-padding predictions.
        mask = token_labels != -100
        preds = (probs[mask] >= 0.5).long().cpu().tolist()
        targets = token_labels[mask].cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(targets)

    n = len(loader)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    return {
        "loss":   total_loss   / n,
        "bce":    total_bce    / n,
        "smooth": total_smooth / n,
        "f1":     f1,
        "acc":    acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg: Optional[Config] = None,
    model: Optional[HDFNSpanModel] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
) -> HDFNSpanModel:
    """
    Run the full HDFN-Span training pipeline.

    When called without arguments (from ``main.py``), all components are
    initialised internally using the global config and data pipeline.

    Args:
        cfg:          Project configuration (uses global singleton if None).
        model:        Pre-built model (built from cfg if None).
        train_loader: Training DataLoader (built from cfg if None).
        val_loader:   Validation DataLoader (built from cfg if None).
        device:       Compute device (auto-detected if None).

    Returns:
        The best :class:`HDFNSpanModel` (loaded from checkpoint) after all
        epochs complete.
    """
    if cfg is None:
        cfg = get_config()
    if device is None:
        device = get_device()

    # ── Build data loaders if not supplied ──────────────────────────────── #
    if train_loader is None or val_loader is None:
        from dataset import get_dataloaders
        train_loader, val_loader, _, _ = get_dataloaders(cfg)

    # ── Build model if not supplied ──────────────────────────────────────── #
    if model is None:
        model = build_model(cfg)
    model = model.to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────── #
    # Use separate weight-decay groups: no decay for biases and layer norms.
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimiser = torch.optim.AdamW(param_groups, lr=cfg.learning_rate, eps=1e-6)

    # ── Scheduler — linear warm-up → linear decay ─────────────────────────── #
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(
        "Training config — epochs=%d  total_steps=%d  warmup_steps=%d  lr=%.2e",
        cfg.epochs, total_steps, warmup_steps, cfg.learning_rate,
    )

    # ── Ensure results directory exists ──────────────────────────────────── #
    from utils import ensure_dir
    ensure_dir(cfg.results_dir)

    # ── Training loop ─────────────────────────────────────────────────────── #
    best_val_f1 = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        logger.info("=" * 60)
        logger.info("EPOCH %d / %d", epoch, cfg.epochs)
        logger.info("=" * 60)

        train_metrics = train_one_epoch(
            model, train_loader, optimiser, scheduler, device, cfg, epoch
        )
        val_metrics = validate(model, val_loader, device, cfg)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_acc": val_metrics["acc"],
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %d summary — train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | val_acc=%.4f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["f1"],
            val_metrics["acc"],
        )

        # ── Checkpoint best model ─────────────────────────────────────────── #
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), cfg.model_save_path)
            logger.info(
                "✔ New best model saved  (val_f1=%.4f)  → %s",
                best_val_f1,
                cfg.model_save_path,
            )

    logger.info(
        "Training complete.  Best val F1 = %.4f at epoch %d.",
        best_val_f1,
        best_epoch,
    )

    # ── Restore best weights ─────────────────────────────────────────────── #
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    logger.info("Best checkpoint reloaded from %s", cfg.model_save_path)

    # ── Save training history ─────────────────────────────────────────────── #
    import json
    history_path = os.path.join(cfg.results_dir, "training_history.json")
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2)
    logger.info("Training history saved to %s", history_path)

    return model
