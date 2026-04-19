"""
main.py — HDFN-Span Execution Pipeline
========================================
Entry point for the Hybrid Deep Fusion Network for Token-Level AI Attribution
project.  Supports three operating modes via command-line arguments:

  --mode train
      Load data, build model, run full training loop, save best checkpoint.

  --mode evaluate
      Load best checkpoint, run full evaluation suite (metrics + plots).

  --mode ablation
      Train and evaluate four architectural variants:
        1. Full              — all components enabled.
        2. No_SSS_SDS        — feature module disabled.
        3. No_Attention      — fusion attention disabled.
        4. No_GPT2           — GPT-2 encoder disabled.
      Results are printed as a Markdown table and saved to results/ablation.md.

Usage examples
──────────────
  python main.py --mode train
  python main.py --mode evaluate
  python main.py --mode ablation
  python main.py --mode train --epochs 3 --batch_size 4 --lr 1e-5
"""

import argparse
import copy
import os
import sys
from typing import Dict

from config import Config, get_config, print_config
from utils import set_seed, get_device, get_logger, ensure_dir

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="HDFN-Span",
        description=(
            "Hybrid Deep Fusion Network for Token-Level AI Attribution.\n"
            "Research paper: 'HDFN-Span: Token-Level AI Attribution via "
            "Multi-Encoder Deep Fusion.'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Mode ── #
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "ablation"],
        default="train",
        help="Execution mode: train | evaluate | ablation  (default: train)",
    )

    # ── Hyperparameter overrides ── #
    parser.add_argument("--epochs",     type=int,   default=None, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int,   default=None, help="Batch size.")
    parser.add_argument("--lr",         type=float, default=None, help="Learning rate.")
    parser.add_argument("--max_length", type=int,   default=None, help="Max tokenisation length.")
    parser.add_argument("--seed",       type=int,   default=None, help="Random seed.")

    # ── I/O overrides ── #
    parser.add_argument("--data_path",  type=str, default=None, help="Path to the CSV dataset.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to save/load model checkpoint.")
    parser.add_argument("--results_dir",type=str, default=None, help="Directory for output files.")

    # ── Ablation flags ── #
    parser.add_argument("--no_features",  action="store_true", help="Disable SSS/SDS feature module.")
    parser.add_argument("--no_attention", action="store_true", help="Disable fusion attention layer.")
    parser.add_argument("--no_gpt2",      action="store_true", help="Disable GPT-2 encoder.")

    return parser.parse_args()


def apply_args_to_config(cfg: Config, args: argparse.Namespace) -> Config:
    """
    Override :class:`~config.Config` fields with CLI argument values.

    Only non-None / non-False CLI values are applied so that defaults
    in the Config class remain effective when arguments are not provided.

    Args:
        cfg:  Base configuration to modify (in-place).
        args: Parsed CLI namespace.

    Returns:
        Modified *cfg*.
    """
    if args.epochs     is not None: cfg.epochs        = args.epochs
    if args.batch_size is not None: cfg.batch_size     = args.batch_size
    if args.lr         is not None: cfg.learning_rate  = args.lr
    if args.max_length is not None: cfg.max_length     = args.max_length
    if args.seed       is not None: cfg.seed           = args.seed
    if args.data_path  is not None: cfg.data_path      = args.data_path
    if args.model_path is not None: cfg.model_save_path = args.model_path
    if args.results_dir is not None: cfg.results_dir   = args.results_dir

    if args.no_features:  cfg.use_features  = False
    if args.no_attention: cfg.use_attention = False
    if args.no_gpt2:      cfg.use_gpt2      = False

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Mode: train
# ─────────────────────────────────────────────────────────────────────────────

def run_train(cfg: Config) -> None:
    """
    Execute the full training pipeline.

    1. Load data.
    2. Build model.
    3. Train for ``cfg.epochs`` epochs.
    4. Save best checkpoint.
    """
    from dataset import get_dataloaders
    from model import build_model
    from train import train

    device = get_device()

    logger.info("Loading data …")
    train_loader, val_loader, test_loader, tokenizers = get_dataloaders(cfg)

    logger.info("Building model …")
    model = build_model(cfg)

    logger.info("Starting training …")
    trained_model = train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    logger.info("Training complete.  Model saved to %s", cfg.model_save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Mode: evaluate
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluate(cfg: Config) -> Dict[str, float]:
    """
    Load the saved checkpoint and run the full evaluation suite.

    Returns:
        Dict of all computed metrics.
    """
    from dataset import get_dataloaders, load_dataframe
    from evaluate import evaluate

    device = get_device()

    logger.info("Loading test data …")
    train_loader, val_loader, test_loader, tokenizers = get_dataloaders(cfg)

    # Load raw test dataframe for robustness evaluation.
    import pandas as pd
    df = load_dataframe(cfg.data_path)
    df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * cfg.train_split)
    n_val   = int(n * cfg.val_split)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

    logger.info("Running evaluation suite …")
    metrics = evaluate(
        cfg=cfg,
        test_loader=test_loader,
        device=device,
        test_df=test_df,
        tokenizers=tokenizers,
    )

    # Print a clean summary table.
    print("\n" + "=" * 55)
    print("  HDFN-Span — Final Evaluation Results")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<30} {v:.4f}")
    print("=" * 55 + "\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Mode: ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(cfg: Config) -> None:
    """
    Train and evaluate four ablation variants of HDFN-Span.

    Variants
    --------
    1. Full          — all modules enabled.
    2. No_SSS_SDS    — use_features = False.
    3. No_Attention  — use_attention = False.
    4. No_GPT2       — use_gpt2 = False.

    Each variant is trained from scratch, evaluated on the test set, and
    its F1, Accuracy, and Span IoU are recorded.  Results are printed as a
    Markdown table and saved to ``results/ablation.md``.
    """
    from dataset import get_dataloaders, load_dataframe
    from model import build_model
    from train import train
    from evaluate import evaluate

    device = get_device()

    logger.info("=" * 60)
    logger.info("ABLATION STUDY — %d variants", len(cfg.ablation_variants))
    logger.info("=" * 60)

    # Load dataset once (all variants share the same split).
    logger.info("Loading dataset …")
    import pandas as pd
    df = load_dataframe(cfg.data_path)

    results: Dict[str, Dict[str, float]] = {}

    variant_configs = {
        "Full":        dict(use_features=True,  use_attention=True,  use_gpt2=True),
        "No_SSS_SDS":  dict(use_features=False, use_attention=True,  use_gpt2=True),
        "No_Attention":dict(use_features=True,  use_attention=False, use_gpt2=True),
        "No_GPT2":     dict(use_features=True,  use_attention=True,  use_gpt2=False),
    }

    for variant_name, flags in variant_configs.items():
        logger.info("\n" + "─" * 55)
        logger.info("ABLATION VARIANT: %s", variant_name)
        logger.info("─" * 55)

        # Deep-copy config and apply ablation flags.
        var_cfg = copy.deepcopy(cfg)
        var_cfg.use_features  = flags["use_features"]
        var_cfg.use_attention = flags["use_attention"]
        var_cfg.use_gpt2      = flags["use_gpt2"]

        # Give each variant its own checkpoint path.
        base, ext = os.path.splitext(cfg.model_save_path)
        var_cfg.model_save_path = f"{base}_{variant_name}{ext}"

        # Build data loaders.
        train_loader, val_loader, test_loader, tokenizers = get_dataloaders(var_cfg)

        # Build and train the variant model.
        model = build_model(var_cfg)
        trained_model = train(
            cfg=var_cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        # Evaluate.
        n = len(df)
        n_train = int(n * var_cfg.train_split)
        n_val   = int(n * var_cfg.val_split)
        test_df = df.sample(frac=1, random_state=var_cfg.seed).reset_index(drop=True)
        test_df = test_df.iloc[n_train + n_val:].reset_index(drop=True)

        metrics = evaluate(
            cfg=var_cfg,
            model=trained_model,
            test_loader=test_loader,
            device=device,
            test_df=test_df,
            tokenizers=tokenizers,
        )

        results[variant_name] = {
            "Accuracy": metrics.get("accuracy",      0.0),
            "Precision": metrics.get("precision",    0.0),
            "Recall":   metrics.get("recall",         0.0),
            "F1":       metrics.get("f1",             0.0),
            "IoU":      metrics.get("mean_span_iou",  0.0),
        }

        logger.info(
            "Variant '%s' — F1=%.4f  Acc=%.4f  IoU=%.4f",
            variant_name,
            results[variant_name]["F1"],
            results[variant_name]["Accuracy"],
            results[variant_name]["IoU"],
        )

    # ── Print Markdown table ───────────────────────────────────────────────── #
    md_lines = [
        "",
        "# HDFN-Span Ablation Study Results",
        "",
        "| Variant       | Accuracy | Precision | Recall | F1     | Span IoU |",
        "|:--------------|:--------:|:---------:|:------:|:------:|:--------:|",
    ]
    for variant_name, m in results.items():
        md_lines.append(
            f"| {variant_name:<13} "
            f"| {m['Accuracy']:.4f}   "
            f"| {m['Precision']:.4f}     "
            f"| {m['Recall']:.4f}  "
            f"| {m['F1']:.4f}  "
            f"| {m['IoU']:.4f}     |"
        )
    md_lines.append("")
    md_table = "\n".join(md_lines)

    print(md_table)

    # ── Save to file ───────────────────────────────────────────────────────── #
    ensure_dir(cfg.results_dir)
    ablation_path = os.path.join(cfg.results_dir, "ablation.md")
    with open(ablation_path, "w", encoding="utf-8") as fh:
        fh.write(md_table)
    logger.info("Ablation table saved to %s", ablation_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point — parse args, configure, and dispatch to the right mode."""
    args = parse_args()
    cfg = get_config()
    cfg = apply_args_to_config(cfg, args)

    # Set random seed for reproducibility.
    set_seed(cfg.seed)

    # Print active configuration.
    print_config(cfg)

    # Ensure results directory exists.
    ensure_dir(cfg.results_dir)

    logger.info("Mode: %s", args.mode)

    if args.mode == "train":
        run_train(cfg)

    elif args.mode == "evaluate":
        run_evaluate(cfg)

    elif args.mode == "ablation":
        run_ablation(cfg)

    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
