"""
dataset.py — HDFN-Span Dataset & Data Pipeline
================================================
Implements:
  • ``HDFNSpanDataset`` — a PyTorch Dataset that loads the CSV, tokenises text
    with three separate tokenisers (DeBERTa, RoBERTa, GPT-2), generates
    token-level labels, and applies dynamic adversarial augmentation.
  • ``get_dataloaders`` — convenience factory that splits the dataset and
    returns train / val / test ``DataLoader`` objects.

Adversarial augmentation variants (applied stochastically during training):
  1. Word-swap noise   — swap random adjacent word pairs.
  2. Simulated human edits — drop every 10th word.
  3. Mix text          — concatenate the first half of a human sample and
                         the second half of an AI sample; token labels are
                         assigned accordingly.
"""

import re
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer

from config import get_config
from utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_words(text: str) -> List[str]:
    """Split text on whitespace, preserving punctuation attached to words."""
    return text.split()


def augment_word_swap(text: str, swap_prob: float = 0.15) -> str:
    """
    Adversarial augmentation: randomly swap adjacent word pairs.

    Each consecutive pair (w_i, w_{i+1}) is swapped with probability
    *swap_prob*.  Non-overlapping swaps are performed in a single pass
    to avoid cascading effects.

    Args:
        text:      Input string.
        swap_prob: Per-pair swap probability.

    Returns:
        Augmented string with some adjacent words swapped.
    """
    words = _tokenize_words(text)
    if len(words) < 2:
        return text
    i = 0
    while i < len(words) - 1:
        if random.random() < swap_prob:
            words[i], words[i + 1] = words[i + 1], words[i]
            i += 2  # skip the next pair to avoid overlapping swaps
        else:
            i += 1
    return " ".join(words)


def augment_drop_every_nth(text: str, n: int = 10) -> str:
    """
    Adversarial augmentation: simulate human editing by dropping every *n*-th word.

    This mimics a human condensing or lightly editing an AI-generated passage.

    Args:
        text: Input string.
        n:    Word-drop interval (default 10).

    Returns:
        Augmented string with every *n*-th word removed.
    """
    words = _tokenize_words(text)
    kept = [w for idx, w in enumerate(words) if (idx + 1) % n != 0]
    return " ".join(kept) if kept else text


def augment_mix_text(
    human_text: str,
    ai_text: str,
) -> Tuple[str, List[int]]:
    """
    Adversarial augmentation: splice the first half of a human text with
    the second half of an AI text.

    Word-level labels are 0 for the human portion and 1 for the AI portion.
    These are later mapped to token-level labels during tokenisation.

    Args:
        human_text: A human-written sample.
        ai_text:    An AI-generated sample.

    Returns:
        Tuple of:
          * mixed_text    — concatenated string.
          * word_labels   — List[int] of per-word labels (0 = human, 1 = AI).
    """
    human_words = _tokenize_words(human_text)
    ai_words = _tokenize_words(ai_text)

    half_human = human_words[: max(1, len(human_words) // 2)]
    half_ai = ai_words[len(ai_words) // 2 :]

    mixed_words = half_human + half_ai
    word_labels = [0] * len(half_human) + [1] * len(half_ai)
    return " ".join(mixed_words), word_labels


# ─────────────────────────────────────────────────────────────────────────────
# Core Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HDFNSpanDataset(Dataset):
    """
    Token-level AI-attribution dataset for HDFN-Span.

    Each sample contains:
      • ``deberta_input``  — ``{input_ids, attention_mask}`` for DeBERTa.
      • ``roberta_input``  — ``{input_ids, attention_mask}`` for RoBERTa.
      • ``gpt2_input``     — ``{input_ids, attention_mask}`` for GPT-2.
      • ``token_labels``   — ``LongTensor`` of shape ``(max_length,)`` with
                              per-position binary labels aligned to the DeBERTa
                              tokenisation (the master sequence).
      • ``doc_label``      — Scalar int (1 = AI, 0 = Human) for the whole doc.
      • ``text``           — Raw string (for feature computation at model time).

    Args:
        df:                 Pre-filtered ``pd.DataFrame`` with ``text_content``
                            and ``label`` columns.
        deberta_tokenizer:  Initialised DeBERTa ``AutoTokenizer``.
        roberta_tokenizer:  Initialised RoBERTa ``AutoTokenizer``.
        gpt2_tokenizer:     Initialised GPT-2 ``AutoTokenizer``.
        max_length:         Maximum token sequence length.
        augment:            If ``True``, apply stochastic adversarial augmentation.
    """

    # Augmentation strategies and their relative sampling probabilities.
    _AUG_NONE = "none"
    _AUG_SWAP = "swap"
    _AUG_DROP = "drop"
    _AUG_MIX = "mix"
    _AUG_WEIGHTS = [0.50, 0.20, 0.20, 0.10]  # none / swap / drop / mix

    def __init__(
        self,
        df: pd.DataFrame,
        deberta_tokenizer: AutoTokenizer,
        roberta_tokenizer: AutoTokenizer,
        gpt2_tokenizer: AutoTokenizer,
        max_length: int = 512,
        augment: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.deberta_tokenizer = deberta_tokenizer
        self.roberta_tokenizer = roberta_tokenizer
        self.gpt2_tokenizer = gpt2_tokenizer
        self.max_length = max_length
        self.augment = augment

        # Keep separate indices for human and AI samples (needed for mix aug).
        self._human_idx = self.df.index[self.df["label"] == 0].tolist()
        self._ai_idx = self.df.index[self.df["label"] == 1].tolist()

        logger.info(
            "Dataset created  — total: %d  (AI: %d | Human: %d) | augment=%s",
            len(self.df),
            len(self._ai_idx),
            len(self._human_idx),
            augment,
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _choose_augmentation(self, label: int) -> str:
        """Sample an augmentation strategy name using fixed weights."""
        strategies = [
            self._AUG_NONE,
            self._AUG_SWAP,
            self._AUG_DROP,
            self._AUG_MIX,
        ]
        return random.choices(strategies, weights=self._AUG_WEIGHTS, k=1)[0]

    def _apply_augmentation(
        self,
        text: str,
        label: int,
        strategy: str,
    ) -> Tuple[str, Optional[List[int]]]:
        """
        Apply the selected *strategy* to *text*.

        Returns:
            Tuple of (augmented_text, word_labels_or_None).
            word_labels is only set for the ``mix`` strategy where the label
            assignment is mixed at the word level.
        """
        word_labels: Optional[List[int]] = None

        if strategy == self._AUG_SWAP:
            text = augment_word_swap(text)

        elif strategy == self._AUG_DROP:
            text = augment_drop_every_nth(text)

        elif strategy == self._AUG_MIX:
            # For mix augmentation we need both a human and an AI sample.
            if self._human_idx and self._ai_idx:
                if label == 1:
                    # Current sample is AI; pick a random human sample.
                    partner_idx = random.choice(self._human_idx)
                    human_text = self.df.at[partner_idx, "text_content"]
                    text, word_labels = augment_mix_text(human_text, text)
                else:
                    # Current sample is Human; pick a random AI sample.
                    partner_idx = random.choice(self._ai_idx)
                    ai_text = self.df.at[partner_idx, "text_content"]
                    text, word_labels = augment_mix_text(text, ai_text)
            # If only one class exists, fall back to no augmentation.

        return text, word_labels

    def _tokenize(
        self,
        tokenizer: AutoTokenizer,
        text: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenise *text* and return padded/truncated tensors.

        Returns:
            Dict with ``input_ids`` and ``attention_mask`` tensors of shape
            ``(max_length,)``.
        """
        encoding = tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),          # (L,)
            "attention_mask": encoding["attention_mask"].squeeze(0), # (L,)
        }

    def _build_token_labels_uniform(self, label: int) -> torch.Tensor:
        """
        Build token-level labels for a uniformly-labelled document.

        Every real token position gets the document-level label; padding
        positions receive -100 (ignored during loss computation).

        Args:
            label: 1 for AI, 0 for Human.

        Returns:
            LongTensor of shape ``(max_length,)`` with values in {0, 1, -100}.
        """
        # We use the DeBERTa tokenisation length as the master sequence.
        # The actual non-padding positions are identified via attention_mask.
        token_labels = torch.full((self.max_length,), fill_value=-100, dtype=torch.long)
        # We don't have the encoding here, so we set all positions; the
        # caller will mask padding positions after the fact.
        token_labels[:] = label
        return token_labels

    def _build_token_labels_mixed(
        self,
        encoding: Dict[str, torch.Tensor],
        word_labels: List[int],
        text: str,
        tokenizer: AutoTokenizer,
    ) -> torch.Tensor:
        """
        Build token-level labels for a mixed (spliced) document using
        word-level annotations and the tokeniser's offset mappings.

        Strategy:
          1. Re-tokenise with ``return_offsets_mapping=True`` to align
             character offsets to tokens.
          2. Map each token to its source word via the character offsets.
          3. Assign the word's label to every sub-word token it produces.

        Args:
            encoding:    Already-computed tokenisation (NOT used directly here
                         since we need offset mapping — re-tokenised internally).
            word_labels: Per-word binary labels List[int].
            text:        The mixed text string.
            tokenizer:   Tokeniser to use for re-tokenisation.

        Returns:
            LongTensor of shape ``(max_length,)`` with values in {0, 1, -100}.
        """
        # Re-tokenise with offset_mapping (only works for Fast tokenisers).
        words = text.split()
        token_labels_list = []

        # Compute character-level word boundaries.
        char_pos = 0
        word_char_spans: List[Tuple[int, int]] = []
        for word in words:
            start = char_pos
            end = start + len(word)
            word_char_spans.append((start, end))
            char_pos = end + 1  # +1 for the space

        try:
            enc_with_offsets = tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offset_mapping = enc_with_offsets["offset_mapping"].squeeze(0)  # (L, 2)
            attention_mask = enc_with_offsets["attention_mask"].squeeze(0)

            # For each token position, determine the word label.
            token_labels = torch.full((self.max_length,), fill_value=-100, dtype=torch.long)
            for tok_idx in range(self.max_length):
                if attention_mask[tok_idx] == 0:
                    # Padding
                    break
                tok_start, tok_end = offset_mapping[tok_idx].tolist()
                if tok_start == tok_end == 0:
                    # Special token (CLS / SEP / PAD)
                    token_labels[tok_idx] = -100
                    continue
                # Find which word this token belongs to.
                matched_label = 0  # default to human
                for w_idx, (w_start, w_end) in enumerate(word_char_spans):
                    if tok_start >= w_start and tok_end <= w_end + 1:
                        if w_idx < len(word_labels):
                            matched_label = word_labels[w_idx]
                        break
                token_labels[tok_idx] = matched_label

        except Exception:
            # Fallback: label the first half of tokens as human (0),
            # the second half as AI (1) — approximates 50/50 mix.
            n_real = int(encoding["attention_mask"].sum().item())
            token_labels = torch.full((self.max_length,), fill_value=-100, dtype=torch.long)
            half = n_real // 2
            token_labels[:half] = 0
            token_labels[half:n_real] = 1

        return token_labels

    # ---------------------------------------------------------------------- #
    # Dataset interface                                                        #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text: str = str(row["text_content"])
        label: int = int(row["label"])  # 1 = AI, 0 = Human

        word_labels: Optional[List[int]] = None

        # ------ Adversarial augmentation (training only) ------------------- #
        if self.augment:
            strategy = self._choose_augmentation(label)
            text, word_labels = self._apply_augmentation(text, label, strategy)

        # ------ Tokenise with all three tokenisers ------------------------- #
        deberta_enc = self._tokenize(self.deberta_tokenizer, text)
        roberta_enc = self._tokenize(self.roberta_tokenizer, text)
        gpt2_enc = self._tokenize(self.gpt2_tokenizer, text)

        # ------ Build token-level labels ----------------------------------- #
        if word_labels is not None:
            token_labels = self._build_token_labels_mixed(
                deberta_enc, word_labels, text, self.deberta_tokenizer
            )
        else:
            # Uniform labelling: all real tokens share the document label.
            token_labels = torch.full(
                (self.max_length,), fill_value=label, dtype=torch.long
            )
            # Mask padding positions (attention mask == 0) with -100.
            padding_mask = deberta_enc["attention_mask"] == 0
            token_labels[padding_mask] = -100

        return {
            "deberta_input_ids": deberta_enc["input_ids"],
            "deberta_attention_mask": deberta_enc["attention_mask"],
            "roberta_input_ids": roberta_enc["input_ids"],
            "roberta_attention_mask": roberta_enc["attention_mask"],
            "gpt2_input_ids": gpt2_enc["input_ids"],
            "gpt2_attention_mask": gpt2_enc["attention_mask"],
            "token_labels": token_labels,
            "doc_label": torch.tensor(label, dtype=torch.long),
            # Store the text as a plain Python string; collate_fn will handle it.
        }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizers(cfg) -> Tuple[AutoTokenizer, AutoTokenizer, AutoTokenizer]:
    """
    Instantiate and return the three tokenisers specified in *cfg*.

    The GPT-2 tokeniser is configured with a pad token because GPT-2 has no
    padding token by default.

    Returns:
        Tuple of (deberta_tokenizer, roberta_tokenizer, gpt2_tokenizer).
    """
    logger.info("Loading DeBERTa tokenizer …")
    deberta_tok = AutoTokenizer.from_pretrained(cfg.deberta_model_name)

    logger.info("Loading RoBERTa tokenizer …")
    roberta_tok = AutoTokenizer.from_pretrained(cfg.roberta_model_name)

    logger.info("Loading GPT-2 tokenizer …")
    gpt2_tok = AutoTokenizer.from_pretrained(cfg.gpt2_model_name)
    # GPT-2 has no native padding token; use the EOS token as padding.
    gpt2_tok.pad_token = gpt2_tok.eos_token

    return deberta_tok, roberta_tok, gpt2_tok


def load_dataframe(data_path: str) -> pd.DataFrame:
    """
    Load and validate the CSV dataset.

    Expected columns: ``text_content``, ``label``.
    Label convention: 1 = AI-generated, 0 = Human-written.

    Args:
        data_path: Path to the CSV file.

    Returns:
        Cleaned ``pd.DataFrame``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If required columns are missing.
    """
    import os
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            "Please place 'ai_human_content_detection_dataset.csv' in the "
            "project root directory."
        )

    df = pd.read_csv(data_path)
    required = {"text_content", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"The CSV is missing required columns: {missing}.\n"
            f"Found columns: {list(df.columns)}"
        )

    # Normalise: ensure labels are 0 or 1.
    df["label"] = df["label"].astype(int)
    valid_labels = df["label"].isin([0, 1])
    if not valid_labels.all():
        logger.warning(
            "Dropping %d rows with labels outside {0, 1}.",
            (~valid_labels).sum(),
        )
        df = df[valid_labels]

    # Drop rows with empty text.
    df = df.dropna(subset=["text_content"])
    df = df[df["text_content"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)

    logger.info(
        "Loaded dataset: %d samples  (AI=%d, Human=%d)",
        len(df),
        (df["label"] == 1).sum(),
        (df["label"] == 0).sum(),
    )
    return df


def get_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple]:
    """
    Build train / val / test ``DataLoader`` objects.

    Args:
        cfg: Project :class:`~config.Config` instance.

    Returns:
        Tuple of:
          * train_loader
          * val_loader
          * test_loader
          * tokenizers tuple — (deberta_tok, roberta_tok, gpt2_tok)
    """
    df = load_dataframe(cfg.data_path)
    deberta_tok, roberta_tok, gpt2_tok = load_tokenizers(cfg)

    # Split the dataframe deterministically.
    n = len(df)
    n_train = int(n * cfg.train_split)
    n_val = int(n * cfg.val_split)
    n_test = n - n_train - n_val

    # Shuffle with fixed seed for reproducibility.
    df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train: n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    logger.info("Split sizes — train: %d | val: %d | test: %d", n_train, n_val, n_test)

    tok_args = (deberta_tok, roberta_tok, gpt2_tok)
    common_kw = dict(max_length=cfg.max_length)

    train_ds = HDFNSpanDataset(train_df, *tok_args, augment=True, **common_kw)
    val_ds = HDFNSpanDataset(val_df, *tok_args, augment=False, **common_kw)
    test_ds = HDFNSpanDataset(test_df, *tok_args, augment=False, **common_kw)

    loader_kw = dict(
        batch_size=cfg.batch_size,
        num_workers=0,  # set > 0 on Linux for faster loading
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, tok_args
