"""
model.py — HDFNSpanModel: Hybrid Deep Fusion Network for Token-Level AI Attribution
====================================================================================
Architecture overview
─────────────────────
                         ┌──────────────────────────────────────────────────┐
  Input text ──────────► │ Primary Encoder   : DeBERTa-v3-base (768 d)      │
                         │ Secondary Encoder : RoBERTa-base   (768 d)        │
                         │ Tertiary Encoder  : GPT-2          (768 d)        │
                         │ Feature Module    : SSS + SDS  →  64 d            │
                         └──────────────────────────────────────────────────┘
                                          │  Concatenate [768·3 + 64 = 2368 d]
                                          ▼
                         ┌──────────────────────────────────────────────────┐
                         │ Deep Fusion: Linear → GELU → Dropout → 512 d     │
                         │ MultiheadAttention (8 heads, 512 d)               │
                         └──────────────────────────────────────────────────┘
                                          │  (B, L, 512)
                                          ▼
                         ┌──────────────────────────────────────────────────┐
                         │ Token-level Classification Head: Linear(512 → 1) │
                         │ Sigmoid → per-token AI probability               │
                         └──────────────────────────────────────────────────┘

Ablation flags (from config):
  • use_features : disable SSS / SDS module → concatenation dim = 768·3
  • use_attention: skip MultiheadAttention → fused projection output used directly
  • use_gpt2     : skip GPT-2 encoder      → concatenation dim reduced by 768

The model is designed so that disabling components does NOT change the
classification head interface — projection layers are re-sized accordingly.
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoConfig,
    DebertaV2Model,
    RobertaModel,
    GPT2Model,
)

from config import Config, get_config
from utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Module: Semantic Smoothness & Stylometric Drift
# ─────────────────────────────────────────────────────────────────────────────

class FeatureModule(nn.Module):
    """
    Computes two complementary hand-crafted features and projects them into
    a learnable embedding that can be concatenated with contextual embeddings.

    Features computed
    -----------------
    SSS — Semantic Smoothness Score
        Uses a sentence-transformer to embed each sentence.  The pairwise
        cosine similarities of consecutive sentence embeddings are computed,
        and the *variance* of those similarities serves as the SSS.  High
        variance suggests abrupt semantic shifts — a hallmark of mixed
        AI/human authorship.

    SDS — Stylometric Drift Score
        Captures writing-style inconsistency via the variance of:
          • sentence-level character lengths (sentence-length variance), and
          • word-level character lengths (word-length variance).
        Both sub-scores are averaged to form the SDS.

    Both scalars are normalised, concatenated into a 2-d vector, and
    projected through a learned linear layer to ``feature_dim`` dimensions.
    The result is then broadcast across the token sequence length so it can
    be concatenated with per-token encoder outputs.

    Args:
        feature_dim:      Output dimensionality of the linear projection.
        sbert_model_name: HuggingFace / SentenceTransformers model identifier.
    """

    def __init__(self, feature_dim: int = 64, sbert_model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.feature_dim = feature_dim

        # Lazy-import sentence-transformers to keep the dependency optional at
        # import time (heavy library).
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(sbert_model_name)
            # Freeze SBERT weights — we only use it for inference.
            for param in self._sbert.parameters():
                param.requires_grad = False
            self._sbert_available = True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed.  SSS will default to 0.0."
            )
            self._sbert_available = False

        # Project [SSS, SDS] → feature_dim learnable embedding.
        self.projection = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def _split_sentences(self, text: str) -> list:
        """Naive sentence splitter on '.', '!', '?'."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s.strip()]

    def _compute_sss(self, text: str) -> float:
        """
        Semantic Smoothness Score (SSS).

        Returns the variance of pairwise cosine similarities between
        consecutive sentence embeddings.  Returns 0 if fewer than 2
        sentences are detected or SBERT is unavailable.
        """
        if not self._sbert_available:
            return 0.0

        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.0

        # Encode all sentences in a single batch for efficiency.
        with torch.no_grad():
            embs = self._sbert.encode(
                sentences, convert_to_tensor=True, show_progress_bar=False
            )  # (n_sentences, emb_dim)

        # Cosine similarity between consecutive pairs.
        # Add a tiny epsilon and check for zero norms to be safe.
        sims = F.cosine_similarity(embs[:-1], embs[1:] + 1e-8, dim=-1)  # (n_sentences - 1,)
        
        if sims.numel() < 1:
            return 0.0
        if sims.numel() == 1:
            # Only one pair similarity; variance is zero by definition.
            return 0.0
        return float(sims.var().item())

    def _compute_sds(self, text: str) -> float:
        """
        Stylometric Drift Score (SDS).

        Averages:
          1. Variance of sentence character lengths.
          2. Variance of word character lengths.
        """
        import re
        sentences = self._split_sentences(text)
        words = text.split()

        if not sentences or not words:
            return 0.0

        sent_lengths = [len(s) for s in sentences]
        word_lengths = [len(w) for w in words]

        sent_var = float(torch.tensor(sent_lengths, dtype=torch.float).var().item()) \
            if len(sent_lengths) > 1 else 0.0
        word_var = float(torch.tensor(word_lengths, dtype=torch.float).var().item()) \
            if len(word_lengths) > 1 else 0.0

        # Replace potential NaN from variance with zeros.
        if np.isnan(sent_var): sent_var = 0.0
        if np.isnan(word_var): word_var = 0.0

        # Clamp and normalise to a [0, 1] scale (approximate).
        sent_var_norm = min(sent_var / 1000.0, 1.0)
        word_var_norm = min(word_var / 10.0, 1.0)

        return (sent_var_norm + word_var_norm) / 2.0

    def forward(
        self,
        texts: list,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute feature embeddings and broadcast to the token sequence length.

        Args:
            texts:   List of raw strings, one per sample in the batch.
            seq_len: Target sequence length L to broadcast over.
            device:  Target device for output tensor.

        Returns:
            FloatTensor of shape ``(B, L, feature_dim)``.
        """
        batch_features = []
        for text in texts:
            sss = self._compute_sss(text)
            sds = self._compute_sds(text)
            batch_features.append([sss, sds])

        feat_vec = torch.tensor(batch_features, dtype=torch.float32, device=device)  # (B, 2)
        feat_emb = self.projection(feat_vec)                                          # (B, feature_dim)
        feat_emb = feat_emb.unsqueeze(1).expand(-1, seq_len, -1)                     # (B, L, feature_dim)
        return feat_emb


# ─────────────────────────────────────────────────────────────────────────────
# Main HDFN-Span Model
# ─────────────────────────────────────────────────────────────────────────────

class HDFNSpanModel(nn.Module):
    """
    Hybrid Deep Fusion Network for Token-Level AI Attribution (HDFN-Span).

    Three transformer encoders fuse their hidden states with optional
    hand-crafted features, a deep fusion projection, and a multihead
    self-attention layer.  A token-level linear classifier then emits
    per-token AI probabilities.

    Args:
        cfg: Project :class:`~config.Config` instance.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        # ------------------------------------------------------------------ #
        # 1. Primary Encoder — DeBERTa-v3-base                               #
        # ------------------------------------------------------------------ #
        logger.info("Loading DeBERTa-v3-base encoder …")
        self.deberta = AutoModel.from_pretrained(cfg.deberta_model_name)
        self.deberta_norm = nn.LayerNorm(cfg.deberta_hidden)

        # ------------------------------------------------------------------ #
        # 2. Secondary Encoder — RoBERTa-base                                #
        # ------------------------------------------------------------------ #
        logger.info("Loading RoBERTa-base encoder …")
        self.roberta = AutoModel.from_pretrained(cfg.roberta_model_name)
        self.roberta_norm = nn.LayerNorm(cfg.roberta_hidden)

        # ------------------------------------------------------------------ #
        # 3. Tertiary Encoder — GPT-2 (conditional on ablation flag)         #
        # ------------------------------------------------------------------ #
        if cfg.use_gpt2:
            logger.info("Loading GPT-2 encoder …")
            self.gpt2 = AutoModel.from_pretrained(cfg.gpt2_model_name)
            self.gpt2_norm = nn.LayerNorm(cfg.gpt2_hidden)
        else:
            self.gpt2 = None
            self.gpt2_norm = None
            logger.info("GPT-2 encoder disabled (use_gpt2=False).")

        # ------------------------------------------------------------------ #
        # 4. Feature Module — SSS + SDS                                      #
        # ------------------------------------------------------------------ #
        if cfg.use_features:
            self.feature_module = FeatureModule(
                feature_dim=cfg.feature_hidden,
                sbert_model_name=cfg.sbert_model_name,
            )
            feature_contrib = cfg.feature_hidden
        else:
            self.feature_module = None
            feature_contrib = 0
            logger.info("Feature module disabled (use_features=False).")

        # ------------------------------------------------------------------ #
        # 5. Deep Fusion Layer                                                #
        # ------------------------------------------------------------------ #
        # Compute concatenated dimension.
        concat_dim = (
            cfg.deberta_hidden
            + cfg.roberta_hidden
            + (cfg.gpt2_hidden if cfg.use_gpt2 else 0)
            + feature_contrib
        )

        # Projection: concat_dim → fusion_hidden.
        self.concatenation_norm = nn.LayerNorm(concat_dim)
        self.fusion_projection = nn.Sequential(
            nn.Linear(concat_dim, cfg.fusion_hidden),
            nn.GELU(),
            nn.Dropout(p=cfg.dropout),
        )

        # Multihead self-attention for dynamic encoder weighting.
        if cfg.use_attention:
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=cfg.fusion_hidden,
                num_heads=cfg.num_attention_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(cfg.fusion_hidden)
            self.post_attention_norm = nn.LayerNorm(cfg.fusion_hidden)
        else:
            self.fusion_attention = None
            self.attention_norm = None
            self.post_attention_norm = None
            logger.info("Fusion attention disabled (use_attention=False).")

        # ------------------------------------------------------------------ #
        # 6. Token-level Classification Head                                  #
        # ------------------------------------------------------------------ #
        self.classifier = nn.Linear(cfg.fusion_hidden, 1)  # → per-token logit

        # ------------------------------------------------------------------ #
        # 7. Weight Initialization for New Layers                             #
        # ------------------------------------------------------------------ #
        self._init_weights()

        logger.info(
            "HDFNSpanModel built  |  concat_dim=%d  fusion_hidden=%d",
            concat_dim,
            cfg.fusion_hidden,
        )

    def _init_weights(self) -> None:
        """
        Custom initialization for newly-added fusion and classification layers.
        Uses Xavier (Glorot) initialization to maintain stable variance.
        """
        new_modules = [
            self.fusion_projection,
            self.fusion_attention,
            self.classifier,
        ]
        if self.feature_module:
            new_modules.append(self.feature_module.projection)

        for module in new_modules:
            if module is None:
                continue
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.MultiheadAttention):
                    # MultiheadAttention's internal weight parameters.
                    nn.init.xavier_uniform_(m.in_proj_weight)
                    if m.in_proj_bias is not None:
                        nn.init.zeros_(m.in_proj_bias)
                    nn.init.xavier_uniform_(m.out_proj.weight)
                    if m.out_proj.bias is not None:
                        nn.init.zeros_(m.out_proj.bias)

    # ---------------------------------------------------------------------- #
    # Encoder helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _encode_deberta(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run DeBERTa forward pass.  
        NOTE: DeBERTa-v3 is notoriously unstable in mixed precision (NaNs).  
        We force it to run in float32 for stability.
        """
        device_type = "cuda" if input_ids.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            outputs = self.deberta(
                input_ids=input_ids.long(),
                attention_mask=attention_mask.float(),
            )
            return outputs.last_hidden_state.float()  # (B, L, 768)

    def _encode_roberta(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run RoBERTa forward pass, returning last hidden state (B, L, H)."""
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # (B, L, 768)

    def _encode_gpt2(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run GPT-2 forward pass, returning last hidden state (B, L, H)."""
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # (B, L, 768)

    # ---------------------------------------------------------------------- #
    # Forward pass                                                             #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        deberta_input_ids: torch.Tensor,
        deberta_attention_mask: torch.Tensor,
        roberta_input_ids: torch.Tensor,
        roberta_attention_mask: torch.Tensor,
        gpt2_input_ids: torch.Tensor,
        gpt2_attention_mask: torch.Tensor,
        texts: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass of HDFN-Span.

        Args:
            deberta_input_ids:       (B, L) token IDs for DeBERTa.
            deberta_attention_mask:  (B, L) attention mask for DeBERTa.
            roberta_input_ids:       (B, L) token IDs for RoBERTa.
            roberta_attention_mask:  (B, L) attention mask for RoBERTa.
            gpt2_input_ids:          (B, L) token IDs for GPT-2.
            gpt2_attention_mask:     (B, L) attention mask for GPT-2.
            texts:                   List of raw strings (length B) needed by
                                     the feature module.  If ``None`` and
                                     ``use_features=True``, features default
                                     to zero.

        Returns:
            Tuple of:
              - logits: FloatTensor ``(B, L)`` — raw pre-sigmoid scores.
              - probs:  FloatTensor ``(B, L)`` — per-token AI probabilities in [0, 1].
        """
        B, L = deberta_input_ids.shape
        device = deberta_input_ids.device

        # ── 1. Encode with each transformer ─────────────────────────────── #
        h_deberta = self._encode_deberta(deberta_input_ids, deberta_attention_mask)  # (B, L, 768)
        h_deberta = self.deberta_norm(h_deberta)

        h_roberta = self._encode_roberta(roberta_input_ids, roberta_attention_mask)  # (B, L, 768)
        h_roberta = self.roberta_norm(h_roberta)

        parts = [h_deberta, h_roberta]

        if self.gpt2 is not None:
            h_gpt2 = self._encode_gpt2(gpt2_input_ids, gpt2_attention_mask)  # (B, L, 768)
            h_gpt2 = self.gpt2_norm(h_gpt2)
            parts.append(h_gpt2)

        # ── 2. Feature module ────────────────────────────────────────────── #
        if self.feature_module is not None:
            if texts is None:
                # Fallback: empty features.
                feat_emb = torch.zeros(B, L, self.cfg.feature_hidden, device=device)
            else:
                feat_emb = self.feature_module(texts, seq_len=L, device=device)  # (B, L, 64)
            parts.append(feat_emb)

        # ── 3. Concatenate all representations ──────────────────────────── #
        fused = torch.cat(parts, dim=-1)  # (B, L, concat_dim)
        fused = self.concatenation_norm(fused)

        # ── 4. Fusion projection ─────────────────────────────────────────── #
        projected = self.fusion_projection(fused)  # (B, L, fusion_hidden)

        # ── 5. Multihead self-attention ──────────────────────────────────── #
        if self.fusion_attention is not None:
            # key_padding_mask: True indicates positions to IGNORE (i.e. padding).
            # DeBERTa attention_mask: 1 = real token, 0 = padding.
            key_padding_mask = deberta_attention_mask == 0  # (B, L) bool

            # Normalise before attention to stabilize scores.
            normed_projected = self.attention_norm(projected)
            attn_output, _ = self.fusion_attention(
                query=normed_projected,
                key=normed_projected,
                value=normed_projected,
                key_padding_mask=key_padding_mask,
            )  # (B, L, fusion_hidden)
            # Residual connection + post-attention normalization.
            final_hidden = self.post_attention_norm(projected + attn_output)  # (B, L, 512)
        else:
            final_hidden = projected  # (B, L, 512)

        # ── 6. Token-level classification ───────────────────────────────── #
        logits = self.classifier(final_hidden).squeeze(-1)  # (B, L)
        probs = torch.sigmoid(logits)                        # (B, L)  ∈ [0, 1]

        return logits, probs  # (B, L) each


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: Optional[Config] = None) -> HDFNSpanModel:
    """
    Construct an :class:`HDFNSpanModel` from a :class:`~config.Config` object.

    Args:
        cfg: Configuration.  Uses the global singleton if ``None``.

    Returns:
        Initialised :class:`HDFNSpanModel` (on CPU; move to device externally).
    """
    if cfg is None:
        cfg = get_config()
    model = HDFNSpanModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total trainable parameters: %s", f"{n_params:,}")
    return model
