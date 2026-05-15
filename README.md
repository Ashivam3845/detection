---
title: HDFN-SPAN AI Content Detector
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: mit
---

# HDFN-SPAN: Token-Level AI Attribution

A **Hierarchical Drift Fusion Network** that detects AI-generated text at the **token level** — not just a binary document score, but a full per-token heatmap showing exactly which words were written by an AI vs. a human.

## Architecture
- **DeBERTa-v3-base** + **RoBERTa-base** + **GPT-2** tri-encoder fusion
- **SSS** (Semantic Smoothness Score) + **SDS** (Stylometric Drift Score) handcrafted features
- 8-head deep fusion self-attention
- Span-level IoU evaluation

## How to Use
Paste any text into the box and click **Evaluate Content** to see:
- Per-token AI probability heatmap (red = AI, green = Human)
- Overall AI confidence gauge
- Stylometric and semantic drift scores
