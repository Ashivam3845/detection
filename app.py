"""
app.py — HDFN-Span Interactive Web Interface
===============================================
A premium Streamlit application for real-time AI content detection
and token-level attribution using the HDFN-Span architecture.

Features:
- Triple-Encoder Fusion (DeBERTa, RoBERTa, GPT-2).
- Stylometric Feature Heatmap (SSS/SDS).
- Token-Level Highlighting.
- Model Confidence Gauges.
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from transformers import AutoTokenizer
import os

from config import Config, get_config
from model import HDFNSpanModel, build_model
from utils import get_device

# ─────────────────────────────────────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HDFN-Span AI Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a professional, premium aesthetic.
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stTextArea textarea {
        background-color: #1A1C24;
        color: #E0E0E0;
        border: 1px solid #30363D;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }
    .stButton button {
        background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        border-radius: 6px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(0,0,0,0.39);
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #357ABD 0%, #4A90E2 100%);
        border: none;
        color: white;
    }
    .metric-card {
        background-color: #1A1C24;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363D;
        text-align: center;
    }
    .highlight-ai {
        background-color: rgba(255, 75, 75, 0.3);
        border-bottom: 2px solid #FF4B4B;
        padding: 2px 0;
    }
    .highlight-human {
        background-color: rgba(75, 255, 75, 0.1);
        padding: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_resources():
    """Load model, state_dict, and all tokenizers."""
    cfg = get_config()
    device = get_device()
    
    # 1. Tokenizers
    dt = AutoTokenizer.from_pretrained(cfg.deberta_model_name)
    rt = AutoTokenizer.from_pretrained(cfg.roberta_model_name)
    gt = AutoTokenizer.from_pretrained(cfg.gpt2_model_name)
    # GPT-2 needs a pad token.
    if gt.pad_token is None:
        gt.pad_token = gt.eos_token
        
    # 2. Model
    model = build_model(cfg).to(device)
    if os.path.exists(cfg.model_save_path):
        model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
        model.eval()
    
    return cfg, model, (dt, rt, gt), device

# ─────────────────────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────────────────────

def get_predictions(text, cfg, model, tokenizers, device):
    """Run triple-encoder inference on one text."""
    dt, rt, gt = tokenizers
    
    # Pre-tokenize (simplified for single sample)
    d_enc = dt(text, max_length=cfg.max_length, padding="max_length", truncation=True, return_tensors="pt")
    r_enc = rt(text, max_length=cfg.max_length, padding="max_length", truncation=True, return_tensors="pt")
    g_enc = gt(text, max_length=cfg.max_length, padding="max_length", truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        logits, probs = model(
            deberta_input_ids=d_enc["input_ids"].to(device),
            deberta_attention_mask=d_enc["attention_mask"].to(device),
            roberta_input_ids=r_enc["input_ids"].to(device),
            roberta_attention_mask=r_enc["attention_mask"].to(device),
            gpt2_input_ids=g_enc["input_ids"].to(device),
            gpt2_attention_mask=g_enc["attention_mask"].to(device),
            texts=[text]
        )
    
    # Convert to numpy
    probs_np = probs[0].cpu().numpy()
    mask_np = d_enc["attention_mask"][0].cpu().numpy()
    
    # Filter only real tokens (ignore padding)
    real_probs = probs_np[mask_np == 1]
    
    # Map back to DeBERTa tokens for visualisation
    tokens = dt.convert_ids_to_tokens(d_enc["input_ids"][0][mask_np == 1])
    
    # Stylometric scores (for visual only - approximate)
    sss_val = 0.0
    sds_val = 0.0
    if model.feature_module:
        sss_val = model.feature_module._compute_sss(text)
        sds_val = model.feature_module._compute_sds(text)
        
    return tokens, real_probs, sss_val, sds_val

# ─────────────────────────────────────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🤖 HDFN-Span AI Content Attribution")
    st.markdown("---")
    
    cfg, model, tokenizers, device = load_resources()
    
    col_input, col_stats = st.columns([2, 1], gap="large")
    
    with col_input:
        st.subheader("📝 Input Content")
        input_text = st.text_area(
            "Paste document or sentences to analyze:",
            placeholder="e.g., The implications of machine learning in modern medicine are profound...",
            height=350
        )
        
        btn_eval = st.button("🚀 Evaluate Content")
        
    with col_stats:
        st.subheader("📊 Attribution Stats")
        if not btn_eval:
            st.info("Input text and click 'Evaluate' to see local attribution.")
            
    if btn_eval and input_text.strip():
        with st.spinner("Analyzing deep fusion representations..."):
            tokens, probs, sss, sds = get_predictions(input_text, cfg, model, tokenizers, device)
            
            # ── Overall Score ───────────────────────────────────────────────── #
            overall_ai_score = float(np.mean(probs))
            
            with col_stats:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = overall_ai_score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "AI Confidence Score", 'font': {'size': 18, 'color': '#E0E0E0'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#E0E0E0"},
                        'bar': {'color': "#FF4B4B" if overall_ai_score > 0.5 else "#4A90E2"},
                        'bgcolor': "#1A1C24",
                        'borderwidth': 2,
                        'bordercolor': "#30363D",
                        'steps': [
                            {'range': [0, 50], 'color': '#111317'},
                            {'range': [50, 100], 'color': '#1E1012'}
                        ],
                    }
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "#E0E0E0"})
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Cards
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='color: #888; font-size: 0.9rem;'>Stylometric Drift Score (SDS)</div>
                    <div style='color: #4A90E2; font-size: 1.8rem; font-weight: 700;'>{sds:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='color: #888; font-size: 0.9rem;'>Semantic Stability Score (SSS)</div>
                    <div style='color: #4A90E2; font-size: 1.8rem; font-weight: 700;'>{sss:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Token Highlighting ──────────────────────────────────────────── #
            st.divider()
            st.subheader("🔍 Token-Level Attribution Heatmap")
            
            html_bits = []
            for t, p in zip(tokens, probs):
                # Clean up DeBERTa tokens
                clean_t = t.replace(" ", " ").replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "")
                if not clean_t: continue
                
                color_class = "highlight-ai" if p > 0.5 else "highlight-human"
                html_bits.append(f"<span class='{color_class}' title='p={p:.3f}'>{clean_t}</span>")
            
            st.markdown(
                f"<div style='line-height: 2.2; font-size: 1.1rem; color: #BBB;'>"
                f"{''.join(html_bits)}"
                f"</div>", 
                unsafe_allow_html=True
            )
            
            st.caption("Hover over highlighted tokens to see exact AI probability scores.")

if __name__ == "__main__":
    main()
