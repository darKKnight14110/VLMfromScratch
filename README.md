# PaliGemma Vision-Language Model (VLM) from Scratch

Welcome to this deep dive implementation of **PaliGemma**, a powerful Vision-Language Model (VLM), coded completely from scratch using PyTorch. This project is inspired by the educational walkthrough by Umar Jamil.

---

## 🚀 What is PaliGemma?

**PaliGemma** is a state-of-the-art multimodal model developed to process both **images and text** seamlessly. It builds on top of concepts from Transformers, Vision Transformers (ViT), CLIP-style contrastive learning, and advanced token sampling methods.

---

## 🧰 Features Covered

We implement the inner mechanics of modern models:

### 🔤 Transformer Fundamentals
- Token Embeddings and Positional Encodings
- Multi-Head Self Attention (MHSA)
- Feedforward Networks and GELU activation
- Attention Masks: Causal & Non-Causal
- Logits, Softmax, and Numerical Stability
- Rotary Positional Embedding (RoPE)
- Grouped Query Attention (GQA)
- Weight Tying between embedding & output
- KV-Cache: Efficient decoding via prefilling

### 🖼 Vision Side
- Vision Transformer (ViT) backbone
- Patch Embeddings and Class Tokens
- Image Token Insertion (as in PaliGemma)

### 🔁 Training Concepts
- Contrastive Learning: CLIP & SigLIP-style loss
- Normalization layers: BatchNorm, LayerNorm, RMSNorm
- Cross Entropy Loss – Numerical Stability
- Top-P (Nucleus) Sampling and Temperature Scaling

---
