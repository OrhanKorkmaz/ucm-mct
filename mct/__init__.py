"""Unified Cognitive Mesh v2 — Mikro Bilişsel Tercüman (MCT) paketi.

Modüller:
  concepts  — çekirdek konsept seti (EN/TR, frekans sıralı)
  extract   — LLM gizli durum çıkarımı (mean-pooled, katman bantları)
  cka       — Centered Kernel Alignment katman taraması
  models    — MCT mimarisi: yarı-ortogonal çekirdek + kapılı artık MLP
  losses    — simetrik InfoNCE + MSE çapası
  train     — eğitim döngüleri (ikili MCT ve nötr-uzay ortak eğitim)
  metrics   — retrieval, kosinüs, geometri korunumu, k-NN, analoji
  sae       — Top-K Sparse Autoencoder köprüsü
  patching  — activation patching ile işlevsellik testi (H3)
"""

DEVICE = None


def get_device():
    global DEVICE
    if DEVICE is None:
        import os
        import torch
        forced = os.environ.get("MCT_DEVICE")
        if forced:
            DEVICE = torch.device(forced)
        elif torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
        else:
            DEVICE = torch.device("cpu")
    return DEVICE
