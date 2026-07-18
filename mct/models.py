"""MCT mimarisi.

GatedResidualMCT:  y = mu_out + s * W_orth (x - mu_in) + g * MLP(x - mu_in)
  - W_orth: yarı-ortogonal çekirdek. Procrustes çözümüyle TAM başlatılır;
    eğitim boyunca yumuşak ortogonallik cezasıyla (‖WᵀW − I‖²) yarı-
    ortogonal tutulur. (Not: torch'un sert `orthogonal` parametrizasyonu
    denendi; householder `right_inverse` başlatması ilk optimizasyon
    adımında sayısal olarak çöküyor ve MPS'te `householder_product`
    desteklenmiyor — PoC raporundaki "dikgenliği yumuşak ceza olarak
    uygula" önerisi bu iki nedenle benimsendi.)
  - s: öğrenilebilir ölçek (uzaylar izometrik değilse gerekli)
  - g: sıfırdan başlayan öğrenilebilir kapı — model hiçbir zaman
    Procrustes taban çizgisinden kötü başlamaz
"""
import numpy as np
import torch
import torch.nn as nn


def procrustes_fit(A, B):
    """Yarı-ortogonal Procrustes: min ||s·A W^T - B||_F,  W^T W = I.

    A: [N, d_a], B: [N, d_b] (merkezlenmiş). Dönüş: W [d_b, d_a], s.
    """
    M = B.T @ A                      # [d_b, d_a]
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt                       # [d_b, d_a], satırları/sütunları ortonormal
    s = S.sum() / (A ** 2).sum()
    return W.astype(np.float32), float(s)


class GatedResidualMCT(nn.Module):
    def __init__(self, d_in, d_out, hidden=1024, mu_in=None, mu_out=None,
                 W0=None, s0=1.0):
        super().__init__()
        self.core = nn.Linear(d_in, d_out, bias=False)
        if W0 is not None:
            with torch.no_grad():
                self.core.weight.copy_(torch.as_tensor(W0))
        self.log_s = nn.Parameter(torch.tensor(float(np.log(max(s0, 1e-6)))))
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("mu_in", torch.zeros(d_in) if mu_in is None
                             else torch.as_tensor(mu_in, dtype=torch.float32))
        self.register_buffer("mu_out", torch.zeros(d_out) if mu_out is None
                             else torch.as_tensor(mu_out, dtype=torch.float32))

    def forward(self, x):
        xc = x - self.mu_in
        return (self.mu_out + torch.exp(self.log_s) * self.core(xc)
                + self.gate * self.mlp(xc))

    def orth_penalty(self):
        """‖WᵀW − I‖²_F / d  (dikey matris için WᵀW, yatay için WWᵀ)."""
        W = self.core.weight
        if W.shape[0] >= W.shape[1]:
            G = W.T @ W
        else:
            G = W @ W.T
        d = G.shape[0]
        I = torch.eye(d, device=W.device, dtype=W.dtype)
        return ((G - I) ** 2).sum() / d

    def orth_error(self):
        with torch.no_grad():
            return float(self.orth_penalty().sqrt())


def make_mct(A_train, B_train, hidden=1024):
    """Eğitim verisinden merkezleme + Procrustes başlatmalı MCT üretir."""
    mu_a, mu_b = A_train.mean(0), B_train.mean(0)
    W0, s0 = procrustes_fit(A_train - mu_a, B_train - mu_b)
    return GatedResidualMCT(A_train.shape[1], B_train.shape[1], hidden,
                            mu_in=mu_a, mu_out=mu_b, W0=W0, s0=s0)


class RidgeBaseline:
    """Kısıtsız lineer taban çizgisi (kapalı form)."""

    def __init__(self, A, B, lam=1.0):
        d = A.shape[1]
        self.mu_a, self.mu_b = A.mean(0), B.mean(0)
        Ac, Bc = A - self.mu_a, B - self.mu_b
        self.W = np.linalg.solve(Ac.T @ Ac + lam * np.eye(d), Ac.T @ Bc)

    def __call__(self, A):
        return (A - self.mu_a) @ self.W + self.mu_b


class ProcrustesBaseline:
    def __init__(self, A, B):
        self.mu_a, self.mu_b = A.mean(0), B.mean(0)
        self.W, self.s = procrustes_fit(A - self.mu_a, B - self.mu_b)

    def __call__(self, A):
        return self.s * (A - self.mu_a) @ self.W.T + self.mu_b
