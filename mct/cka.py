"""Centered Kernel Alignment (lineer CKA) katman taraması.

Katman çifti seçimi sezgiyle değil ölçümle: A'nın her aday katmanı ile
B'nin her aday katmanı arasında CKA hesaplanır; Katman-2 bandı
(derinliğin %30-50'si) ve Katman-3 bandı (%60-75) içinde CKA'yı
maksimize eden çiftler seçilir.
"""
import numpy as np


def linear_cka(X, Y):
    """X: [N, d1], Y: [N, d2] — lineer CKA (Kornblith vd. 2019)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = Y.T @ X
    num = (xty ** 2).sum()
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den) if den > 0 else 0.0


def band(n_layers, lo, hi):
    """Derinlik oranı [lo, hi] aralığındaki katman indeksleri (1..n)."""
    return [l for l in range(1, n_layers + 1) if lo <= l / n_layers <= hi]


def cka_matrix(acts_a, acts_b, layers_a, layers_b):
    M = np.zeros((len(layers_a), len(layers_b)))
    for i, la in enumerate(layers_a):
        for j, lb in enumerate(layers_b):
            M[i, j] = linear_cka(acts_a[la], acts_b[lb])
    return M


def pick_pair(acts_a, acts_b, n_layers_a, n_layers_b, lo, hi):
    la_band = band(n_layers_a, lo, hi)
    lb_band = band(n_layers_b, lo, hi)
    M = cka_matrix(acts_a, acts_b, la_band, lb_band)
    i, j = np.unravel_index(M.argmax(), M.shape)
    return {"layer_a": la_band[i], "layer_b": lb_band[j],
            "cka": float(M[i, j]), "matrix": M.tolist(),
            "layers_a": la_band, "layers_b": lb_band}
