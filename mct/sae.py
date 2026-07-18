"""Top-K Sparse Autoencoder köprüsü (Anthropic tarzı, Gao vd. 2024 Top-K).

Transferde yalnızca aktif top-k özelliğin (indeks, değer) çiftleri taşınır.
Sıkıştırma oranı = d_model * 16bit / (k * (16bit değer + 16bit indeks))
               ≈ d / (2k)  (fp16 + uint16 indeks varsayımıyla).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TopKSAE(nn.Module):
    def __init__(self, d, n_features, k):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d, n_features)
        self.dec = nn.Linear(n_features, d)
        with torch.no_grad():  # tied-transpose başlatma
            self.dec.weight.copy_(self.enc.weight.T)

    def encode(self, x):
        f = torch.relu(self.enc(x))
        if self.k < f.shape[-1]:
            topv, topi = f.topk(self.k, dim=-1)
            f = torch.zeros_like(f).scatter_(-1, topi, topv)
        return f

    def forward(self, x):
        return self.dec(self.encode(x))


def train_sae(sae, X, device, epochs=40, batch_size=256, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    sae.to(device).train()
    mu = torch.as_tensor(X.mean(0)).to(device)
    sd = torch.as_tensor(X.std() + 1e-6).to(device)
    dl = DataLoader(TensorDataset(torch.as_tensor(X)), batch_size=batch_size,
                    shuffle=True)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    for ep in range(epochs):
        for (x,) in dl:
            x = (x.to(device) - mu) / sd
            loss = ((sae(x) - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    sae.eval()
    sae.mu, sae.sd = mu, sd
    return sae


@torch.no_grad()
def sae_roundtrip(sae, X, device, batch_size=1024):
    outs, active = [], []
    for i in range(0, len(X), batch_size):
        x = (torch.as_tensor(X[i:i + batch_size]).to(device) - sae.mu) / sae.sd
        f = sae.encode(x)
        active.append((f > 0).float().sum(-1).mean().item())
        outs.append((sae.dec(f) * sae.sd + sae.mu).cpu().numpy())
    return np.concatenate(outs), float(np.mean(active))
