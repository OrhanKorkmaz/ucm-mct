"""Eğitim döngüleri: (1) ikili MCT, (2) nötr ortak uzay ortak eğitimi."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import mct_loss, symmetric_infonce, cosine_anchor


def train_mct(mct, A_train, B_train, device, epochs=60, batch_size=256,
              lr=1e-3, tau=0.07, lam_anchor=0.25, lam_orth=1.0,
              use_infonce=True, seed=42):
    torch.manual_seed(seed)
    mct.to(device).train()
    ds = TensorDataset(torch.as_tensor(A_train), torch.as_tensor(B_train))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=len(ds) > batch_size)
    opt = torch.optim.AdamW(mct.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(dl)))
    log = []
    for ep in range(epochs):
        tot = 0.0
        for a, b in dl:
            a, b = a.to(device), b.to(device)
            loss, _ = mct_loss(mct(a), b, tau, lam_anchor, use_infonce)
            loss = loss + lam_orth * mct.orth_penalty()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        log.append(tot / max(1, len(dl)))
    mct.eval()
    return log


@torch.no_grad()
def apply_mct(mct, A, device, batch_size=1024):
    mct.to(device).eval()
    outs = []
    for i in range(0, len(A), batch_size):
        outs.append(mct(torch.as_tensor(A[i:i + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(outs)


class NeutralHub(torch.nn.Module):
    """Hub-and-spoke: her (model, katman-bandı) için encoder + decoder.

    Nötr uzay hiçbir modele ait değildir; tüm spoke'ların ortak
    contrastive eğitimiyle sıfırdan öğrenilir. 2N tercümanla N(N-1)
    yönlü çeviri: kaynak-encoder → hub → hedef-decoder.
    """

    def __init__(self, spoke_dims, hub_dim=512, hidden=1024,
                 mus=None):
        super().__init__()
        from .models import GatedResidualMCT
        self.hub_dim = hub_dim
        self.encoders = torch.nn.ModuleDict()
        self.decoders = torch.nn.ModuleDict()
        for name, d in spoke_dims.items():
            mu = None if mus is None else mus.get(name)
            self.encoders[name] = GatedResidualMCT(
                d, hub_dim, hidden, mu_in=mu)
            self.decoders[name] = GatedResidualMCT(
                hub_dim, d, hidden, mu_out=mu)

    def translate(self, x, src, dst):
        return self.decoders[dst](self.encoders[src](x))


def train_hub(hub, spokes_train, device, epochs=80, batch_size=256, lr=1e-3,
              tau=0.07, lam_anchor=0.25, lam_recon=0.5, lam_orth=0.1,
              lam_cycle=0.0, cycle_pairs=None, seed=42):
    """spokes_train: {name: np.ndarray [N, d]} — aynı N konsept, hizalı sıra.

    Kayıp = tüm spoke çiftleri arasında hub'da simetrik InfoNCE
           + her spoke için decode-rekonstrüksiyon (kosinüs)
           + çapraz çeviri çapası (src→hub→dst kosinüs).
    """
    torch.manual_seed(seed)
    names = list(spokes_train)
    N = len(next(iter(spokes_train.values())))
    hub.to(device).train()
    tensors = {k: torch.as_tensor(v) for k, v in spokes_train.items()}
    opt = torch.optim.AdamW(hub.parameters(), lr=lr, weight_decay=1e-4)
    n_batches = max(1, N // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * n_batches)
    log = []
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for bi in range(n_batches):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            x = {k: tensors[k][idx].to(device) for k in names}
            z = {k: hub.encoders[k](x[k]) for k in names}
            loss = 0.0
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    loss = loss + symmetric_infonce(z[a], z[b], tau)
            for k in names:
                loss = loss + lam_recon * cosine_anchor(hub.decoders[k](z[k]), x[k])
            for i, a in enumerate(names):
                for b in names:
                    if a != b:
                        loss = loss + lam_anchor * cosine_anchor(
                            hub.decoders[b](z[a]), x[b])
            for k in names:
                loss = loss + lam_orth * (hub.encoders[k].orth_penalty()
                                          + hub.decoders[k].orth_penalty())
            if lam_cycle > 0:
                # vec2vec tarzı cycle-consistency: a -> hub -> b -> hub -> a
                for a, b in (cycle_pairs or []):
                    y_b = hub.decoders[b](z[a])
                    x_cyc = hub.decoders[a](hub.encoders[b](y_b))
                    loss = loss + lam_cycle * cosine_anchor(x_cyc, x[a])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        log.append(tot / n_batches)
    hub.eval()
    return log


@torch.no_grad()
def hub_translate(hub, X, src, dst, device, batch_size=1024):
    hub.to(device).eval()
    outs = []
    for i in range(0, len(X), batch_size):
        x = torch.as_tensor(X[i:i + batch_size]).to(device)
        outs.append(hub.translate(x, src, dst).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def hub_encode(hub, X, src, device, batch_size=1024):
    hub.to(device).eval()
    outs = []
    for i in range(0, len(X), batch_size):
        x = torch.as_tensor(X[i:i + batch_size]).to(device)
        outs.append(hub.encoders[src](x).cpu().numpy())
    return np.concatenate(outs)
