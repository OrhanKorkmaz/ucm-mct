"""H4 kapanış denemesi — BatchTopK SAE + aux-k ölü-özellik canlandırma.

Öncekinden farklar (literatür reçetesi, Bussmann vd. 2412.06410):
  - BatchTopK: örnek-başına değil batch-genelinde top-(k·B) seçimi
  - aux-k kaybı: uzun süre ateşlenmeyen özellikler artık hatayı
    rekonstrükte etmeye zorlanır (ölü özellik canlanır)
  - Daha çok veri (6 spoke × 8k = 48k hub vektörü) + 100 epoch
  - Değerlendirme: 3-aile hub'ının llama_L10->qwen_L18 yolu
    (SAE'siz referans top-1 %87.7)
H4: oran >= 8x'te top-1 düşüşü <= 5 puan.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mct import get_device
from mct.concepts import load_or_build, train_test_split
from mct.metrics import mean_cosine, retrieval
from mct.standardize import zapply, zfit
from mct.train import NeutralHub, hub_encode

HUB_DIM = 768


class BatchTopKSAE(nn.Module):
    def __init__(self, d, n_features, k, k_aux=64, dead_steps=200):
        super().__init__()
        self.k, self.k_aux, self.dead_steps = k, k_aux, dead_steps
        self.enc = nn.Linear(d, n_features)
        self.dec = nn.Linear(n_features, d)
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.T)
        self.register_buffer("last_fired", torch.zeros(n_features))
        self.step = 0

    def encode_batchtopk(self, x):
        f = torch.relu(self.enc(x))
        kb = self.k * x.shape[0]
        flat = f.flatten()
        if kb < flat.numel():
            thresh = flat.topk(kb).values[-1]
            f = torch.where(f >= thresh, f, torch.zeros_like(f))
        return f

    def encode_eval(self, x):  # tek-mesaj: örnek-başına top-k
        f = torch.relu(self.enc(x))
        topv, topi = f.topk(self.k, dim=-1)
        return torch.zeros_like(f).scatter_(-1, topi, topv)

    def forward_train(self, x):
        f = self.encode_batchtopk(x)
        recon = self.dec(f)
        with torch.no_grad():
            fired = (f > 0).any(0)
            self.last_fired[fired] = self.step
            dead = (self.step - self.last_fired) > self.dead_steps
        aux = None
        if dead.any():
            f_all = torch.relu(self.enc(x))
            f_dead = torch.where(dead[None, :], f_all, torch.zeros_like(f_all))
            kv, ki = f_dead.topk(min(self.k_aux, int(dead.sum())), dim=-1)
            f_aux = torch.zeros_like(f_all).scatter_(-1, ki, kv)
            aux = self.dec(f_aux)
        self.step += 1
        return recon, aux


def train_batchtopk(sae, X, device, epochs=100, batch_size=512, lr=1e-3,
                    alpha_aux=1 / 32, seed=42):
    torch.manual_seed(seed)
    sae.to(device).train()
    mu = torch.as_tensor(X.mean(0)).to(device)
    sd = torch.as_tensor(X.std() + 1e-6).to(device)
    Xt = torch.as_tensor(X)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    N = len(X)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for s in range(0, N - batch_size + 1, batch_size):
            x = ((Xt[perm[s:s + batch_size]].to(device)) - mu) / sd
            recon, aux = sae.forward_train(x)
            loss = ((recon - x) ** 2).mean()
            if aux is not None:
                loss = loss + alpha_aux * ((aux - (x - recon.detach())) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    sae.eval(); sae.mu, sae.sd = mu, sd
    return sae


@torch.no_grad()
def roundtrip(sae, X, device, bs=2048):
    outs, active = [], []
    for i in range(0, len(X), bs):
        x = (torch.as_tensor(X[i:i + bs]).to(device) - sae.mu) / sae.sd
        f = sae.encode_eval(x)
        active.append((f > 0).float().sum(-1).mean().item())
        outs.append((sae.dec(f) * sae.sd + sae.mu).cpu().numpy())
    dead = int(((sae.step - sae.last_fired) > sae.dead_steps).sum())
    return np.concatenate(outs), float(np.mean(active)), dead


if __name__ == "__main__":
    device = get_device()
    cpu = torch.device("cpu")
    concepts = load_or_build()
    tr_i, te_i = train_test_split(len(concepts))
    z = np.load(ROOT / "data" / "acts_cf2.npz")
    raw = {k: z[k].astype(np.float32) for k in z.files if k != "anchor"}
    zs = np.load(ROOT / "data" / "acts_smol.npz")
    for k in zs.files:
        raw[k] = zs[k].astype(np.float32)
    spokes = {k: zapply(v, *zfit(v[tr_i])) for k, v in raw.items()}

    hub = NeutralHub({k: v.shape[1] for k, v in spokes.items()}, hub_dim=HUB_DIM)
    hub.load_state_dict(torch.load(ROOT / "results" / "hub_3family.pt",
                                   weights_only=True))
    hub.to(cpu).eval()

    Z_tr = np.concatenate([hub_encode(hub, spokes[k][tr_i], k, cpu)
                           for k in spokes])
    Z_te = hub_encode(hub, spokes["llama_L10"][te_i], "llama_L10", cpu)
    print(f"SAE eğitim verisi: {Z_tr.shape}", flush=True)

    @torch.no_grad()
    def e2e(Zv, tag):
        pred = hub.decoders["qwen_L18"](torch.as_tensor(Zv)).numpy()
        m = {**retrieval(pred, spokes["qwen_L18"], te_i),
             "cosine": mean_cosine(pred, spokes["qwen_L18"][te_i])}
        print(f"  {tag}: top1={m['top1']:.3f} cos={m['cosine']:.3f}", flush=True)
        return m

    ref = e2e(Z_te, "SAE'siz referans")
    results = {"reference": ref, "sweep": []}
    for F in (2048, 4096):
        for k in (16, 32, 48):
            t0 = time.time()
            sae = BatchTopKSAE(HUB_DIM, F, k)
            train_batchtopk(sae, Z_tr, device, epochs=100)
            rec, active, dead = roundtrip(sae, Z_te, device)
            m = e2e(rec, f"F={F} k={k}")
            row = {"F": F, "k": k, "ratio_x": HUB_DIM / (2 * k),
                   "recon_cos": mean_cosine(rec, Z_te), "active": active,
                   "dead": dead, **m,
                   "drop_pts": round(100 * (ref["top1"] - m["top1"]), 1),
                   "train_s": round(time.time() - t0)}
            results["sweep"].append(row)
            print(f"    oran={row['ratio_x']:.0f}x düşüş={row['drop_pts']} puan "
                  f"ölü={dead} [{row['train_s']}s]", flush=True)

    cands = [r for r in results["sweep"] if r["ratio_x"] >= 8]
    best = max(cands, key=lambda r: r["top1"]) if cands else None
    if best:
        results["H4"] = {"best": {k: best[k] for k in ("F", "k", "ratio_x",
                                                       "top1", "drop_pts")},
                         "pass": bool(best["drop_pts"] <= 5.0)}
        print(f"H4: F={best['F']} k={best['k']} @{best['ratio_x']:.0f}x "
              f"düşüş {best['drop_pts']} puan -> "
              f"{'GEÇTİ' if results['H4']['pass'] else 'kaldı'}", flush=True)
    (ROOT / "results" / "h4_batchtopk.json").write_text(
        json.dumps(results, indent=2))
    print("H4-BTK DONE")
