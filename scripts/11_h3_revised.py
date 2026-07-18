"""H3-revize — çok dilli bağlamsal MCT + tek-pozisyon enjeksiyon taraması.

Çok-dillilik ana eksen (temel amaç):
  - Eğitim DİL-DENGELİ batch'lerle (her batch %50 EN + %50 TR; TR listesi
    epoch içinde yeniden karılarak döndürülür, batch içi kopya yok).
  - Tüm metrikler TR/EN ayrık raporlanır.

İki çeviri yolu (ICML 2501.14082 + CKA bulgularımız):
  MCT-mid: llama.L8  -> qwen.L12  (orta bant, "varlık temsilleri")
  MCT-sem: llama.L10 -> qwen.L18  (anlamsal bant, CKA seçimi)

Enjeksiyon: yalnızca SON token, replace + norm kalibrasyonu,
şiddet alpha ∈ {1, 2}; üç koşul (temiz / MCT / rastgele).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mct import get_device
from mct.concepts import load_or_build, train_test_split
from mct.extract import load_model
from mct.losses import mct_loss
from mct.metrics import mean_cosine, retrieval
from mct.models import make_mct
from mct.patching import kl_div, patch_next_token
from mct.standardize import zapply, zfit
from mct.train import apply_mct

TEMPLATES = {"en": "The word '{}' means", "tr": "'{}' kelimesinin anlamı"}
ROUTES = {"mid": ("llama", 8, "qwen", 12), "sem": ("llama", 10, "qwen", 18)}


def load_ctx(key, layer):
    z = np.load(ROOT / "data" / f"acts_ctx_{key}.npz")
    return z[f"layer_{layer}"].astype(np.float32)


def train_balanced(mct, A, B, en_idx, tr_idx, device, epochs=60,
                   batch_size=256, lr=1e-3, lam_orth=1.0, seed=42):
    """Dil-dengeli InfoNCE eğitimi: her batch yarı EN yarı TR."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mct.to(device).train()
    At, Bt = torch.as_tensor(A), torch.as_tensor(B)
    opt = torch.optim.AdamW(mct.parameters(), lr=lr, weight_decay=1e-4)
    half = batch_size // 2
    n_batches = max(1, len(en_idx) // half)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * n_batches)
    tr_pool = rng.permutation(tr_idx)
    tr_ptr = 0
    for ep in range(epochs):
        en_pool = rng.permutation(en_idx)
        for bi in range(n_batches):
            en_b = en_pool[bi * half:(bi + 1) * half]
            if tr_ptr + half > len(tr_pool):
                tr_pool = rng.permutation(tr_idx); tr_ptr = 0
            tr_b = tr_pool[tr_ptr:tr_ptr + half]; tr_ptr += half
            idx = torch.as_tensor(np.concatenate([en_b, tr_b]))
            a, b = At[idx].to(device), Bt[idx].to(device)
            loss, _ = mct_loss(mct(a), b)
            loss = loss + lam_orth * mct.orth_penalty()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    mct.eval()
    return mct


if __name__ == "__main__":
    device = get_device()
    cpu = torch.device("cpu")  # MCT eğitimi CPU'da (küçük), enjeksiyon MPS'te
    concepts = load_or_build()
    words = [c["text"] for c in concepts]
    langs = np.array([c["lang"] for c in concepts])
    tr_i, te_i = train_test_split(len(concepts))
    out = {"routes": {}}

    preds_raw = {}   # enjeksiyon için gerçek-uzay çeviriler
    B_raw_all = {}
    for name, (ka, la, kb, lb) in ROUTES.items():
        A_raw, B_raw = load_ctx(ka, la), load_ctx(kb, lb)
        mu_a, sd_a = zfit(A_raw[tr_i]); mu_b, sd_b = zfit(B_raw[tr_i])
        A, B = zapply(A_raw, mu_a, sd_a), zapply(B_raw, mu_b, sd_b)
        en_tr = tr_i[langs[tr_i] == "en"]; tr_tr = tr_i[langs[tr_i] == "tr"]
        t0 = time.time()
        mct = make_mct(A[tr_i], B[tr_i])
        train_balanced(mct, A, B, en_tr, tr_tr, cpu)
        pred = apply_mct(mct, A, cpu)
        m = {"train_s": round(time.time() - t0, 1)}
        for lg in ("en", "tr"):
            mask = langs[te_i] == lg
            m[f"retrieval_{lg}"] = {
                **retrieval(pred[te_i][mask], B, te_i[mask]),
                "cosine": mean_cosine(pred[te_i][mask], B[te_i][mask]),
                "n": int(mask.sum())}
        out["routes"][name] = m
        print(name, {k: v for k, v in m.items() if k != "train_s"}, flush=True)
        preds_raw[name] = pred * sd_b + mu_b
        B_raw_all[name] = B_raw
        torch.save(mct.state_dict(), ROOT / "results" / f"mct_ctx_{name}.pt")

    # --- Enjeksiyon taraması (tek pozisyon, replace, alpha) ---
    tok, model = load_model("qwen", device)
    rng = np.random.default_rng(0)
    te_en = te_i[langs[te_i] == "en"]; te_tr = te_i[langs[te_i] == "tr"]
    sel = np.concatenate([rng.choice(te_en, 100, replace=False),
                          rng.choice(te_tr, 50, replace=False)])
    shuffle = rng.permutation(len(concepts))
    inj = {}
    for name, (ka, la, kb, lb) in ROUTES.items():
        for alpha in (1.0, 2.0):
            rows = []
            t0 = time.time()
            for i in sel:
                lg = langs[i]
                tmpl = TEMPLATES[lg]
                lp_c = patch_next_token(model, tok, device, words[i], lb,
                                        None, tmpl, "last")
                lp_m = patch_next_token(model, tok, device, words[i], lb,
                                        preds_raw[name][i], tmpl, "last", alpha)
                lp_r = patch_next_token(model, tok, device, words[i], lb,
                                        B_raw_all[name][shuffle[i]], tmpl,
                                        "last", alpha)
                rows.append({"lang": lg,
                             "kl_mct": kl_div(lp_c, lp_m),
                             "kl_rand": kl_div(lp_c, lp_r),
                             "agree_mct": int(lp_m.argmax() == lp_c.argmax()),
                             "agree_rand": int(lp_r.argmax() == lp_c.argmax())})
            def agg(rs):
                if not rs:
                    return {}
                a = {k: float(np.mean([r[k] for r in rs]))
                     for k in ("kl_mct", "kl_rand", "agree_mct", "agree_rand")}
                a["kl_ratio"] = a["kl_mct"] / max(a["kl_rand"], 1e-9)
                a["n"] = len(rs)
                return a
            res = {"all": agg(rows),
                   "en": agg([r for r in rows if r["lang"] == "en"]),
                   "tr": agg([r for r in rows if r["lang"] == "tr"]),
                   "seconds": round(time.time() - t0, 1)}
            res["H3_pass"] = bool(res["all"]["kl_ratio"] < 0.2
                                  and res["all"]["agree_mct"] > 0.6)
            inj[f"{name}_alpha{alpha:g}"] = res
            print(f"{name} a={alpha:g}: KLr={res['all']['kl_ratio']:.3f} "
                  f"uyum={res['all']['agree_mct']:.3f} "
                  f"(EN {res['en']['agree_mct']:.2f}/TR {res['tr']['agree_mct']:.2f}) "
                  f"H3={'GEÇTİ' if res['H3_pass'] else 'kaldı'}", flush=True)
    out["injection"] = inj
    (ROOT / "results" / "h3_revised.json").write_text(json.dumps(out, indent=2))
    print("H3-REVISED DONE")
