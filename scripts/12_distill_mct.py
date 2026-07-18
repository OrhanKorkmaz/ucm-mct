"""H3-revize-2 — Konsept-sonlu şablon + uçtan uca distilasyon MCT.

İki düzeltme (temel amaç: çok dilli çeviri yeteneği):
 1. Konsept-SONLU şablonlar (EN "Word: X" / TR "Kelime: X"): son token
    konseptin kendi tokenı — TR'de son tokenın hep "anlamı" olması
    sorununu çözer; çıkarım ve enjeksiyon pozisyonu doğal olarak örtüşür.
 2. Distilasyon (tuned-lens ilkesi): vektör-eğitimli MCT, Qwen'in L12
    çıkışına enjekte edilip donuk kalan katmanlardan geçirilerek temiz
    next-token dağılımına KL ile hizalanır. Gradyan yalnız MCT'ye akar.

Karşılaştırma: vektör-MCT vs distilasyon-MCT, TR/EN ayrık, alpha=1 replace.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mct import get_device
from mct.concepts import load_or_build, train_test_split
from mct.extract import MODELS, extract_last_token, load_model
from mct.losses import mct_loss
from mct.metrics import mean_cosine, retrieval
from mct.models import make_mct
from mct.patching import kl_div, patch_next_token
from mct.standardize import zapply, zfit
from mct.train import apply_mct

TEMPLATES = {"en": "Word: {}", "tr": "Kelime: {}"}
LA, LB = 8, 12  # orta bant rotası (11'de enjeksiyonda en iyi)


def get_ctx(key, layer, prompts, device):
    path = ROOT / "data" / f"acts_cf_{key}.npz"
    if path.exists():
        z = np.load(path)
        return z[f"layer_{layer}"].astype(np.float32)
    acts = extract_last_token(key, prompts, device, [layer])
    np.savez_compressed(path, **{f"layer_{l}": v for l, v in acts.items()})
    return acts[layer].astype(np.float32)


def train_balanced(mct, A, B, en_idx, tr_idx, device, epochs=60,
                   batch_size=256, lr=1e-3, lam_orth=1.0, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mct.to(device).train()
    At, Bt = torch.as_tensor(A), torch.as_tensor(B)
    opt = torch.optim.AdamW(mct.parameters(), lr=lr, weight_decay=1e-4)
    half = batch_size // 2
    n_batches = max(1, len(en_idx) // half)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * n_batches)
    tr_pool = rng.permutation(tr_idx); tr_ptr = 0
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


def distill(mct, model, tok, prompts_all, A_std, stats_b, train_idx, device,
            epochs=8, batch_size=48, lr=1e-4, lam_anchor=0.5, B_std=None,
            seed=42):
    """Enjeksiyon-altında KL distilasyonu. Qwen donuk (fp32), grad -> MCT."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu_b, sd_b = [torch.as_tensor(x, dtype=torch.float32, device=device)
                  for x in stats_b]
    for p in model.parameters():
        p.requires_grad_(False)
    mct.to(device).train()
    A_t = torch.as_tensor(A_std)
    B_t = torch.as_tensor(B_std)
    opt = torch.optim.AdamW(mct.parameters(), lr=lr)
    layers = model.model.layers
    log = []
    for ep in range(epochs):
        perm = rng.permutation(train_idx)
        tot, nb = 0.0, 0
        for s in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[s:s + batch_size]
            enc = tok([prompts_all[i] for i in idx], return_tensors="pt",
                      padding=True).to(device)
            rows = torch.arange(len(idx), device=device)
            last = enc["attention_mask"].sum(1) - 1
            with torch.no_grad():
                clean = model(**enc).logits[rows, last]
                logp_clean = F.log_softmax(clean.float(), -1)
            v = mct(A_t[idx].to(device)) * sd_b + mu_b  # gerçek uzay
            state = {}

            def hook(module, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs = hs.clone()
                vv = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
                cn = hs[rows, last].norm(dim=-1, keepdim=True)
                hs[rows, last] = (vv * cn).to(hs.dtype)
                return (hs,) + output[1:] if isinstance(output, tuple) else hs

            h = layers[LB - 1].register_forward_hook(hook)
            try:
                patched = model(**enc).logits[rows, last]
            finally:
                h.remove()
            logp_p = F.log_softmax(patched.float(), -1)
            kl = (logp_clean.exp() * (logp_clean - logp_p)).sum(-1).mean()
            anchor = 1 - F.cosine_similarity(
                mct(A_t[idx].to(device)), B_t[idx].to(device), dim=-1).mean()
            loss = kl + lam_anchor * anchor
            opt.zero_grad(); loss.backward(); opt.step()
            tot += kl.item(); nb += 1
        log.append(tot / max(nb, 1))
        print(f"  distill ep{ep}: KL={log[-1]:.4f}", flush=True)
    mct.eval()
    return log


def eval_injection(model, tok, device, words, langs, sel, pred_raw, B_raw,
                   shuffle):
    rows = []
    for i in sel:
        tmpl = TEMPLATES[langs[i]]
        lp_c = patch_next_token(model, tok, device, words[i], LB, None, tmpl, "last")
        lp_m = patch_next_token(model, tok, device, words[i], LB, pred_raw[i],
                                tmpl, "last")
        lp_r = patch_next_token(model, tok, device, words[i], LB,
                                B_raw[shuffle[i]], tmpl, "last")
        rows.append({"lang": langs[i], "kl_mct": kl_div(lp_c, lp_m),
                     "kl_rand": kl_div(lp_c, lp_r),
                     "agree_mct": int(lp_m.argmax() == lp_c.argmax()),
                     "agree_rand": int(lp_r.argmax() == lp_c.argmax())})

    def agg(rs):
        a = {k: float(np.mean([r[k] for r in rs]))
             for k in ("kl_mct", "kl_rand", "agree_mct", "agree_rand")}
        a["kl_ratio"] = a["kl_mct"] / max(a["kl_rand"], 1e-9)
        a["n"] = len(rs)
        return a
    return {"all": agg(rows), "en": agg([r for r in rows if r["lang"] == "en"]),
            "tr": agg([r for r in rows if r["lang"] == "tr"])}


if __name__ == "__main__":
    device = get_device()
    cpu = torch.device("cpu")
    concepts = load_or_build()
    words = [c["text"] for c in concepts]
    langs = np.array([c["lang"] for c in concepts])
    prompts = [TEMPLATES[c["lang"]].format(c["text"]) for c in concepts]
    tr_i, te_i = train_test_split(len(concepts))

    A_raw = get_ctx("llama", LA, prompts, device)
    B_raw = get_ctx("qwen", LB, prompts, device)
    mu_a, sd_a = zfit(A_raw[tr_i]); mu_b, sd_b = zfit(B_raw[tr_i])
    A, B = zapply(A_raw, mu_a, sd_a), zapply(B_raw, mu_b, sd_b)
    en_tr = tr_i[langs[tr_i] == "en"]; tr_tr = tr_i[langs[tr_i] == "tr"]

    out = {"template": TEMPLATES, "route": f"llama.L{LA}->qwen.L{LB}"}

    print("1) Vektör-MCT (dil-dengeli, konsept-sonlu şablon):", flush=True)
    mct = make_mct(A[tr_i], B[tr_i])
    train_balanced(mct, A, B, en_tr, tr_tr, cpu)
    pred = apply_mct(mct, A, cpu)
    for lg in ("en", "tr"):
        m = langs[te_i] == lg
        out[f"retrieval_{lg}"] = {**retrieval(pred[te_i][m], B, te_i[m]),
                                  "cosine": mean_cosine(pred[te_i][m], B[te_i][m]),
                                  "n": int(m.sum())}
        print(f"  {lg}: {out[f'retrieval_{lg}']}", flush=True)

    # Qwen fp32 (distilasyon gradyanı için sayısal güvenli)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["qwen"])
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["qwen"], dtype=torch.float32).to(device).eval()

    rng = np.random.default_rng(0)
    te_en = te_i[langs[te_i] == "en"]; te_tr = te_i[langs[te_i] == "tr"]
    sel = np.concatenate([rng.choice(te_en, 100, replace=False),
                          rng.choice(te_tr, 50, replace=False)])
    shuffle = rng.permutation(len(concepts))

    pred_raw = pred * sd_b + mu_b
    t0 = time.time()
    out["injection_vector_mct"] = eval_injection(
        model, tok, device, words, langs, sel, pred_raw, B_raw, shuffle)
    r = out["injection_vector_mct"]
    print(f"2) Enjeksiyon (vektör-MCT): KLr={r['all']['kl_ratio']:.3f} "
          f"uyum={r['all']['agree_mct']:.3f} "
          f"(EN {r['en']['agree_mct']:.2f}/TR {r['tr']['agree_mct']:.2f}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

    print("3) Distilasyon ince ayarı (KL, Qwen donuk):", flush=True)
    mct_d = make_mct(A[tr_i], B[tr_i])
    mct_d.load_state_dict(mct.state_dict())
    mct_d.to(device)
    # dil-dengeli distilasyon alt-kümesi: 2000 EN + 1600 TR
    sub = np.concatenate([rng.choice(en_tr, 2000, replace=False),
                          rng.choice(tr_tr, min(1600, len(tr_tr)), replace=False)])
    distill(mct_d, model, tok, prompts, A, (mu_b, sd_b), sub, device,
            B_std=B)
    pred_d = apply_mct(mct_d, A, device)
    for lg in ("en", "tr"):
        m = langs[te_i] == lg
        out[f"retrieval_distill_{lg}"] = {
            **retrieval(pred_d[te_i][m], B, te_i[m]),
            "cosine": mean_cosine(pred_d[te_i][m], B[te_i][m]), "n": int(m.sum())}
        print(f"  {lg}: {out[f'retrieval_distill_{lg}']}", flush=True)
    pred_d_raw = pred_d * sd_b + mu_b
    out["injection_distill_mct"] = eval_injection(
        model, tok, device, words, langs, sel, pred_d_raw, B_raw, shuffle)
    r = out["injection_distill_mct"]
    r["H3_pass"] = bool(r["all"]["kl_ratio"] < 0.2 and r["all"]["agree_mct"] > 0.6)
    print(f"4) Enjeksiyon (distill-MCT): KLr={r['all']['kl_ratio']:.3f} "
          f"uyum={r['all']['agree_mct']:.3f} "
          f"(EN {r['en']['agree_mct']:.2f}/TR {r['tr']['agree_mct']:.2f}) "
          f"H3={'GEÇTİ' if r['H3_pass'] else 'kaldı'}", flush=True)

    torch.save(mct_d.state_dict(), ROOT / "results" / "mct_distill_mid.pt")
    (ROOT / "results" / "h3_distill.json").write_text(json.dumps(out, indent=2))
    print("DISTILL DONE")
