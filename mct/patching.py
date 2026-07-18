"""Activation patching ile işlevsellik testi (H3).

Üç koşullu protokol:
  temiz geçiş     — hedef modelin kendi aktivasyonu (alt sınır, KL=0)
  MCT-enjeksiyon  — kaynak modelden çevrilen vektör enjekte edilir (test)
  rastgele vektör — karıştırılmış konsept vektörü (üst sınır / kontrol)

Enjeksiyondan önce norm kalibrasyonu: çevrilen vektör, temiz aktivasyonun
normuna ölçeklenir (anizotropi konisi dışına düşmeyi engeller).
"""
import numpy as np
import torch
import torch.nn.functional as F


def _get_layers(model):
    return model.model.layers


@torch.no_grad()
def patch_next_token(model, tok, device, text, layer, vector=None,
                     template=" {}", positions="all", alpha=1.0):
    """Vektörü enjekte edip son-pozisyon next-token dağılımını döndürür.

    positions="all": BOS hariç tüm prompt pozisyonları (eski protokol).
    positions="last": yalnızca son token (ICML 2501.14082 protokolü).
    alpha: enjeksiyon şiddeti (norm çarpanı).
    vector=None → temiz geçiş.
    """
    prompt = template.format(text)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    ids = enc["input_ids"][0]
    if positions == "last":
        pos = [len(ids) - 1]
    else:
        pos = [i for i, t in enumerate(ids.tolist())
               if t != tok.bos_token_id]
    handle = None
    if vector is not None:
        v = torch.as_tensor(vector, dtype=torch.float32, device=device)

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            vv = (v / (v.norm() + 1e-6)).to(hs.dtype)
            for p in pos:
                clean_norm = hs[0, p].norm()
                hs[0, p] = vv * clean_norm * alpha  # norm kalibrasyonu × şiddet
            return (hs,) + output[1:] if isinstance(output, tuple) else hs

        handle = _get_layers(model)[layer - 1].register_forward_hook(hook)
    try:
        logits = model(**enc).logits[0, -1].float()
    finally:
        if handle:
            handle.remove()
    return F.log_softmax(logits, dim=-1).cpu()


def kl_div(logp_p, logp_q):
    """KL(p ‖ q) — her ikisi log-prob."""
    p = logp_p.exp()
    return float((p * (logp_p - logp_q)).sum())


def eval_patching(model, tok, device, concepts, translated, clean_vectors,
                  layer, template=" {} ", seed=0, max_n=200):
    """concepts: metin listesi; translated[i]: MCT çevirisi;
    clean_vectors[i]: hedef modelin kendi vektörü (rastgele kontrol için
    karıştırılır)."""
    rng = np.random.default_rng(seed)
    n = min(max_n, len(concepts))
    idx = rng.choice(len(concepts), n, replace=False)
    shuffle = rng.permutation(len(concepts))
    rows = []
    for i in idx:
        text = concepts[i]
        lp_clean = patch_next_token(model, tok, device, text, layer, None, template)
        lp_mct = patch_next_token(model, tok, device, text, layer,
                                  translated[i], template)
        lp_rand = patch_next_token(model, tok, device, text, layer,
                                   clean_vectors[shuffle[i]], template)
        rows.append({
            "kl_mct": kl_div(lp_clean, lp_mct),
            "kl_rand": kl_div(lp_clean, lp_rand),
            "top1_agree_mct": int(lp_mct.argmax() == lp_clean.argmax()),
            "top1_agree_rand": int(lp_rand.argmax() == lp_clean.argmax()),
        })
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["kl_ratio"] = agg["kl_mct"] / max(agg["kl_rand"], 1e-9)
    agg["n"] = n
    return agg, rows
