"""E3 ön-alma: kanarya seti + drift tespiti + ucuz yeniden-kalibrasyon.

Senaryo: llama güncellendi (ince-ayar) ve L10 temsilleri kaydı. Simülasyon:
  X_drift(a) = (1-a)·L10 + a·L8 + gürültü   (yapısal drift: temsiller
  erken-katman geometrisine kayar; Feature Drift bulgusuyla uyumlu)

Protokol:
 1. Kanarya seti: 500 sabit konsept. Drift ölçüsü: linear-CKA(temiz, driftli).
 2. Etki: donuk 3-aile hub'ında llama_L10->qwen_L18 top-1'i ne kadar düşüyor?
 3. Onarım: YALNIZ kanarya üzerinde kapalı-form Procrustes adaptörü
    (driftli -> temiz uzay) öne eklenir; top-1 geri geliyor mu?
Çıktı: drift eşiği kuralı (CKA < eşik -> yeniden kalibre et).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mct.concepts import load_or_build, train_test_split
from mct.cka import linear_cka
from mct.metrics import retrieval
from mct.models import procrustes_fit
from mct.standardize import zapply, zfit
from mct.train import NeutralHub, hub_translate

HUB_DIM = 768

if __name__ == "__main__":
    cpu = torch.device("cpu")
    rng = np.random.default_rng(7)
    concepts = load_or_build()
    tr_i, te_i = train_test_split(len(concepts))
    z = np.load(ROOT / "data" / "acts_cf2.npz")
    raw = {k: z[k].astype(np.float32) for k in z.files if k != "anchor"}
    zs = np.load(ROOT / "data" / "acts_smol.npz")
    for k in zs.files:
        raw[k] = zs[k].astype(np.float32)
    stats = {k: zfit(v[tr_i]) for k, v in raw.items()}
    spokes = {k: zapply(v, *stats[k]) for k, v in raw.items()}

    hub = NeutralHub({k: v.shape[1] for k, v in spokes.items()}, hub_dim=HUB_DIM)
    hub.load_state_dict(torch.load(ROOT / "results" / "hub_3family.pt",
                                   weights_only=True))
    hub.to(cpu).eval()

    canary = rng.choice(te_i, 500, replace=False)          # kanarya seti
    eval_i = np.array([i for i in te_i if i not in set(canary)])

    def top1(X_llama_std):
        pred = hub_translate(hub, X_llama_std[eval_i], "llama_L10",
                             "qwen_L18", cpu)
        return retrieval(pred, spokes["qwen_L18"], eval_i)["top1"]

    ref = top1(spokes["llama_L10"])
    print(f"referans top-1 (driftsiz): {ref:.3f}", flush=True)

    out = {"reference_top1": ref, "rows": []}
    L10, L8 = raw["llama_L10"], raw["llama_L8"]
    for a in (0.05, 0.1, 0.2, 0.4):
        drift = (1 - a) * L10 + a * L8
        drift = drift + rng.normal(0, 0.02 * drift.std(), drift.shape).astype(np.float32)
        cka = linear_cka(drift[canary], L10[canary])
        Xd = zapply(drift, *stats["llama_L10"])   # eski istatistiklerle (gerçekçi)
        t1_broken = top1(Xd)
        # onarım: kanaryada kapalı-form Procrustes (driftli std -> temiz std)
        Xd_c, Xc_c = Xd[canary], spokes["llama_L10"][canary]
        mu_d, mu_c = Xd_c.mean(0), Xc_c.mean(0)
        W, s = procrustes_fit(Xd_c - mu_d, Xc_c - mu_c)
        X_fixed = (Xd - mu_d) @ (s * W).T + mu_c
        t1_fixed = top1(X_fixed.astype(np.float32))
        row = {"alpha": a, "canary_cka": round(float(cka), 4),
               "top1_broken": t1_broken, "top1_recalibrated": t1_fixed,
               "recovery_pct": round(100 * t1_fixed / ref, 1)}
        out["rows"].append(row)
        print(f"a={a}: CKA={row['canary_cka']} kırık={t1_broken:.3f} "
              f"onarılmış={t1_fixed:.3f} (geri kazanım %{row['recovery_pct']})",
              flush=True)

    # eşik kuralı: %95 geri-kazanımın altına düşmeden önceki en düşük CKA
    ok = [r for r in out["rows"] if r["top1_broken"] >= 0.95 * ref]
    out["rule"] = {"cka_threshold": max((r["canary_cka"] for r in out["rows"]
                                         if r not in ok), default=None),
                   "note": "kanarya CKA bu eşiğin altına inerse Procrustes "
                           "yeniden-kalibrasyonu tetikle (kapalı-form, ~sn)"}
    (ROOT / "results" / "canary_drift.json").write_text(json.dumps(out, indent=2))
    print("CANARY DONE")
