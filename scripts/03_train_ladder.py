"""FAZ 1 / Adım 2 — MCT eğitimi: yöntem karşılaştırması + mikro ölçek merdiveni.

Yol: llama.L3 -> qwen.L3 (CKA ile seçilen anlamsal katman çifti).
Yöntemler: Procrustes (kapalı form), Ridge, MCT-cos, MCT-InfoNCE.
Merdiven: 500 -> 2000 -> 8000 eğitim konsepti; test daima aynı 2000.
Metrikler: cosine, top-1/5/20 retrieval (10k havuz, zor koşul), geometri ρ.
H1: geometri ρ (Procrustes çekirdek) ; H2: top-1 >= %80.
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
from mct.extract import load_acts
from mct.metrics import geometry_spearman, mean_cosine, retrieval
from mct.models import ProcrustesBaseline, RidgeBaseline, make_mct
from mct.standardize import zapply, zfit
from mct.train import apply_mct, train_mct

if __name__ == "__main__":
    device = get_device()
    cka = json.loads((ROOT / "results" / "cka.json").read_text())
    LA, LB = cka["L3"]["layer_a"], cka["L3"]["layer_b"]
    print(f"katman çifti: llama L{LA} -> qwen L{LB}")

    A = load_acts("llama", [LA])[LA]
    B = load_acts("qwen", [LB])[LB]
    concepts = load_or_build()
    tr_idx, te_idx = train_test_split(len(concepts))
    # z-skor standardizasyonu (eğitim istatistikleriyle; sink boyut düzeltmesi)
    mu_a, sd_a = zfit(A[tr_idx]); mu_b, sd_b = zfit(B[tr_idx])
    A, B = zapply(A, mu_a, sd_a), zapply(B, mu_b, sd_b)
    np.savez(ROOT / "results" / "std_stats.npz",
             mu_a=mu_a, sd_a=sd_a, mu_b=mu_b, sd_b=sd_b)
    A_tr_full, B_tr_full = A[tr_idx], B[tr_idx]
    A_te, B_te = A[te_idx], B[te_idx]
    pool = B  # 10k havuz — test vektörleri de içinde (zor koşul)

    def evaluate(pred_te, name):
        m = {"cosine": mean_cosine(pred_te, B_te),
             **retrieval(pred_te, pool, te_idx),
             "geometry_rho": geometry_spearman(A_te, pred_te)}
        print(f"  {name}: cos={m['cosine']:.3f} top1={m['top1']:.3f} "
              f"top5={m['top5']:.3f} top20={m['top20']:.3f} rho={m['geometry_rho']:.3f}",
              flush=True)
        return m

    results = {"layer_pair": {"llama": LA, "qwen": LB}, "methods": {},
               "ladder": {}}

    # --- Yöntem karşılaştırması (8000 eğitim) ---
    print("Yöntem karşılaştırması (n_train=8000):", flush=True)
    t0 = time.time()
    proc = ProcrustesBaseline(A_tr_full, B_tr_full)
    results["methods"]["procrustes"] = evaluate(proc(A_te), "Procrustes")
    ridge = RidgeBaseline(A_tr_full, B_tr_full)
    results["methods"]["ridge"] = evaluate(ridge(A_te), "Ridge")
    for tag, use_nce in [("mct_cos", False), ("mct_infonce", True)]:
        mct = make_mct(A_tr_full, B_tr_full)
        train_mct(mct, A_tr_full, B_tr_full, device, epochs=60,
                  use_infonce=use_nce)
        results["methods"][tag] = evaluate(apply_mct(mct, A_te, device), tag)
        if tag == "mct_infonce":
            torch.save(mct.state_dict(), ROOT / "results" / "mct_llama3_qwen3.pt")
    print(f"  süre: {time.time()-t0:.0f}s", flush=True)

    # --- Mikro ölçek merdiveni ---
    print("Mikro ölçek merdiveni (MCT-InfoNCE + Procrustes):", flush=True)
    for n in (500, 2000, 8000):
        A_tr, B_tr = A_tr_full[:n], B_tr_full[:n]
        proc = ProcrustesBaseline(A_tr, B_tr)
        m_p = evaluate(proc(A_te), f"procrustes@{n}")
        mct = make_mct(A_tr, B_tr)
        train_mct(mct, A_tr, B_tr, device, epochs=60, use_infonce=True)
        m_i = evaluate(apply_mct(mct, A_te, device), f"infonce@{n}")
        results["ladder"][str(n)] = {"procrustes": m_p, "mct_infonce": m_i}

    (ROOT / "results" / "ladder.json").write_text(json.dumps(results, indent=2))
    print("LADDER DONE")
