"""FAZ 1 / Adım 1b — CKA katman taraması.

Katman-2 bandı (derinliğin %30-50'si) ve Katman-3 bandı (%60-75) içinde
Llama-katmanı × Qwen-katmanı CKA matrisi hesaplanır; her bant için CKA'yı
maksimize eden çift seçilir. Ayrıca model-içi derinlik çevirisi (FAZ 2)
için aynı modelin L2-L3 katmanları raporlanır.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mct.cka import band, cka_matrix, pick_pair
from mct.extract import load_acts
from mct.standardize import zapply, zfit

BANDS = {"L2": (0.30, 0.50), "L3": (0.60, 0.75)}
N_SCAN = 2000  # tarama alt-örneklemi

if __name__ == "__main__":
    acts_l = load_acts("llama")
    acts_q = load_acts("qwen")
    nl_l = max(acts_l) ; nl_q = max(acts_q)
    print(f"llama: {nl_l} katman d={acts_l[1].shape[1]}; "
          f"qwen: {nl_q} katman d={acts_q[1].shape[1]}")
    rng = np.random.default_rng(0)
    sub = rng.choice(len(acts_l[1]), N_SCAN, replace=False)

    def z(v):
        # sink/outlier boyutların CKA'yı domine etmesini engelle
        return zapply(v[sub], *zfit(v[sub]))

    acts_l_s = {k: z(v) for k, v in acts_l.items()}
    acts_q_s = {k: z(v) for k, v in acts_q.items()}
    for name, acts in (("llama", acts_l_s), ("qwen", acts_q_s)):
        bad = sum(int(np.isnan(v).any()) for v in acts.values())
        assert bad == 0, f"{name}: {bad} katmanda NaN!"

    out = {}
    for name, (lo, hi) in BANDS.items():
        res = pick_pair(acts_l_s, acts_q_s, nl_l, nl_q, lo, hi)
        out[name] = res
        print(f"{name}: llama L{res['layer_a']} <-> qwen L{res['layer_b']} "
              f"(CKA={res['cka']:.3f}; bant llama {res['layers_a']}, "
              f"qwen {res['layers_b']})")

    # model-içi L2->L3 CKA (derinlik çevirisi referansı)
    for key, acts_s, nl in [("llama", acts_l_s, nl_l), ("qwen", acts_q_s, nl_q)]:
        l2 = out["L2"]["layer_a" if key == "llama" else "layer_b"]
        l3 = out["L3"]["layer_a" if key == "llama" else "layer_b"]
        M = cka_matrix(acts_s, acts_s, [l2], [l3])
        out[f"{key}_intra_L2L3_cka"] = float(M[0, 0])
        print(f"{key} içi L{l2}->L{l3} CKA: {M[0,0]:.3f}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "cka.json").write_text(json.dumps(out, indent=2))
    print("CKA DONE")
