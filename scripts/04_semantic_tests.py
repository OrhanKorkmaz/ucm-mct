"""FAZ 1 / Adım 3 — Anlamsal uzay denemeleri.

En iyi MCT (InfoNCE) ile: k-NN tutarlılığı, analoji (vektör aritmetiği),
TR/EN alt-küme ayrık raporlama, niteliksel komşuluk örnekleri.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mct import get_device
from mct.concepts import load_or_build, train_test_split
from mct.extract import load_acts
from mct.metrics import (analogy_score, knn_consistency, mean_cosine,
                         retrieval, _norm)
from mct.models import GatedResidualMCT
from mct.train import apply_mct

if __name__ == "__main__":
    device = get_device()
    cka = json.loads((ROOT / "results" / "cka.json").read_text())
    LA, LB = cka["L3"]["layer_a"], cka["L3"]["layer_b"]
    A = load_acts("llama", [LA])[LA]
    B = load_acts("qwen", [LB])[LB]
    st = np.load(ROOT / "results" / "std_stats.npz")
    A = (A - st["mu_a"]) / st["sd_a"]
    B = (B - st["mu_b"]) / st["sd_b"]
    concepts = load_or_build()
    words = [c["text"] for c in concepts]
    langs = np.array([c["lang"] for c in concepts])
    tr_idx, te_idx = train_test_split(len(concepts))

    mct = GatedResidualMCT(A.shape[1], B.shape[1])
    mct.load_state_dict(torch.load(ROOT / "results" / "mct_llama3_qwen3.pt",
                                   weights_only=True))
    pred_all = apply_mct(mct, A, device)
    pred_te = pred_all[te_idx]
    out = {}

    # dil-ayrık retrieval (test kümesi içinde)
    for lang in ("en", "tr"):
        m = langs[te_idx] == lang
        if m.sum() == 0:
            continue
        out[f"retrieval_{lang}"] = {
            **retrieval(pred_te[m], B, te_idx[m]),
            "cosine": mean_cosine(pred_te[m], B[te_idx][m]),
            "n": int(m.sum())}
        print(lang, out[f"retrieval_{lang}"], flush=True)

    # k-NN tutarlılığı (test kümesi, k=10)
    out["knn_jaccard_k10"] = knn_consistency(pred_te, B[te_idx], k=10)
    print("knn:", out["knn_jaccard_k10"], flush=True)

    # analoji: çevrilen vektörlerle, havuz = 10k çevirilmiş uzay (B-uzayı)
    vec_by_word = {w: pred_all[i] for i, w in enumerate(words)}
    out["analogy_translated"] = analogy_score(vec_by_word, words, B)
    # referans: hedef uzayın kendi analoji başarımı
    vec_by_word_b = {w: B[i] for i, w in enumerate(words)}
    out["analogy_native_b"] = analogy_score(vec_by_word_b, words, B)
    print("analogy:", out["analogy_translated"], "native:",
          out["analogy_native_b"], flush=True)

    # niteliksel örnekler
    Q = _norm(B)
    examples = {}
    for w in ("economy", "müzik", "doctor", "bilim", "happy"):
        if w not in vec_by_word:
            continue
        sims = _norm(vec_by_word[w][None]) @ Q.T
        top = np.argsort(-sims[0])[:5]
        examples[w] = [words[i] for i in top]
    out["neighbors"] = examples
    print(json.dumps(examples, ensure_ascii=False, indent=1), flush=True)

    (ROOT / "results" / "semantic.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print("SEMANTIC DONE")
