"""Çekirdek konsept seti üretimi.

10.000 konsept = 8.000 EN + 2.000 TR, wordfreq frekans listelerinden.
Tek-token/çok-token ayrımı yapılmaz; çıkarım aşamasında kelimenin tüm
token'ları üzerinden mean-pooling uygulanır.
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

_WORD_RE = re.compile(r"^[a-zçğıöşüâîû]+$", re.IGNORECASE)


def build_concepts(n_en=8000, n_tr=2000, seed=42):
    from wordfreq import top_n_list

    en = [w for w in top_n_list("en", n_en * 2)
          if _WORD_RE.match(w) and len(w) >= 2][:n_en]
    tr_seen = set(en)
    tr = []
    for w in top_n_list("tr", n_tr * 4):
        if _WORD_RE.match(w) and len(w) >= 2 and w not in tr_seen:
            tr.append(w)
            tr_seen.add(w)
        if len(tr) >= n_tr:
            break
    concepts = [{"text": w, "lang": "en"} for w in en] + \
               [{"text": w, "lang": "tr"} for w in tr]
    return concepts


def load_or_build(path=None, **kw):
    path = Path(path) if path else DATA / "concepts.json"
    if path.exists():
        return json.loads(path.read_text())
    concepts = build_concepts(**kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(concepts, ensure_ascii=False))
    return concepts


def train_test_split(n_total, n_test=2000, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    return idx[n_test:], idx[:n_test]
