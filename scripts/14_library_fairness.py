"""Kütüphaneler-arası tercüme sadakati testi (kavram netleştirmesi).

Soru: TR'deki düşük skorlar TERCÜMANIN mı, yoksa modellerin kendi iç
kütüphanelerinin (temsil kalitesinin) mi eseri?

Yöntem — üç ölçüm, dil başına:
 1. Kaynak öz-tutarlılık: llama.L8, aynı konsepti iki farklı şablonla
    kodladığında kendini bulabiliyor mu? (şablon-A havuzu, şablon-B sorgu)
    Bu, kaynak kütüphanenin dil-bazlı tavanıdır (tercüman yok!).
 2. Hedef öz-tutarlılık: qwen.L12 için aynısı.
 3. Modeller-arası MCT çevirisi: llama.L8(şablon-A) -> qwen.L12(şablon-A).

Göreli sadakat = çeviri top-1 / min(öz-tutarlılık tavanları).
EN ve TR göreli sadakat birbirine yakınsa tercüman dil-adildir;
açık modellerin kütüphanesindedir (dil çıktısı en sonda üretilir).
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
from mct.extract import extract_last_token
from mct.metrics import retrieval
from mct.models import make_mct
from mct.standardize import zapply, zfit

TEMPL_A = {"en": "Word: {}", "tr": "Kelime: {}"}
TEMPL_B = {"en": "The term: {}", "tr": "Terim: {}"}
LA, LB = 8, 12

sys.path.insert(0, str(ROOT / "scripts"))


def get_acts(tag, templates, device):
    path = ROOT / "data" / f"acts_fair_{tag}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k].astype(np.float32) for k in z.files}
    concepts = load_or_build()
    prompts = [templates[c["lang"]].format(c["text"]) for c in concepts]
    out = {}
    for key, layer in (("llama", LA), ("qwen", LB)):
        t0 = time.time()
        acts = extract_last_token(key, prompts, device, [layer])
        out[key] = acts[layer].astype(np.float32)
        print(f"[{tag}/{key}] {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(path, **{k: v.astype(np.float16) for k, v in out.items()})
    return out


if __name__ == "__main__":
    device = get_device()
    cpu = torch.device("cpu")
    concepts = load_or_build()
    langs = np.array([c["lang"] for c in concepts])
    tr_i, te_i = train_test_split(len(concepts))

    A = get_acts("A", TEMPL_A, device)   # şablon-A: her iki model
    B = get_acts("B", TEMPL_B, device)   # şablon-B: her iki model

    out = {}
    # 1-2) öz-tutarlılık (tercümansız, model-içi): şablon-B sorgu, şablon-A havuz
    for key in ("llama", "qwen"):
        XA = zapply(A[key], *zfit(A[key][tr_i]))
        XB = zapply(B[key], *zfit(B[key][tr_i]))
        for lg in ("en", "tr"):
            m = langs[te_i] == lg
            out[f"self_{key}_{lg}"] = retrieval(XB[te_i][m], XA, te_i[m])
            print(f"öz-tutarlılık {key}/{lg}: "
                  f"top1={out[f'self_{key}_{lg}']['top1']:.3f}", flush=True)

    # 3) modeller-arası MCT (şablon-A -> şablon-A), dil-dengeli eğitim
    from importlib import import_module
    h3 = import_module("11_h3_revised")
    XA_l = zapply(A["llama"], *zfit(A["llama"][tr_i]))
    XA_q = zapply(A["qwen"], *zfit(A["qwen"][tr_i]))
    en_tr = tr_i[langs[tr_i] == "en"]; tr_tr = tr_i[langs[tr_i] == "tr"]
    mct = make_mct(XA_l[tr_i], XA_q[tr_i])
    h3.train_balanced(mct, XA_l, XA_q, en_tr, tr_tr, cpu)
    from mct.train import apply_mct
    pred = apply_mct(mct, XA_l, cpu)
    for lg in ("en", "tr"):
        m = langs[te_i] == lg
        out[f"cross_{lg}"] = retrieval(pred[te_i][m], XA_q, te_i[m])
        print(f"çeviri llama->qwen/{lg}: top1={out[f'cross_{lg}']['top1']:.3f}",
              flush=True)

    # göreli sadakat
    for lg in ("en", "tr"):
        ceil = min(out[f"self_llama_{lg}"]["top1"], out[f"self_qwen_{lg}"]["top1"])
        out[f"fidelity_{lg}"] = {
            "ceiling_top1": ceil,
            "cross_top1": out[f"cross_{lg}"]["top1"],
            "relative": out[f"cross_{lg}"]["top1"] / max(ceil, 1e-9)}
        print(f"göreli sadakat {lg}: çeviri {out[f'cross_{lg}']['top1']:.3f} / "
              f"tavan {ceil:.3f} = {out[f'fidelity_{lg}']['relative']:.3f}",
              flush=True)

    (ROOT / "results" / "library_fairness.json").write_text(
        json.dumps(out, indent=2))
    print("FAIRNESS DONE")
