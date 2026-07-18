"""FAZ 2 / Adım 5a — Katman-2 -> nötr uzay -> Katman-3 derinlik çevirisi (H5).

Model-A'nın işlem-süreci (L2) vektörü hub üzerinden Model-B'nin anlamsal
(L3) uzayına çevrilir; hedef modelin KENDİ L3 temsiliyle kosinüs benzerliği
ölçülür. H5 eşiği: >= 0.75.

Referans çizgileri:
  - model-içi derinlik çevirisi (llama.L2 -> hub -> llama.L3)
  - anlamsal-katman çevirisi (llama.L3 -> hub -> qwen.L3)
  - alt sınır: L2'yi L3'e çeviri YAPMADAN kıyas (ortak boyut yok ->
    Procrustes-lineer taban çizgisi ile).
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
from mct.metrics import mean_cosine, retrieval
from mct.models import ProcrustesBaseline
from mct.train import NeutralHub, hub_translate

HUB_DIM = 512

if __name__ == "__main__":
    device = get_device()
    cka = json.loads((ROOT / "results" / "cka.json").read_text())
    L = {"llama_L2": cka["L2"]["layer_a"], "llama_L3": cka["L3"]["layer_a"],
         "qwen_L2": cka["L2"]["layer_b"], "qwen_L3": cka["L3"]["layer_b"]}
    acts_l = load_acts("llama", sorted({L["llama_L2"], L["llama_L3"]}))
    acts_q = load_acts("qwen", sorted({L["qwen_L2"], L["qwen_L3"]}))
    spokes = {"llama_L2": acts_l[L["llama_L2"]], "llama_L3": acts_l[L["llama_L3"]],
              "qwen_L2": acts_q[L["qwen_L2"]], "qwen_L3": acts_q[L["qwen_L3"]]}
    concepts = load_or_build()
    tr_idx, te_idx = train_test_split(len(concepts))
    from mct.standardize import zapply, zfit
    spokes = {k: zapply(v, *zfit(v[tr_idx])) for k, v in spokes.items()}

    hub = NeutralHub({k: v.shape[1] for k, v in spokes.items()}, hub_dim=HUB_DIM)
    hub.load_state_dict(torch.load(ROOT / "results" / "neutral_hub.pt",
                                   weights_only=True))

    out = {}
    routes = [("llama_L2", "qwen_L3"), ("qwen_L2", "llama_L3"),
              ("llama_L2", "llama_L3"), ("qwen_L2", "qwen_L3"),
              ("llama_L3", "qwen_L3")]
    for src, dst in routes:
        pred = hub_translate(hub, spokes[src][te_idx], src, dst, device)
        m = {"cosine": mean_cosine(pred, spokes[dst][te_idx]),
             **retrieval(pred, spokes[dst], te_idx)}
        out[f"{src}->{dst}"] = m
        print(f"{src}->hub->{dst}: cos={m['cosine']:.3f} "
              f"top1={m['top1']:.3f}", flush=True)

    # taban çizgisi: doğrudan Procrustes L2->L3 (hub'sız, ikili)
    for src, dst in [("llama_L2", "qwen_L3"), ("llama_L2", "llama_L3")]:
        pb = ProcrustesBaseline(spokes[src][tr_idx], spokes[dst][tr_idx])
        pred = pb(spokes[src][te_idx])
        out[f"baseline_procrustes_{src}->{dst}"] = {
            "cosine": mean_cosine(pred, spokes[dst][te_idx]),
            **retrieval(pred, spokes[dst], te_idx)}
        print(f"[baseline] {src}->{dst}: "
              f"cos={out[f'baseline_procrustes_{src}->{dst}']['cosine']:.3f}",
              flush=True)

    h5 = out["llama_L2->qwen_L3"]["cosine"]
    out["H5_pass"] = bool(h5 >= 0.75)
    print(f"H5 (llama.L2 -> qwen.L3 kosinüs) = {h5:.3f} -> "
          f"{'GEÇTİ' if h5 >= 0.75 else 'KALDI'}")
    (ROOT / "results" / "depth.json").write_text(json.dumps(out, indent=2))
    print("DEPTH DONE")
