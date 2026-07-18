"""FAZ 2 / Adım 4 — Nötr ortak uzayda birleştirme (hub-and-spoke ortak eğitim).

4 spoke: llama.L2, llama.L3, qwen.L2, qwen.L3 — her biri kendi encoder ve
decoder-MCT'sini eğitir; nötr uzay (512d) hiçbir modele ait değildir ve
ortak contrastive eğitimle sıfırdan öğrenilir. Birleştirme sonrası her
kaynak->hedef yönü hub üzerinden çalışır; ayrı ikili tercüman gerekmez.

Değerlendirme: tüm yönlü çiftler için hub-üzerinden retrieval; anlamsal
yol (L3->L3) doğrudan ikili MCT (03) ile karşılaştırılır.
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
from mct.metrics import mean_cosine, retrieval
from mct.train import NeutralHub, hub_translate, train_hub

HUB_DIM = 512

if __name__ == "__main__":
    device = get_device()
    cka = json.loads((ROOT / "results" / "cka.json").read_text())
    layers = {"llama_L2": ("llama", cka["L2"]["layer_a"]),
              "llama_L3": ("llama", cka["L3"]["layer_a"]),
              "qwen_L2": ("qwen", cka["L2"]["layer_b"]),
              "qwen_L3": ("qwen", cka["L3"]["layer_b"])}
    acts_l = load_acts("llama", sorted({v[1] for v in layers.values()
                                        if v[0] == "llama"}))
    acts_q = load_acts("qwen", sorted({v[1] for v in layers.values()
                                       if v[0] == "qwen"}))
    spokes = {name: (acts_l if m == "llama" else acts_q)[l]
              for name, (m, l) in layers.items()}
    concepts = load_or_build()
    tr_idx, te_idx = train_test_split(len(concepts))
    from mct.standardize import zapply, zfit
    stats = {k: zfit(v[tr_idx]) for k, v in spokes.items()}
    spokes = {k: zapply(v, *stats[k]) for k, v in spokes.items()}

    hub = NeutralHub({k: v.shape[1] for k, v in spokes.items()},
                     hub_dim=HUB_DIM,
                     mus={k: v[tr_idx].mean(0) for k, v in spokes.items()})
    t0 = time.time()
    log = train_hub(hub, {k: v[tr_idx] for k, v in spokes.items()},
                    device, epochs=80)
    print(f"hub eğitimi: {time.time()-t0:.0f}s, son kayıp {log[-1]:.3f}",
          flush=True)
    torch.save(hub.state_dict(), ROOT / "results" / "neutral_hub.pt")

    results = {"hub_dim": HUB_DIM, "layers": {k: v[1] for k, v in layers.items()},
               "pairs": {}}
    for src in spokes:
        for dst in spokes:
            if src == dst:
                continue
            pred = hub_translate(hub, spokes[src][te_idx], src, dst, device)
            m = {**retrieval(pred, spokes[dst], te_idx),
                 "cosine": mean_cosine(pred, spokes[dst][te_idx])}
            results["pairs"][f"{src}->{dst}"] = m
            print(f"{src}->{dst}: top1={m['top1']:.3f} top5={m['top5']:.3f} "
                  f"cos={m['cosine']:.3f}", flush=True)

    (ROOT / "results" / "hub.json").write_text(json.dumps(results, indent=2))
    print("HUB DONE")
