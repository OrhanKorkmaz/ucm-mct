"""H5 final denemesi: hub_dim 768 + çapraz-çeviri çapası lam_anchor=1.0.

Gerekçe: InfoNCE ayrıştırmayı keskinleştirirken kosinüsü düşürür (FAZ 1'de
ölçülen ödünleşim). H5 metriği kosinüs olduğundan çapraz-çeviri kosinüs
kaybının ağırlığı yükseltilir; hub kapasitesi 512→768.
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
from mct.metrics import mean_cosine, retrieval
from mct.standardize import zapply, zfit
from mct.train import NeutralHub, hub_translate, train_hub

if __name__ == "__main__":
    cpu = torch.device("cpu")
    concepts = load_or_build()
    langs = np.array([c["lang"] for c in concepts])
    tr_i, te_i = train_test_split(len(concepts))
    z = np.load(ROOT / "data" / "acts_cf2.npz")
    spokes = {k: z[k].astype(np.float32) for k in z.files if k != "anchor"}
    spokes = {k: zapply(v, *zfit(v[tr_i])) for k, v in spokes.items()}
    cross = [("llama_L8", "qwen_L12"), ("llama_L8", "qwen_L18"),
             ("llama_L10", "qwen_L18"), ("qwen_L12", "llama_L10")]

    hub = NeutralHub({k: v.shape[1] for k, v in spokes.items()}, hub_dim=768)
    t0 = time.time()
    train_hub(hub, {k: v[tr_i] for k, v in spokes.items()}, cpu, epochs=60,
              lam_anchor=1.0, lam_cycle=0.25, cycle_pairs=cross)
    print(f"eğitim {time.time()-t0:.0f}s", flush=True)

    pred = hub_translate(hub, spokes["llama_L8"][te_i], "llama_L8", "qwen_L18", cpu)
    out = {"H5_cos": mean_cosine(pred, spokes["qwen_L18"][te_i])}
    out["H5_pass"] = bool(out["H5_cos"] >= 0.75)
    for lg in ("en", "tr"):
        m = langs[te_i] == lg
        out[f"retrieval_{lg}"] = retrieval(pred[m], spokes["qwen_L18"], te_i[m])
    print(f"H5 cos={out['H5_cos']:.3f} ({'GEÇTİ' if out['H5_pass'] else 'kaldı'}) "
          f"EN top1={out['retrieval_en']['top1']:.3f} "
          f"TR top1={out['retrieval_tr']['top1']:.3f}", flush=True)
    torch.save(hub.state_dict(), ROOT / "results" / "hub_final768.pt")
    (ROOT / "results" / "h5_final.json").write_text(json.dumps(out, indent=2))
    print("H5-FINAL DONE")
