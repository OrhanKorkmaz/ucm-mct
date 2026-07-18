"""FAZ 2 — SAE köprüsü: nötr uzay üzerinde Top-K SAE sıkıştırma taraması (H4).

İletim hattı: llama.L3 -> encoder -> hub(512d) -> [SAE kodla/çöz] ->
decoder -> qwen.L3 retrieval.

Sıkıştırma oranı = hub_dim / (2k)  (fp16 değer + uint16 indeks / özellik).
Tarama: sözlük F ∈ {128..4096} (0.25x..8x), k ∈ {8,16,32,64}.
H4: >= 8x sıkıştırmada top-1 bozulması <= 5 puan.
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
from mct.sae import TopKSAE, sae_roundtrip, train_sae
from mct.train import NeutralHub, hub_encode

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
    hub.to(device).eval()

    # hub temsilleri (SAE eğitimi tüm spoke'ların hub izdüşümü üzerinde)
    Z_tr = np.concatenate([hub_encode(hub, spokes[k][tr_idx], k, device)
                           for k in spokes])
    Z_src_te = hub_encode(hub, spokes["llama_L3"][te_idx], "llama_L3", device)

    @torch.no_grad()
    def decode_eval(Z, tag):
        z = torch.as_tensor(Z).to(device)
        pred = hub.decoders["qwen_L3"](z).cpu().numpy()
        m = {**retrieval(pred, spokes["qwen_L3"], te_idx),
             "cosine": mean_cosine(pred, spokes["qwen_L3"][te_idx])}
        print(f"  {tag}: top1={m['top1']:.3f} top5={m['top5']:.3f} "
              f"cos={m['cosine']:.3f}", flush=True)
        return m

    print("Referans (SAE'siz):", flush=True)
    ref = decode_eval(Z_src_te, "hub-direct")
    results = {"reference_no_sae": ref, "sweep": []}

    for F in (128, 256, 512, 1024, 2048, 4096):
        for k in (8, 16, 32, 64):
            if k >= F:
                continue
            t0 = time.time()
            sae = TopKSAE(HUB_DIM, F, k)
            train_sae(sae, Z_tr, device, epochs=30)
            rec, active = sae_roundtrip(sae, Z_src_te, device)
            m = decode_eval(rec, f"F={F} k={k}")
            recon_cos = mean_cosine(rec, Z_src_te)
            results["sweep"].append({
                "F": F, "k": k, "ratio_x": HUB_DIM / (2 * k),
                "recon_cosine": recon_cos, "active_mean": active,
                **m, "train_s": round(time.time() - t0, 1)})
            print(f"    F={F} k={k}: oran={HUB_DIM/(2*k):.0f}x "
                  f"recon_cos={recon_cos:.3f}", flush=True)

    # H4: >= 8x sıkıştırmada en iyi konfig
    cands = [r for r in results["sweep"] if r["ratio_x"] >= 8]
    best = max(cands, key=lambda r: r["top1"]) if cands else None
    if best:
        results["H4"] = {"best": best,
                         "top1_drop_pts": round(100 * (ref["top1"] - best["top1"]), 1),
                         "pass": bool(ref["top1"] - best["top1"] <= 0.05)}
        print(f"H4: {best['F']}F/k{best['k']} @ {best['ratio_x']:.0f}x, "
              f"düşüş {results['H4']['top1_drop_pts']} puan -> "
              f"{'GEÇTİ' if results['H4']['pass'] else 'KALDI'}")
    (ROOT / "results" / "sae.json").write_text(json.dumps(results, indent=2))
    print("SAE DONE")
