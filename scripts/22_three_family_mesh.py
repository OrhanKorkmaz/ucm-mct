"""Üç aileli mesh — "sadece Llama ile sürdürmeyeceğiz" testi.

3. aile: HuggingFaceTB/SmolLM2-360M (bağımsız eğitim verisi, 32 katman,
d=960) — Llama/Qwen'den bağımsız üçüncü kütüphane.

Ölçülen engeller (ileriyi ön alma):
  E1 Kapasite/karışım: 6 spoke'lu hub'da llama→qwen yolunun kalitesi,
     4 spoke'lu hub'a (top-1 %72.5 / H5 0.778 referans) göre geriliyor mu?
     (joint eğitimde interferans engeli)
  E2 Yeni üye maliyeti: mevcut protokole yeni model 2 tercümanla katılıyor
     mu — tüm yönlerde çalışan çeviri (any-to-any 6 çapraz yol).
  E3 Derinlik çevirisi 3. ailede: smol.L16 -> hub -> qwen.L18 vb.
  E4 Dil: TR/EN ayrık (göreli sadakat bulgusunun 3. modelde tekrarı).
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
from mct import extract as ex
from mct.metrics import mean_cosine, retrieval
from mct.standardize import zapply, zfit
from mct.train import NeutralHub, hub_translate, train_hub

ex.MODELS["smol"] = "HuggingFaceTB/SmolLM2-360M"
TEMPLATES = {"en": "Word: {}", "tr": "Kelime: {}"}
SMOL_LAYERS = [16, 22]  # ~%50 ve ~%69 derinlik
HUB_DIM = 768

if __name__ == "__main__":
    device = get_device()
    cpu = torch.device("cpu")
    concepts = load_or_build()
    langs = np.array([c["lang"] for c in concepts])
    prompts = [TEMPLATES[c["lang"]].format(c["text"]) for c in concepts]
    tr_i, te_i = train_test_split(len(concepts))

    z = np.load(ROOT / "data" / "acts_cf2.npz")
    raw = {k: z[k].astype(np.float32) for k in z.files if k != "anchor"}
    smol_path = ROOT / "data" / "acts_smol.npz"
    if smol_path.exists():
        zs = np.load(smol_path)
        for k in zs.files:
            raw[k] = zs[k].astype(np.float32)
    else:
        t0 = time.time()
        acts = ex.extract_last_token("smol", prompts, device, SMOL_LAYERS)
        np.savez_compressed(smol_path, **{f"smol_L{l}": v.astype(np.float16)
                                          for l, v in acts.items()})
        for l, v in acts.items():
            raw[f"smol_L{l}"] = v.astype(np.float32)
        print(f"[smol] katmanlar {SMOL_LAYERS}: {time.time()-t0:.0f}s", flush=True)

    spokes = {k: zapply(v, *zfit(v[tr_i])) for k, v in raw.items()}
    print("spokes:", {k: v.shape[1] for k, v in spokes.items()}, flush=True)

    cross = [("llama_L8", "qwen_L12"), ("llama_L8", "qwen_L18"),
             ("llama_L10", "qwen_L18"), ("qwen_L12", "llama_L10"),
             ("smol_L16", "qwen_L18"), ("smol_L16", "llama_L10"),
             ("llama_L8", "smol_L22"), ("qwen_L12", "smol_L22")]
    hub = NeutralHub({k: v.shape[1] for k, v in spokes.items()}, hub_dim=HUB_DIM)
    t0 = time.time()
    train_hub(hub, {k: v[tr_i] for k, v in spokes.items()}, cpu, epochs=50,
              lam_anchor=1.0, lam_cycle=0.25, cycle_pairs=cross)
    print(f"hub eğitimi (6 spoke): {time.time()-t0:.0f}s", flush=True)
    torch.save(hub.state_dict(), ROOT / "results" / "hub_3family.pt")

    out = {"spokes": {k: int(v.shape[1]) for k, v in spokes.items()}}
    routes = [("llama_L10", "qwen_L18"), ("qwen_L18", "llama_L10"),
              ("smol_L22", "qwen_L18"), ("qwen_L18", "smol_L22"),
              ("smol_L22", "llama_L10"), ("llama_L10", "smol_L22"),
              ("llama_L8", "qwen_L18"), ("smol_L16", "qwen_L18"),
              ("smol_L16", "llama_L10")]
    for src, dst in routes:
        pred = hub_translate(hub, spokes[src][te_i], src, dst, cpu)
        m = {"cosine": mean_cosine(pred, spokes[dst][te_i]),
             **retrieval(pred, spokes[dst], te_i)}
        out[f"{src}->{dst}"] = m
        print(f"{src}->{dst}: cos={m['cosine']:.3f} top1={m['top1']:.3f}",
              flush=True)

    # dil-ayrık (E4): smol->qwen anlamsal yol
    pred = hub_translate(hub, spokes["smol_L22"][te_i], "smol_L22", "qwen_L18", cpu)
    for lg in ("en", "tr"):
        m = langs[te_i] == lg
        out[f"smol->qwen_{lg}"] = retrieval(pred[m], spokes["qwen_L18"], te_i[m])
        print(f"smol->qwen [{lg}]: top1={out[f'smol->qwen_{lg}']['top1']:.3f}",
              flush=True)

    # E1 kıyas: 2-aile referansları
    out["ref_2family"] = {"llama_L10->qwen_L18_top1": 0.725,
                          "H5_llama_L8->qwen_L18_cos": 0.778}
    (ROOT / "results" / "three_family.json").write_text(json.dumps(out, indent=2))
    print("3FAMILY DONE")
