"""H5-revize + çok dilli çapa spoke.

İki hub karşılaştırılır (temel amaç: çok dilli çeviri yeteneği):
  hub-A: 4 spoke (llama.L8, llama.L10, qwen.L12, qwen.L18) — konsept-sonlu
         bağlamsal aktivasyonlar + cycle-consistency (vec2vec ilkesi)
  hub-B: aynı 4 spoke + ÇAPA: paraphrase-multilingual-MiniLM-L12-v2
         (50+ dil, TR dahil) kelime embedding'leri 5. spoke olarak.

Ölçümler:
  H5: llama.L8 -> hub -> qwen.L18 kosinüs (hedef >= 0.75)
  Çok dillilik: llama.L8 -> qwen.L18 retrieval TR/EN ayrık (çapa etkisi)
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
from mct.metrics import mean_cosine, retrieval
from mct.standardize import zapply, zfit
from mct.train import NeutralHub, hub_translate, train_hub

TEMPLATES = {"en": "Word: {}", "tr": "Kelime: {}"}
ANCHOR = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LAYERS = {"llama": [8, 10], "qwen": [12, 18]}


@torch.no_grad()
def anchor_embed(texts, device, batch_size=128):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ANCHOR)
    model = AutoModel.from_pretrained(ANCHOR, dtype=torch.float16).to(device).eval()
    out = []
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True,
                  truncation=True).to(device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return np.concatenate(out).astype(np.float32)


def eval_hub(hub, spokes, te_i, langs, device, tag):
    res = {}
    pred = hub_translate(hub, spokes["llama_L8"][te_i], "llama_L8", "qwen_L18",
                         device)
    res["H5_cos"] = mean_cosine(pred, spokes["qwen_L18"][te_i])
    res["H5_pass"] = bool(res["H5_cos"] >= 0.75)
    for lg in ("en", "tr"):
        m = langs[te_i] == lg
        res[f"retrieval_{lg}"] = {
            **retrieval(pred[m], spokes["qwen_L18"], te_i[m]),
            "cosine": mean_cosine(pred[m], spokes["qwen_L18"][te_i][m]),
            "n": int(m.sum())}
    print(f"[{tag}] H5 cos={res['H5_cos']:.3f} "
          f"({'GEÇTİ' if res['H5_pass'] else 'kaldı'}) | "
          f"EN top1={res['retrieval_en']['top1']:.3f} "
          f"TR top1={res['retrieval_tr']['top1']:.3f}", flush=True)
    return res


if __name__ == "__main__":
    device = get_device()
    cpu = torch.device("cpu")
    concepts = load_or_build()
    words = [c["text"] for c in concepts]
    langs = np.array([c["lang"] for c in concepts])
    prompts = [TEMPLATES[c["lang"]].format(c["text"]) for c in concepts]
    tr_i, te_i = train_test_split(len(concepts))

    # bağlamsal aktivasyonlar (konsept-sonlu şablon)
    path = ROOT / "data" / "acts_cf2.npz"
    if path.exists():
        z = np.load(path)
        raw = {k: z[k].astype(np.float32) for k in z.files}
    else:
        raw = {}
        for key, ls in LAYERS.items():
            t0 = time.time()
            acts = extract_last_token(key, prompts, device, ls)
            for l, v in acts.items():
                raw[f"{key}_L{l}"] = v.astype(np.float32)
            print(f"[{key}] ctx katmanlar {ls}: {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        raw["anchor"] = anchor_embed(words, device)
        print(f"[anchor] {ANCHOR.split('/')[-1]}: {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(path, **{k: v.astype(np.float16)
                                     for k, v in raw.items()})

    spokes_all = {k: zapply(v, *zfit(v[tr_i])) for k, v in raw.items()}
    out = {}
    cross = [("llama_L8", "qwen_L12"), ("llama_L8", "qwen_L18"),
             ("llama_L10", "qwen_L18"), ("qwen_L12", "llama_L10")]

    for tag, names in [("hub_4spoke", [k for k in spokes_all if k != "anchor"]),
                       ("hub_anchor", list(spokes_all))]:
        spokes = {k: spokes_all[k] for k in names}
        cyc = cross + ([("llama_L10", "anchor"), ("qwen_L18", "anchor")]
                       if "anchor" in names else [])
        hub = NeutralHub({k: v.shape[1] for k, v in spokes.items()}, hub_dim=512)
        t0 = time.time()
        train_hub(hub, {k: v[tr_i] for k, v in spokes.items()}, cpu,
                  epochs=50, lam_cycle=0.25, cycle_pairs=cyc)
        print(f"[{tag}] eğitim {time.time()-t0:.0f}s", flush=True)
        out[tag] = eval_hub(hub, spokes_all, te_i, langs, cpu, tag)
        torch.save(hub.state_dict(), ROOT / "results" / f"{tag}.pt")

    (ROOT / "results" / "h5_multilingual.json").write_text(
        json.dumps(out, indent=2))
    print("H5-ML DONE")
