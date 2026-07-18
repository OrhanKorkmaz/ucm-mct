"""H3-revize / Adım 1 — Bağlamsal son-token çıkarımı (çok dilli).

Her konsept KENDİ DİLİNDE bir şablona gömülür (çok-dillilik ana eksen):
  EN: The word 'X' means
  TR: 'X' kelimesinin anlamı
Son token aktivasyonu alınır. Katmanlar: orta bant (~%50, ICML 2501.14082
"zenginleştirilmiş varlık temsilleri") + anlamsal bant (CKA seçimi).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mct import get_device
from mct.concepts import load_or_build
from mct.extract import extract_last_token

TEMPLATES = {"en": "The word '{}' means", "tr": "'{}' kelimesinin anlamı"}
# orta bant (%50) + CKA-seçimli anlamsal bant katmanları
LAYERS = {"llama": [8, 10], "qwen": [12, 15, 18]}

if __name__ == "__main__":
    device = get_device()
    concepts = load_or_build()
    prompts = [TEMPLATES[c["lang"]].format(c["text"]) for c in concepts]
    print(f"{len(prompts)} bağlamsal prompt (dil-eşli şablon)", flush=True)
    for key, layers in LAYERS.items():
        t0 = time.time()
        acts = extract_last_token(key, prompts, device, layers)
        np.savez_compressed(ROOT / "data" / f"acts_ctx_{key}.npz",
                            **{f"layer_{l}": v for l, v in acts.items()})
        print(f"[{key}] katmanlar {layers}: {time.time()-t0:.0f}s", flush=True)
    print("CTX EXTRACTION DONE", flush=True)
