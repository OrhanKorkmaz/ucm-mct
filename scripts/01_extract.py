"""FAZ 1 / Adım 1 — 10.000 çekirdek konsept için gizli durum çıkarımı.

Her iki modelde TÜM katmanlar için mean-pooled vektörler çıkarılır ve
fp16 .npz olarak saklanır (katman seçimi sonradan CKA ile yapılacak).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mct import get_device
from mct.concepts import load_or_build
from mct.extract import MODELS, extract_all_layers, save_acts

if __name__ == "__main__":
    device = get_device()
    print(f"device: {device}", flush=True)
    concepts = load_or_build()
    texts = [c["text"] for c in concepts]
    print(f"{len(concepts)} konsept "
          f"(EN: {sum(c['lang']=='en' for c in concepts)}, "
          f"TR: {sum(c['lang']=='tr' for c in concepts)})", flush=True)

    for key in MODELS:
        if (Path(__file__).resolve().parent.parent / "data" / f"acts_{key}.npz").exists():
            print(f"[{key}] npz mevcut, atlanıyor", flush=True)
            continue
        t0 = time.time()
        print(f"[{key}] çıkarım başlıyor: {MODELS[key]}", flush=True)
        acts = extract_all_layers(key, texts, device, batch_size=64)
        path = save_acts(key, acts)
        print(f"[{key}] tamam: {len(acts)} katman, {time.time()-t0:.0f}s -> {path}",
              flush=True)
    print("EXTRACTION DONE", flush=True)
