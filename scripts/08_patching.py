"""FAZ 2 / Adım 5b — Activation patching ile işlevsellik testi (H3).

Üç koşul: temiz geçiş / MCT-enjeksiyon / rastgele vektör.
Kaynak: llama.L3 vektörü -> ikili MCT-InfoNCE -> qwen.L3 uzayı; qwen'in
L_qwen3 katmanı çıkışında konsept pozisyonlarına norm-kalibreli enjeksiyon.
H3: KL(MCT ‖ temiz) < 0.2 x KL(rastgele ‖ temiz) ve top-1 uyum > %60.
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
from mct.extract import load_acts, load_model
from mct.models import GatedResidualMCT
from mct.patching import eval_patching
from mct.train import apply_mct

if __name__ == "__main__":
    device = get_device()
    cka = json.loads((ROOT / "results" / "cka.json").read_text())
    LA, LB = cka["L3"]["layer_a"], cka["L3"]["layer_b"]
    A = load_acts("llama", [LA])[LA]
    B_raw = load_acts("qwen", [LB])[LB]
    st = np.load(ROOT / "results" / "std_stats.npz")
    A_std = (A - st["mu_a"]) / st["sd_a"]
    concepts = load_or_build()
    words = [c["text"] for c in concepts]
    tr_idx, te_idx = train_test_split(len(concepts))

    mct = GatedResidualMCT(A.shape[1], B_raw.shape[1])
    mct.load_state_dict(torch.load(ROOT / "results" / "mct_llama3_qwen3.pt",
                                   weights_only=True))
    # MCT standardize uzayda çalışır; enjeksiyon için gerçek uzaya dön
    pred = apply_mct(mct, A_std, device) * st["sd_b"] + st["mu_b"]
    B = B_raw

    tok, model = load_model("qwen", device)
    te_words = [words[i] for i in te_idx]
    t0 = time.time()
    agg, rows = eval_patching(model, tok, device, te_words, pred[te_idx],
                              B[te_idx], LB, template=" {}", max_n=200)
    agg["H3_pass"] = bool(agg["kl_ratio"] < 0.2 and agg["top1_agree_mct"] > 0.6)
    agg["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(agg, indent=2))
    (ROOT / "results" / "patching.json").write_text(
        json.dumps({"aggregate": agg, "rows": rows[:50]}, indent=2))
    print("PATCHING DONE")
