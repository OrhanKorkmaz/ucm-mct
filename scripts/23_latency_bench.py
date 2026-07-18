"""Hız benchmark'ı + hızlandırılmış MCT varyantları.

Soru: vektör kanalı metin kanalından ne kadar hızlı, ve MCT'yi eğitim
aşamasında hangi yöntemlerle hızlandırabiliriz (kalite kaybı ölçülerek)?

Ölçümler:
 A) Metin turu (baseline): qwen 10 token üretir + llama o metni yeniden
    kodlar (çift decode/encode vergisi).
 B) Vektör kanalı: llama zaten hesapladığı L10 durumunu verir -> MCT
    çevirisi -> qwen'e enjeksiyon (tek forward). Ek maliyet = MCT süresi.
 C) MCT varyantları (hepsi aynı veriyle eğitilir, retrieval ile kalite):
    - full:    yarı-ortogonal çekirdek + MLP(1024)   (~4.0M param)
    - lowrank: U·V (r=64) çekirdek + MLP(256)        (~0.5M param)
    - int8:    full'ün dinamik kuantize hali (CPU)
 D) Tel boyutu: fp16 tam vektör vs SAE top-k (indeks,değer) çifti.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mct import get_device
from mct.concepts import load_or_build, train_test_split
from mct.extract import MODELS, load_model
from mct.metrics import mean_cosine, retrieval
from mct.models import GatedResidualMCT, make_mct, procrustes_fit
from mct.standardize import zapply, zfit
from mct.train import apply_mct, train_mct


class LowRankMCT(nn.Module):
    """Hızlandırılmış MCT: düşük-rank çekirdek + küçük MLP."""

    def __init__(self, d_in, d_out, rank=64, hidden=256, mu_in=None, mu_out=None,
                 W0=None, s0=1.0):
        super().__init__()
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_out, bias=False)
        if W0 is not None:  # Procrustes'in rank-r SVD kesmesiyle başlat
            U, S, Vt = np.linalg.svd(W0, full_matrices=False)
            self.up.weight.data = torch.as_tensor(U[:, :rank] * S[:rank])
            self.down.weight.data = torch.as_tensor(Vt[:rank])
        self.log_s = nn.Parameter(torch.tensor(float(np.log(max(s0, 1e-6)))))
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_out))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("mu_in", torch.zeros(d_in) if mu_in is None
                             else torch.as_tensor(mu_in, dtype=torch.float32))
        self.register_buffer("mu_out", torch.zeros(d_out) if mu_out is None
                             else torch.as_tensor(mu_out, dtype=torch.float32))

    def forward(self, x):
        xc = x - self.mu_in
        return (self.mu_out + torch.exp(self.log_s) * self.up(self.down(xc))
                + self.gate * self.mlp(xc))

    def orth_penalty(self):
        W = self.down.weight  # rank-uzayında yumuşak ortogonallik
        G = W @ W.T
        I = torch.eye(G.shape[0], device=W.device)
        return ((G - I) ** 2).sum() / G.shape[0]


def timeit(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000  # ms


if __name__ == "__main__":
    device = get_device()
    cpu = torch.device("cpu")
    concepts = load_or_build()
    tr_i, te_i = train_test_split(len(concepts))
    z = np.load(ROOT / "data" / "acts_cf2.npz")
    A_raw = z["llama_L10"].astype(np.float32)
    B_raw = z["qwen_L18"].astype(np.float32)
    mu_a, sd_a = zfit(A_raw[tr_i]); mu_b, sd_b = zfit(B_raw[tr_i])
    A, B = zapply(A_raw, mu_a, sd_a), zapply(B_raw, mu_b, sd_b)
    out = {"variants": {}}

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    def evaluate(m, dev, tag):
        pred = apply_mct(m, A[te_i], dev)
        met = {**retrieval(pred, B, te_i), "cosine": mean_cosine(pred, B[te_i])}
        x1 = torch.as_tensor(A[:1]).to(dev)
        xb = torch.as_tensor(A[:256]).to(dev)
        with torch.no_grad():
            met["ms_single"] = round(timeit(lambda: m(x1)), 4)
            met["ms_batch256"] = round(timeit(lambda: m(xb)), 3)
        met["params"] = n_params(m)
        out["variants"][tag] = met
        print(f"[{tag}] top1={met['top1']:.3f} cos={met['cosine']:.3f} "
              f"tek-mesaj={met['ms_single']}ms batch256={met['ms_batch256']}ms "
              f"param={met['params']/1e6:.2f}M", flush=True)
        return met

    # full MCT
    full = make_mct(A[tr_i], B[tr_i])
    train_mct(full, A[tr_i], B[tr_i], cpu, epochs=40)
    evaluate(full.to(cpu), cpu, "full_cpu")

    # low-rank MCT
    mu_ac, mu_bc = A[tr_i].mean(0), B[tr_i].mean(0)
    W0, s0 = procrustes_fit(A[tr_i] - mu_ac, B[tr_i] - mu_bc)
    lr_mct = LowRankMCT(A.shape[1], B.shape[1], rank=64, hidden=256,
                        mu_in=mu_ac, mu_out=mu_bc, W0=W0, s0=s0)
    train_mct(lr_mct, A[tr_i], B[tr_i], cpu, epochs=40)
    evaluate(lr_mct.to(cpu), cpu, "lowrank64_cpu")

    # int8 dinamik kuantizasyon (CPU)
    try:
        q = torch.ao.quantization.quantize_dynamic(
            full.to(cpu), {nn.Linear}, dtype=torch.qint8)
        evaluate(q, cpu, "int8_cpu")
    except Exception as e:
        out["variants"]["int8_cpu"] = {"error": str(e)}
        print("int8 hata:", e, flush=True)

    # tel boyutu (D)
    out["wire"] = {
        "fp16_full_vector_bytes": 2 * B.shape[1],
        "hub768_fp16_bytes": 2 * 768,
        "sae_top32_bytes": 32 * 4,   # (uint16 idx + fp16 val)
        "sae_top16_bytes": 16 * 4}

    # metin turu vs vektör kanalı (A/B) — MPS
    tok_q, model_q = load_model("qwen", device)
    prompt = "Word: economy"
    enc = tok_q(prompt, return_tensors="pt").to(device)

    @torch.no_grad()
    def text_generate():
        model_q.generate(**enc, max_new_tokens=10, do_sample=False,
                         pad_token_id=tok_q.pad_token_id)

    out["ms_text_generate10"] = round(timeit(text_generate, n=10, warmup=2), 1)
    del model_q
    if device.type == "mps":
        torch.mps.empty_cache()
    tok_l, model_l = load_model("llama", device)
    enc_l = tok_l("some ten token answer text goes right here now",
                  return_tensors="pt").to(device)

    @torch.no_grad()
    def text_reencode():
        model_l(**enc_l, output_hidden_states=False)

    out["ms_text_reencode"] = round(timeit(text_reencode, n=20, warmup=3), 1)
    out["ms_text_roundtrip"] = out["ms_text_generate10"] + out["ms_text_reencode"]
    out["ms_vector_channel_single"] = out["variants"]["full_cpu"]["ms_single"]
    out["speedup_x"] = round(out["ms_text_roundtrip"]
                             / max(out["ms_vector_channel_single"], 1e-9), 1)
    print(f"metin turu: {out['ms_text_roundtrip']}ms "
          f"(üretim {out['ms_text_generate10']} + yeniden-kodlama "
          f"{out['ms_text_reencode']}) | vektör kanalı: "
          f"{out['ms_vector_channel_single']}ms | hızlanma: {out['speedup_x']}x",
          flush=True)
    (ROOT / "results" / "latency.json").write_text(json.dumps(out, indent=2))
    print("LATENCY DONE")
