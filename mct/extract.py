"""LLM gizli durum çıkarımı.

Her konsept için model tüm katmanlarda çalıştırılır; konseptin kendi
token'ları üzerinden (BOS hariç) mean-pooling ile katman başına tek vektör
elde edilir. Çıktılar fp16 .npz olarak katman-katman saklanır.

Tokenizer uyumsuzluğu (Llama ≠ Qwen token sınırları) bu havuzlama ile
çözülür: iletişim birimi token değil, konsept vektörüdür.
"""
import numpy as np
import torch
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

MODELS = {
    "llama": "unsloth/Llama-3.2-1B",   # meta-llama/Llama-3.2-1B birebir aynası (gated değil)
    "qwen": "Qwen/Qwen2.5-0.5B",
}


def load_model(key, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Sol-padding (unsloth/Llama varsayılanı) causal dikkatte pad satırlarının
    # NaN'ını sonraki katmanlarda gerçek pozisyonlara bulaştırır
    # (0 ağırlık × NaN değer = NaN). Sağ-padding'de pad'ler hep gelecektedir.
    tok.padding_side = "right"
    # bfloat16: Llama-3.2 fp16'da attention taşmasıyla NaN üretiyor (ölçüldü:
    # 10k konseptin ~8.5k'sında NaN); her iki model de bf16 ile eğitilmiş.
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    model.to(device).eval()
    return tok, model


@torch.no_grad()
def extract_last_token(key, prompts, device, layers, batch_size=64):
    """Bağlam şablonlu prompt'ların SON token aktivasyonları (seçili katmanlar).

    H3-revize (ICML 2501.14082 protokolü): iletişim birimi, bağlam-içi
    son-token temsilidir; tek pozisyon enjeksiyonuyla eşleşir.
    """
    tok, model = load_model(key, device)
    d = model.config.hidden_size
    N = len(prompts)
    out = {l: np.zeros((N, d), dtype=np.float16) for l in layers}
    for start in range(0, N, batch_size):
        batch = prompts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=True).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        last = enc["attention_mask"].sum(1) - 1  # sağ-padding'de son gerçek token
        rows = torch.arange(len(batch), device=device)
        for l in layers:
            vec = hs[l][rows, last]
            out[l][start:start + len(batch)] = vec.float().cpu().numpy().astype(np.float16)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return out


@torch.no_grad()
def extract_all_layers(key, texts, device, batch_size=64, template=" {}"):
    """texts listesi için tüm katmanlarda mean-pooled vektörler.

    Dönüş: dict {layer_idx: np.float16 [N, d]}  (0 = embedding çıkışı)
    """
    tok, model = load_model(key, device)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    N = len(texts)
    out = {l: np.zeros((N, d), dtype=np.float16) for l in range(n_layers + 1)}

    for start in range(0, N, batch_size):
        batch = [template.format(t) for t in texts[start:start + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=True).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].clone()
        # BOS'u havuzlamadan çıkar (varsa)
        if tok.bos_token_id is not None:
            mask[enc["input_ids"] == tok.bos_token_id] = 0
        # NOT: pad pozisyonlarında attention satırı tümüyle maskeli olduğundan
        # hidden state NaN olabilir; h*mask (NaN*0=NaN) yerine where kullan.
        mask = mask.unsqueeze(-1).bool()
        denom = mask.sum(1).to(hs[0].dtype).clamp(min=1)
        for l, h in enumerate(hs):
            pooled = torch.where(mask, h, torch.zeros_like(h)).sum(1) / denom
            out[l][start:start + len(batch)] = pooled.float().cpu().numpy().astype(np.float16)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return out


def save_acts(key, acts, path=None):
    path = Path(path) if path else DATA / f"acts_{key}.npz"
    np.savez_compressed(path, **{f"layer_{l}": v for l, v in acts.items()})
    return path


def load_acts(key, layers=None, path=None):
    path = Path(path) if path else DATA / f"acts_{key}.npz"
    z = np.load(path)
    all_layers = sorted(int(k.split("_")[1]) for k in z.files)
    layers = layers if layers is not None else all_layers
    return {l: z[f"layer_{l}"].astype(np.float32) for l in layers}
