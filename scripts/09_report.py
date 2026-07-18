"""Tüm sonuçları toplayıp results/REPORT.md + grafikler üretir."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def j(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_cka(cka):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, band in zip(axes, ("L2", "L3")):
        M = np.array(cka[band]["matrix"])
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(cka[band]["layers_b"])),
                      cka[band]["layers_b"])
        ax.set_yticks(range(len(cka[band]["layers_a"])),
                      cka[band]["layers_a"])
        ax.set_xlabel("Qwen katmanı"); ax.set_ylabel("Llama katmanı")
        ax.set_title(f"{band} bandı CKA (seçim: "
                     f"L{cka[band]['layer_a']}-L{cka[band]['layer_b']}, "
                     f"{cka[band]['cka']:.2f})")
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(R / "fig_cka.png", dpi=130)


def fig_ladder(lad):
    ns = sorted(int(k) for k in lad["ladder"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for meth, style in [("procrustes", "--o"), ("mct_infonce", "-o")]:
        ax.plot(ns, [100 * lad["ladder"][str(n)][meth]["top1"] for n in ns],
                style, label=f"{meth} top-1")
        ax.plot(ns, [100 * lad["ladder"][str(n)][meth]["top5"] for n in ns],
                style.replace("o", "s"), alpha=0.5, label=f"{meth} top-5")
    ax.set_xscale("log"); ax.set_xticks(ns, ns)
    ax.set_xlabel("Eğitim konsepti sayısı"); ax.set_ylabel("%")
    ax.set_title("Mikro ölçek merdiveni (llama.L3 → qwen.L3, test=2000)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(R / "fig_ladder.png", dpi=130)


def fig_sae(sae):
    sw = sae["sweep"]
    ref = sae["reference_no_sae"]["top1"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for F in sorted({r["F"] for r in sw}):
        rows = sorted([r for r in sw if r["F"] == F], key=lambda r: r["ratio_x"])
        ax.plot([r["ratio_x"] for r in rows], [100 * r["top1"] for r in rows],
                "-o", label=f"F={F}")
    ax.axhline(100 * ref, color="crimson", ls="--", lw=1,
               label="SAE'siz referans")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sıkıştırma oranı (hub_dim / 2k)")
    ax.set_ylabel("E2E Top-1 (%)")
    ax.set_title("SAE köprüsü: sıkıştırma ↔ iletim kalitesi")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(R / "fig_sae.png", dpi=130)


def table(rows, headers):
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(x) for x in r) + " |\n"
    return out


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    cka, lad = j("cka.json"), j("ladder.json")
    sem, hub = j("semantic.json"), j("hub.json")
    dep, sae, pat = j("depth.json"), j("sae.json"), j("patching.json")

    if cka: fig_cka(cka)
    if lad: fig_ladder(lad)
    if sae: fig_sae(sae)

    md = ["# Unified Cognitive Mesh v2 — FAZ 1-2 Deney Raporu",
          "",
          "Modeller: **Llama-3.2-1B** (16 katman, d=2048) ↔ **Qwen2.5-0.5B** "
          "(24 katman, d=896); 10.000 konsept (8.000 EN + 2.000 TR, wordfreq "
          "frekans sıralı); eğitim/test = 8.000/2.000; havuz = 10.000 (zor koşul). "
          "Donanım: Apple Silicon MPS, fp16 çıkarım.", ""]

    if cka:
        md += ["## 1. CKA katman taraması", "",
               f"- **Katman-2 (işlem-süreci, %30-50):** llama L{cka['L2']['layer_a']} ↔ "
               f"qwen L{cka['L2']['layer_b']} (CKA={cka['L2']['cka']:.3f})",
               f"- **Katman-3 (anlamsal, %60-75):** llama L{cka['L3']['layer_a']} ↔ "
               f"qwen L{cka['L3']['layer_b']} (CKA={cka['L3']['cka']:.3f})",
               f"- Model-içi L2→L3 CKA: llama {cka['llama_intra_L2L3_cka']:.3f}, "
               f"qwen {cka['qwen_intra_L2L3_cka']:.3f}",
               "", "![CKA](fig_cka.png)", ""]

    if lad:
        md += ["## 2. MCT yöntem karşılaştırması (n_train=8000, llama.L3→qwen.L3)", ""]
        rows = [[m, f"{v['cosine']:.3f}", pct(v['top1']), pct(v['top5']),
                 pct(v['top20']), f"{v['geometry_rho']:.3f}"]
                for m, v in lad["methods"].items()]
        md += [table(rows, ["Yöntem", "Cosine", "Top-1", "Top-5", "Top-20",
                            "Geometri ρ"]), ""]
        md += ["### Mikro ölçek merdiveni", ""]
        rows = []
        for n in sorted(lad["ladder"], key=int):
            for m in ("procrustes", "mct_infonce"):
                v = lad["ladder"][n][m]
                rows.append([n, m, f"{v['cosine']:.3f}", pct(v['top1']),
                             pct(v['top5']), f"{v['geometry_rho']:.3f}"])
        md += [table(rows, ["n_train", "Yöntem", "Cosine", "Top-1", "Top-5",
                            "Geometri ρ"]),
               "", "![Merdiven](fig_ladder.png)", ""]

    if sem:
        md += ["## 3. Anlamsal uzay denemeleri", ""]
        for lang in ("en", "tr"):
            k = f"retrieval_{lang}"
            if k in sem:
                v = sem[k]
                md += [f"- **{lang.upper()}** (n={v['n']}): top-1 {pct(v['top1'])}, "
                       f"top-5 {pct(v['top5'])}, cos {v['cosine']:.3f}"]
        md += [f"- k-NN tutarlılığı (Jaccard@10): {sem['knn_jaccard_k10']:.3f}",
               f"- Analoji (çevrilmiş): top-1 "
               f"{pct(sem['analogy_translated']['analogy_top1'])}, top-5 "
               f"{pct(sem['analogy_translated']['analogy_top5'])} "
               f"(n={sem['analogy_translated']['n_analogies']}); yerli B-uzayı "
               f"referansı top-1 {pct(sem['analogy_native_b']['analogy_top1'])}",
               "", "Niteliksel komşuluklar (çevrilen vektör → qwen uzayı):", ""]
        for w, ns in sem["neighbors"].items():
            md += [f"- `{w}` → {ns}"]
        md += [""]

    if hub:
        md += ["## 4. Nötr ortak uzay (hub-and-spoke, ortak eğitim)", "",
               f"Hub boyutu: {hub['hub_dim']}; 4 spoke × (encoder+decoder) = "
               "8 MCT, tek ortak eğitim; hiçbir çift için ayrı tercüman yok.", ""]
        rows = [[p, pct(v['top1']), pct(v['top5']), f"{v['cosine']:.3f}"]
                for p, v in hub["pairs"].items()]
        md += [table(rows, ["Yön", "Top-1", "Top-5", "Cosine"]), ""]

    if dep:
        md += ["## 5. Derinlik çevirisi (H5)", ""]
        rows = [[p, f"{v['cosine']:.3f}", pct(v.get('top1', 0))]
                for p, v in dep.items() if isinstance(v, dict)]
        md += [table(rows, ["Yol", "Cosine", "Top-1"]),
               f"\n**H5 (llama.L2→hub→qwen.L3 ≥ 0.75 cos): "
               f"{'GEÇTİ ✅' if dep.get('H5_pass') else 'KALDI ❌'}**", ""]

    if sae:
        md += ["## 6. SAE köprüsü (H4)", ""]
        rows = [[r["F"], r["k"], f"{r['ratio_x']:.0f}x",
                 f"{r['recon_cosine']:.3f}", pct(r["top1"]), pct(r["top5"])]
                for r in sae["sweep"]]
        md += [table(rows, ["F", "k", "Oran", "Recon cos", "E2E Top-1",
                            "E2E Top-5"]),
               f"\nSAE'siz referans: top-1 {pct(sae['reference_no_sae']['top1'])}"]
        if "H4" in sae:
            md += [f"\n**H4 (≥8x sıkıştırmada ≤5 puan): en iyi "
                   f"F={sae['H4']['best']['F']}, k={sae['H4']['best']['k']} @ "
                   f"{sae['H4']['best']['ratio_x']:.0f}x, düşüş "
                   f"{sae['H4']['top1_drop_pts']} puan → "
                   f"{'GEÇTİ ✅' if sae['H4']['pass'] else 'KALDI ❌'}**"]
        md += ["", "![SAE](fig_sae.png)", ""]

    if pat:
        a = pat["aggregate"]
        md += ["## 7. Activation patching (H3)", "",
               f"- KL(MCT ‖ temiz) = {a['kl_mct']:.3f}",
               f"- KL(rastgele ‖ temiz) = {a['kl_rand']:.3f}",
               f"- KL oranı = {a['kl_ratio']:.3f} (eşik < 0.2)",
               f"- Top-1 next-token uyumu: MCT {pct(a['top1_agree_mct'])} "
               f"(eşik > %60), rastgele {pct(a['top1_agree_rand'])}",
               f"\n**H3: {'GEÇTİ ✅' if a.get('H3_pass') else 'KALDI ❌'}**", ""]

    (R / "REPORT.md").write_text("\n".join(md))
    print("REPORT DONE ->", R / "REPORT.md")


if __name__ == "__main__":
    main()
