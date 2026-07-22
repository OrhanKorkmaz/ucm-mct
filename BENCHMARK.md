# MCT-Bench — Modeller-Arası Temsil Çevirisi Değerlendirme Standardı

Bu belge, MCT'nin (ve herhangi bir modeller-arası aktivasyon-çeviri
yönteminin) değerlendirilmesi için tekrarlanabilir bir protokol tanımlar.
Amaç: E10 (standart/benchmark eksikliği) engelini kapatmak — sonuçların
karşılaştırılabilir olması.

## 1. Görevler ve metrikler

| Görev | Ölçtüğü | Metrik | Rastgele taban |
|-------|---------|--------|----------------|
| **T1 Kimlik (retrieval)** | çevrilen vektör aynı kavramı buluyor mu | top-1 / top-5 / top-20, havuz boyutu belirtilir | 1/havuz |
| **T2 İşlev (injection)** | çevrilen vektör enjekte edilince alt-akış davranışı | KL-oranı = KL(temiz‖enjekte)/KL(temiz‖rastgele); argmax uyumu | ~0 uyum |
| **T3 Vahşi-metin** | gerçek (Wikipedia) metinde T2 | T2 metrikleri, EN/TR ayrı | rastgele enjeksiyon |
| **T4 Drift** | model güncellemesi (base→instruct) sonrası T1/T2 | retrieval bozulması, KL-oranı | — |
| **T5 Güvenlik** | saldırı tespiti vs zarar | marj = zarar-enerjisi/tespit-enerjisi; takas-tespit-farkı | %5 FP kalibre |
| **T6 Hub** | çok-üye entegrasyon | rota top-1 (A→hub→G), relay düşüşü/sekme, onboarding interferansı | — |
| **T7 Kanal** | latent vs metin fayda | iki-rejim (içerik/dağılım) kazananı; protokol-gereklilik (MCT vs eğitimsiz-proj) | rastgele-vektör |

## 2. Veri

- **Kavram seti:** ~10k EN + TR kavram/kısa-ifade (retrieval havuzu; zor havuz ≥3000).
- **Vahşi-metin:** Wikipedia makalelerinden cümle-önekleri (EN+TR), makale-bazlı
  train/test bölünmesi (sızıntı yok). Bölme: makalelerin son %8'i test.
- **Katman çiftleri:** kaynak/hedef enjeksiyon katmanları CKA ile seçilir,
  raporlanır (ör. Llama L10 → Qwen L18).

## 3. Protokol (zorunlu adımlar)

1. **Standardizasyon:** her boyut z-skoru (train istatistikleriyle); bf16 çıkarım.
2. **Çevirmen:** Procrustes çekirdek + kapılı MLP + norm-kalibreli enjeksiyon.
3. **Bölme:** train/test makale-bazlı; test asla eğitimde görülmez.
4. **Kontroller:** her görevde rastgele-eşleşme tabanı zorunlu.
5. **Raporlama:** havuz boyutu, katman çifti, model sürümleri, dtype belirtilir.

## 4. Referans sonuçlar (bu çalışma, ≤1.5B, 21 Tem 2026)

| Görev | Sonuç | Koşul |
|-------|-------|-------|
| T1 kimlik | top-1 %88, top-5 %99 | zor havuz 3305, çapraz-aile |
| T2 işlev | KL-oranı 0.068, uyum %60 | vahşi-metin |
| T3 EN/TR | EN %66 / TR %53 uyum | rastgele %2.7 |
| T4 drift | retrieval %86.9→%86.9 | base→Instruct |
| T5 güvenlik | manifold-dışı marj 4.0; takas tespit-farkı +0 | iki-katman |
| T6 hub | A→hub→G %96; onboarding %95, interferans 0; relay -%7/sekme | 3 üye |

## 5. Bilinen sınırlar (dürüst raporlama zorunlu)

- **TR kimlik hizası** düşük (göreli sadakat ~0.39) — model-ön-eğitim sınırı.
- **Yetenek nakli** kapsam-dışı (Faz-2): MCT temsil çevirir, hesap/yetenek nakletmez.
- **Ölçek:** ≤1.5B; 3B+ ve içgözlem/ACK açık.

## 6. Nasıl koşulur

`scripts/` altındaki numaralı deneyler her görevi üretir (ör. 29 vahşi-metin,
47-52 hub). Sonuçlar `results/*.json`. Yeni yöntem eklemek için: aynı veri
bölmesi + kontroller + havuz boyutuyla T1-T6'yı raporlayın.
