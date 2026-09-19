# Jev RAG Benchmark Raporu

> Bu rapor sonuç dosyasından otomatik üretilmiştir. `run_kind=fixture` satırları gerçek Jev veya gerçek üretici LLM sonucu değildir.

## Özet

| Dil | Kol | n | nDCG@10 bağlam | Recall@k | EM | F1 | p50 ms | p95 ms | USD/sorgu | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | A | 25 | 0.715 | 0.740 | — | — | 29.3 | 39.5 | 0.000000 | 0.000 |
| en | B | 25 | 0.750 | 0.780 | — | — | 352.0 | 368.5 | 0.000000 | 0.000 |
| en | D | 25 | 0.816 | 0.848 | — | — | 667.4 | 769.2 | 0.000408 | 0.000 |

## Cevap desteği ve hata davranışı

| Dil | Kol | Kaynakta altın cevap | Geçerli atıf | Yanıtsız | Yanlış cevap |
|---|---:|---:|---:|---:|---:|
| en | A | — | — | 0.000 | 0.000 |
| en | B | — | — | 0.000 | 0.000 |
| en | D | — | — | 0.000 | 0.000 |

## Gecikme kırılımı

| Dil | Kol | Retrieval p50/p95 ms | Reranking p50/p95 ms | Generation p50/p95 ms | Uçtan uca p50/p95 ms |
|---|---:|---:|---:|---:|---:|
| en | A | 29.3/39.5 | 0.0/0.0 | 0.0/0.0 | 29.3/39.5 |
| en | B | 29.3/39.5 | 316.9/338.3 | 0.0/0.0 | 352.0/368.5 |
| en | D | 29.3/39.5 | 646.4/740.0 | 0.0/0.0 | 667.4/769.2 |

## Maliyet

- Toplam ölçülen çevrimiçi deney gideri: **$0.010198**.
- Reranking API gideri: **$0.010198**; cevap üretimi API gideri: **$0.000000**.
- Bir defalık indeksleme süresi: **209.7 ms**; API gideri: **$0**.
- Başarılı cevap tanımı: `F1 >= 0.5`; başarılı cevap sayısı: **0**.
- Başarılı cevap başına ölçülen çevrimiçi gider: **—**.
- `en-A`: $0.000000/sorgu (rerank $0.000000 + üretim $0.000000); $0.0000/1000 sorgu.
- `en-B`: $0.000000/sorgu (rerank $0.000000 + üretim $0.000000); $0.0000/1000 sorgu.
- `en-D`: $0.000408/sorgu (rerank $0.000408 + üretim $0.000000); $0.4079/1000 sorgu.

## Eşleştirilmiş bootstrap (%95 GA)

- `en` A→B nDCG@10 farkı: **+0.035** (GA -0.010, +0.100; n=25).
- `en` A→D nDCG@10 farkı: **+0.101** (GA +0.006, +0.215; n=25).

## Yorumlama sınırları

- SciFact qrels yalnızca retrieval değerlendirmesidir; RAG cevap doğruluğu etiketi değildir.
- XQuAD cevapları EM/F1 için kullanılır; kaynakta altın cevabın bulunması ayrıca raporlanır.
- Yerel cross-encoder için API bedeli sıfırdır; gecikme ve yerel donanım süresi maliyet telemetrisidir.
- Tasarruf yüzdesi raporlanacaksa payda A kolunun USD/sorgu değeridir; negatif değerler korunur.
- Küçük smoke örneği kesin sonuç değildir.
