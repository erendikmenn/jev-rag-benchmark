# Jev RAG Benchmark Raporu

> Bu rapor sonuç dosyasından otomatik üretilmiştir. `run_kind=fixture` satırları gerçek Jev veya gerçek üretici LLM sonucu değildir.

## Özet

| Dil | Kol | n | nDCG@10 bağlam | Recall@k | EM | F1 | Başarı | p50 ms | p95 ms | USD/sorgu | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tr | A | 1044 | 0.943 | 0.976 | — | — | — | 529.5 | 880.4 | 0.000000 | 0.000 |
| tr | D | 1044 | 0.981 | 0.994 | — | — | — | 1076.7 | 1534.8 | 0.000394 | 0.000 |
| tr | O | 1044 | 0.986 | 0.994 | — | — | — | 1037.4 | 1607.4 | 0.001000 | 0.000 |

## Cevap desteği ve hata davranışı

| Dil | Kol | Kaynakta altın cevap | Geçerli atıf | Yanıtsız | Yanlış cevap |
|---|---:|---:|---:|---:|---:|
| tr | A | 0.979 | — | 0.000 | 0.000 |
| tr | D | 0.994 | — | 0.000 | 0.000 |
| tr | O | 0.994 | — | 0.000 | 0.000 |

## Gecikme kırılımı

| Dil | Kol | Retrieval p50/p95 ms | Reranking p50/p95 ms | Generation p50/p95 ms | Uçtan uca p50/p95 ms |
|---|---:|---:|---:|---:|---:|
| tr | A | 529.5/880.4 | 0.0/0.0 | 0.0/0.0 | 529.5/880.4 |
| tr | D | 529.5/880.4 | 532.0/705.3 | 0.0/0.0 | 1076.7/1534.8 |
| tr | O | 535.8/915.6 | 466.1/836.2 | 0.0/0.0 | 1037.4/1607.4 |

## Maliyet

- Toplam ölçülen çevrimiçi deney gideri: **$1.455391**.
- Reranking API gideri: **$1.454866**; cevap üretimi API gideri: **$0.000000**.
- Bir defalık indeksleme süresi: **383.7 ms**; API gideri: **$0**.
- Başarılı cevap tanımı: `F1 >= 0.5`; başarılı cevap sayısı: **0**.
- Başarılı cevap başına ölçülen çevrimiçi gider: **—**.
- `tr-A`: $0.000000/sorgu (rerank $0.000000 + üretim $0.000000); $0.0002/1000 sorgu.
- `tr-D`: $0.000394/sorgu (rerank $0.000394 + üretim $0.000000); $0.3937/1000 sorgu.
- `tr-O`: $0.001000/sorgu (rerank $0.001000 + üretim $0.000000); $1.0002/1000 sorgu.

## Eşleştirilmiş bootstrap (%95 GA)

- `tr` A→D nDCG@10 farkı: **+0.038** (GA +0.027, +0.049; n=1044).
- `tr` A→O nDCG@10 farkı: **+0.043** (GA +0.033, +0.054; n=1044).

## Yorumlama sınırları

- SciFact qrels yalnızca retrieval değerlendirmesidir; RAG cevap doğruluğu etiketi değildir.
- XQuAD cevapları EM/F1 için kullanılır; kaynakta altın cevabın bulunması ayrıca raporlanır.
- Yerel cross-encoder için API bedeli sıfırdır; gecikme ve yerel donanım süresi maliyet telemetrisidir.
- Tasarruf yüzdesi raporlanacaksa payda A kolunun USD/sorgu değeridir; negatif değerler korunur.
- Bu koşu seçilen test kümesinin tamamını kapsar; başka alanlara otomatik genellenmemelidir.
