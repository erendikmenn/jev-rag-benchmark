# Jev RAG Benchmark Raporu

> Bu rapor sonuç dosyasından otomatik üretilmiştir. `run_kind=fixture` satırları gerçek Jev veya gerçek üretici LLM sonucu değildir.

## Özet

| Dil | Kol | n | nDCG@10 bağlam | Recall@k | EM | F1 | Başarı | p50 ms | p95 ms | USD/sorgu | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tr | A | 100 | 0.916 | 0.970 | — | — | — | 1.7 | 2.3 | 0.000000 | 0.000 |
| tr | H | 100 | 0.612 | 0.760 | — | — | — | 1183.7 | 1427.1 | 0.005061 | 0.000 |

## Cevap desteği ve hata davranışı

| Dil | Kol | Kaynakta altın cevap | Geçerli atıf | Yanıtsız | Yanlış cevap |
|---|---:|---:|---:|---:|---:|
| tr | A | 0.970 | — | 0.000 | 0.000 |
| tr | H | 0.770 | — | 0.000 | 0.000 |

## Gecikme kırılımı

| Dil | Kol | Retrieval p50/p95 ms | Reranking p50/p95 ms | Generation p50/p95 ms | Uçtan uca p50/p95 ms |
|---|---:|---:|---:|---:|---:|
| tr | A | 1.7/2.3 | 0.0/0.0 | 0.0/0.0 | 1.7/2.3 |
| tr | H | 1.7/2.3 | 1182.1/1425.1 | 0.0/0.0 | 1183.7/1427.1 |

## Maliyet

- Toplam ölçülen çevrimiçi deney gideri: **$0.506090**.
- Reranking API gideri: **$0.506090**; cevap üretimi API gideri: **$0.000000**.
- Bir defalık indeksleme süresi: **10.7 ms**; API gideri: **$0**.
- Başarılı cevap tanımı: `F1 >= 0.5`; başarılı cevap sayısı: **0**.
- Başarılı cevap başına ölçülen çevrimiçi gider: **—**.
- `tr-A`: $0.000000/sorgu (rerank $0.000000 + üretim $0.000000); $0.0000/1000 sorgu.
- `tr-H`: $0.005061/sorgu (rerank $0.005061 + üretim $0.000000); $5.0609/1000 sorgu.

## Eşleştirilmiş bootstrap (%95 GA)

- `tr` A→H nDCG@10 farkı: **-0.304** (GA -0.390, -0.222; n=100).

## Yorumlama sınırları

- SciFact qrels yalnızca retrieval değerlendirmesidir; RAG cevap doğruluğu etiketi değildir.
- XQuAD cevapları EM/F1 için kullanılır; kaynakta altın cevabın bulunması ayrıca raporlanır.
- Yerel cross-encoder için API bedeli sıfırdır; gecikme ve yerel donanım süresi maliyet telemetrisidir.
- Tasarruf yüzdesi raporlanacaksa payda A kolunun USD/sorgu değeridir; negatif değerler korunur.
- Bu koşu seçilen test kümesinin tamamını kapsar; başka alanlara otomatik genellenmemelidir.
