# Jev RAG Benchmark Raporu

> Bu rapor sonuç dosyasından otomatik üretilmiştir. `run_kind=fixture` satırları gerçek Jev veya gerçek üretici LLM sonucu değildir.

## Özet

| Dil | Kol | n | nDCG@10 bağlam | Recall@k | EM | F1 | Başarı | p50 ms | p95 ms | USD/sorgu | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tr | V | 100 | 0.977 | 0.990 | 0.010 | 0.275 | 0.170 | 5004.5 | 15283.6 | 0.002615 | 0.000 |

## Cevap desteği ve hata davranışı

| Dil | Kol | Kaynakta altın cevap | Geçerli atıf | Yanıtsız | Yanlış cevap |
|---|---:|---:|---:|---:|---:|
| tr | V | 0.990 | 0.950 | 0.050 | 0.780 |

## Gecikme kırılımı

| Dil | Kol | Retrieval p50/p95 ms | Reranking p50/p95 ms | Generation p50/p95 ms | Uçtan uca p50/p95 ms |
|---|---:|---:|---:|---:|---:|
| tr | V | 544.1/1003.5 | 523.1/713.4 | 3329.8/13550.9 | 5004.5/15283.6 |

## Maliyet

- Toplam ölçülen çevrimiçi deney gideri: **$0.261499**.
- Reranking API gideri: **$0.039491**; cevap üretimi API gideri: **$0.218885**.
- Bir defalık indeksleme süresi: **94.3 ms**; API gideri: **$0**.
- Başarılı cevap tanımı: `F1 >= 0.5`; başarılı cevap sayısı: **17**.
- Başarılı cevap başına ölçülen çevrimiçi gider: **$0.015382**.
- `tr-V`: $0.002615/sorgu (rerank $0.000395 + üretim $0.002189); $2.6150/1000 sorgu.

## Eşleştirilmiş bootstrap (%95 GA)


## Yorumlama sınırları

- SciFact qrels yalnızca retrieval değerlendirmesidir; RAG cevap doğruluğu etiketi değildir.
- XQuAD cevapları EM/F1 için kullanılır; kaynakta altın cevabın bulunması ayrıca raporlanır.
- Yerel cross-encoder için API bedeli sıfırdır; gecikme ve yerel donanım süresi maliyet telemetrisidir.
- Tasarruf yüzdesi raporlanacaksa payda A kolunun USD/sorgu değeridir; negatif değerler korunur.
- Bu koşu seçilen test kümesinin tamamını kapsar; başka alanlara otomatik genellenmemelidir.
