# Jev RAG Benchmark Raporu

> Bu rapor sonuç dosyasından otomatik üretilmiştir. `run_kind=fixture` satırları gerçek Jev veya gerçek üretici LLM sonucu değildir.

## Özet

| Dil | Kol | n | nDCG@10 bağlam | Recall@k | EM | F1 | p50 ms | p95 ms | USD/sorgu | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tr | A | 5 | 0.800 | 0.800 | 0.000 | 0.066 | 392.0 | 395.1 | 0.000000 | 0.000 |

## Cevap desteği ve hata davranışı

| Dil | Kol | Kaynakta altın cevap | Geçerli atıf | Yanıtsız | Yanlış cevap |
|---|---:|---:|---:|---:|---:|
| tr | A | 0.800 | 0.000 | 0.000 | 1.000 |

## Gecikme kırılımı

| Dil | Kol | Retrieval p50/p95 ms | Reranking p50/p95 ms | Generation p50/p95 ms | Uçtan uca p50/p95 ms |
|---|---:|---:|---:|---:|---:|
| tr | A | 0.3/0.7 | 0.0/0.0 | 391.7/394.4 | 392.0/395.1 |

## Maliyet

- Toplam ölçülen çevrimiçi deney gideri: **$0.000000**.
- Bir defalık indeksleme süresi: **5.5 ms**; API gideri: **$0**.
- Başarılı cevap tanımı: `F1 >= 0.5`; başarılı cevap sayısı: **0**.
- Başarılı cevap başına ölçülen çevrimiçi gider: **—**.
- `tr-A`: $0.000000/sorgu; $0.0000/1000 sorgu.

## Eşleştirilmiş bootstrap (%95 GA)


## Yorumlama sınırları

- SciFact qrels yalnızca retrieval değerlendirmesidir; RAG cevap doğruluğu etiketi değildir.
- XQuAD cevapları EM/F1 için kullanılır; kaynakta altın cevabın bulunması ayrıca raporlanır.
- Yerel cross-encoder için API bedeli sıfırdır; gecikme ve yerel donanım süresi maliyet telemetrisidir.
- Tasarruf yüzdesi raporlanacaksa payda A kolunun USD/sorgu değeridir; negatif değerler korunur.
- Küçük smoke örneği kesin sonuç değildir.
