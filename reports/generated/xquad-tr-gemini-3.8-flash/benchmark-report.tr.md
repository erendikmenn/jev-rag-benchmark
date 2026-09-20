# Jev RAG Benchmark Raporu

> Bu rapor sonuç dosyasından otomatik üretilmiştir. `run_kind=fixture` satırları gerçek Jev veya gerçek üretici LLM sonucu değildir.

## Özet

| Dil | Kol | n | nDCG@10 bağlam | Recall@k | EM | F1 | Başarı | p50 ms | p95 ms | USD/sorgu | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tr | G | 1044 | 0.981 | 0.994 | 0.001 | 0.273 | 0.156 | 4301.6 | 14304.5 | 0.002747 | 0.000 |

## Cevap desteği ve hata davranışı

| Dil | Kol | Kaynakta altın cevap | Geçerli atıf | Yanıtsız | Yanlış cevap |
|---|---:|---:|---:|---:|---:|
| tr | G | 0.994 | 0.939 | 0.061 | 0.783 |

## Gecikme kırılımı

| Dil | Kol | Retrieval p50/p95 ms | Reranking p50/p95 ms | Generation p50/p95 ms | Uçtan uca p50/p95 ms |
|---|---:|---:|---:|---:|---:|
| tr | G | 529.5/880.4 | 532.0/705.3 | 3127.9/13114.2 | 4301.6/14304.5 |

## Maliyet

- Toplam ölçülen çevrimiçi deney gideri: **$2.868223**.
- Reranking API gideri: **$0.410866**; cevap üretimi API gideri: **$2.457358**.
- Bir defalık indeksleme süresi: **0.0 ms**; API gideri: **$0**.
- Başarılı cevap tanımı: `F1 >= 0.5`; başarılı cevap sayısı: **163**.
- Başarılı cevap başına ölçülen çevrimiçi gider: **$0.017596**.
- `tr-G`: $0.002747/sorgu (rerank $0.000394 + üretim $0.002354); $2.7473/1000 sorgu.

## Eşleştirilmiş bootstrap (%95 GA)


## Yorumlama sınırları

- SciFact qrels yalnızca retrieval değerlendirmesidir; RAG cevap doğruluğu etiketi değildir.
- XQuAD cevapları EM/F1 için kullanılır; kaynakta altın cevabın bulunması ayrıca raporlanır.
- Yerel cross-encoder için API bedeli sıfırdır; gecikme ve yerel donanım süresi maliyet telemetrisidir.
- Tasarruf yüzdesi raporlanacaksa payda A kolunun USD/sorgu değeridir; negatif değerler korunur.
- Bu koşu seçilen test kümesinin tamamını kapsar; başka alanlara otomatik genellenmemelidir.
