# Jev RAG Benchmark Raporu

> Bu rapor sonuç dosyasından otomatik üretilmiştir. `run_kind=fixture` satırları gerçek Jev veya gerçek üretici LLM sonucu değildir.

## Özet

| Dil | Kol | n | nDCG@10 bağlam | Recall@k | EM | F1 | Başarı | p50 ms | p95 ms | USD/sorgu | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tr | K | 1044 | 0.981 | 0.994 | 0.009 | 0.292 | 0.186 | 8706.6 | 47628.7 | 0.001025 | 0.000 |

## Cevap desteği ve hata davranışı

| Dil | Kol | Kaynakta altın cevap | Geçerli atıf | Yanıtsız | Yanlış cevap |
|---|---:|---:|---:|---:|---:|
| tr | K | 0.994 | 0.957 | 0.042 | 0.772 |

## Gecikme kırılımı

| Dil | Kol | Retrieval p50/p95 ms | Reranking p50/p95 ms | Generation p50/p95 ms | Uçtan uca p50/p95 ms |
|---|---:|---:|---:|---:|---:|
| tr | K | 529.5/880.4 | 532.0/705.3 | 7640.7/46536.9 | 8706.6/47628.7 |

## Maliyet

- Toplam ölçülen çevrimiçi deney gideri: **$1.070110**.
- Reranking API gideri: **$0.410866**; cevap üretimi API gideri: **$0.659245**.
- Bir defalık indeksleme süresi: **0.0 ms**; API gideri: **$0**.
- Başarılı cevap tanımı: `F1 >= 0.5`; başarılı cevap sayısı: **194**.
- Başarılı cevap başına ölçülen çevrimiçi gider: **$0.005516**.
- `tr-K`: $0.001025/sorgu (rerank $0.000394 + üretim $0.000631); $1.0250/1000 sorgu.

## Eşleştirilmiş bootstrap (%95 GA)


## Yorumlama sınırları

- SciFact qrels yalnızca retrieval değerlendirmesidir; RAG cevap doğruluğu etiketi değildir.
- XQuAD cevapları EM/F1 için kullanılır; kaynakta altın cevabın bulunması ayrıca raporlanır.
- Yerel cross-encoder için API bedeli sıfırdır; gecikme ve yerel donanım süresi maliyet telemetrisidir.
- Tasarruf yüzdesi raporlanacaksa payda A kolunun USD/sorgu değeridir; negatif değerler korunur.
- Bu koşu seçilen test kümesinin tamamını kapsar; başka alanlara otomatik genellenmemelidir.
