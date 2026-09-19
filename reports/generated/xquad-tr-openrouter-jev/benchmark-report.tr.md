# Jev RAG Benchmark Raporu

> Bu rapor sonuç dosyasından otomatik üretilmiştir. `run_kind=fixture` satırları gerçek Jev veya gerçek üretici LLM sonucu değildir.

## Özet

| Dil | Kol | n | nDCG@10 bağlam | Recall@k | EM | F1 | Başarı | p50 ms | p95 ms | USD/sorgu | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tr | A | 1044 | 0.892 | 0.938 | 0.001 | 0.222 | 0.113 | 1118.8 | 1742.5 | 0.000048 | 0.000 |
| tr | D | 1044 | 0.957 | 0.966 | 0.003 | 0.229 | 0.119 | 1685.1 | 2443.7 | 0.000426 | 0.000 |

## Cevap desteği ve hata davranışı

| Dil | Kol | Kaynakta altın cevap | Geçerli atıf | Yanıtsız | Yanlış cevap |
|---|---:|---:|---:|---:|---:|
| tr | A | 0.940 | 0.828 | 0.183 | 0.704 |
| tr | D | 0.969 | 0.857 | 0.166 | 0.716 |

## Gecikme kırılımı

| Dil | Kol | Retrieval p50/p95 ms | Reranking p50/p95 ms | Generation p50/p95 ms | Uçtan uca p50/p95 ms |
|---|---:|---:|---:|---:|---:|
| tr | A | 1.3/2.4 | 0.0/0.0 | 1117.0/1741.6 | 1118.8/1742.5 |
| tr | D | 1.3/2.4 | 541.0/741.6 | 1110.9/1840.6 | 1685.1/2443.7 |

## Maliyet

- Toplam ölçülen çevrimiçi deney gideri: **$0.494115**.
- Reranking API gideri: **$0.393221**; cevap üretimi API gideri: **$0.100894**.
- Bir defalık indeksleme süresi: **5.6 ms**; API gideri: **$0**.
- Başarılı cevap tanımı: `F1 >= 0.5`; başarılı cevap sayısı: **242**.
- Başarılı cevap başına ölçülen çevrimiçi gider: **$0.002042**.
- `tr-A`: $0.000048/sorgu (rerank $0.000000 + üretim $0.000048); $0.0476/1000 sorgu.
- `tr-D`: $0.000426/sorgu (rerank $0.000377 + üretim $0.000049); $0.4257/1000 sorgu.

## Eşleştirilmiş bootstrap (%95 GA)

- `tr` A→D nDCG@10 farkı: **+0.066** (GA +0.052, +0.079; n=1044).
- `tr` A→D F1 farkı: **+0.007** (GA -0.000, +0.014; n=1044).
- `tr` A→D başarı oranı farkı: **+0.006** (GA -0.006, +0.017; n=1044).

## Yorumlama sınırları

- SciFact qrels yalnızca retrieval değerlendirmesidir; RAG cevap doğruluğu etiketi değildir.
- XQuAD cevapları EM/F1 için kullanılır; kaynakta altın cevabın bulunması ayrıca raporlanır.
- Yerel cross-encoder için API bedeli sıfırdır; gecikme ve yerel donanım süresi maliyet telemetrisidir.
- Tasarruf yüzdesi raporlanacaksa payda A kolunun USD/sorgu değeridir; negatif değerler korunur.
- Bu koşu seçilen test kümesinin tamamını kapsar; başka alanlara otomatik genellenmemelidir.
