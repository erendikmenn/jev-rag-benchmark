# Güncel paylaşım taslağı — yayımlanmadı

1.044 Türkçe XQuAD sorusu, aynı sabit 20 aday pasaj.

| Method | Recall@5 ↑ | Gold in top-5 | nDCG@5 ↑ | MRR@5 ↑ | Reranking cost |
|---|---:|---:|---:|---:|---:|
| Hybrid baseline | 97.605% | 1,019 / 1,044 | 94.314% | 93.194% | $0.000000 |
| Jev 1.13 — batch | 99.425% | 1,038 / 1,044 | 98.151% | 97.712% | $0.410866 |
| Jev 1.14 — batch | 99.425% | 1,038 / 1,044 | 98.067% | 97.601% | $0.410866 |
| Jev 1.14 — pointwise | 99.234% | 1,036 / 1,044 | 97.728% | 97.213% | $0.643918 |
| Cohere Rerank 3.5 — native API | 99.425% | 1,038 / 1,044 | 98.626% | 98.348% | $1.044000 |
| Clef Flash — batch, DeepInfra FP8 | 94.444% | 986 / 1,044 | 92.567% | 91.935% | $0.168440 |
| Clef Flash — pointwise, DeepInfra FP8 | 99.521% | 1,039 / 1,044 | 98.270% | 97.840% | $0.195843 |

Jev sürümleri batch altında, Jev 1.14 ve Flash ise pointwise altında karşılaştırılabilir.
Model ve yöntem birlikte değişiyorsa farkı yalnızca modele atfetmemek gerekir.

240 pasajlık küçük ve tekrar kullanılan değerlendirme seti; genel liderlik iddiası değil.
Aday havuzunda doğru pasajı olan soru sayısı 1.039; Recall@5 tavanı %99,521.
Pointwise yöntemler sorgu başına 20 API isteği yapıyor.
Clef Flash: DeepInfra FP8; Jev: TypeSafe. Maliyetler başarılı tam koşu çağrılarına ait.

https://github.com/erendikmenn/jev-rag-benchmark
