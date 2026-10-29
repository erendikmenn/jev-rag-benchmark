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
| Luna Decisions — batch | 98.659% | 1,030 / 1,044 | 97.503% | 97.106% | $1.041619 |
| Luna Decisions — pointwise | 99.234% | 1,036 / 1,044 | 98.268% | 97.933% | $1.077098 |

Jev sürümleri batch altında, Jev 1.14 ve Flash ise pointwise altında karşılaştırılabilir.
Model ve yöntem birlikte değişiyorsa farkı yalnızca modele atfetmemek gerekir.

240 pasajlık küçük ve tekrar kullanılan değerlendirme seti; genel liderlik iddiası değil.
Aday havuzunda doğru pasajı olan soru sayısı 1.039; Recall@5 tavanı %99,521.
Pointwise yöntemler sorgu başına 20 API isteği yapıyor.
Clef Flash: DeepInfra FP8; Jev: TypeSafe. Maliyetler başarılı tam koşu çağrılarına ait.

https://github.com/erendikmenn/jev-rag-benchmark


Aynı Jev 1.13 bağlamında cevap üretimi:

| Model (1,044 questions each) | Token F1 ↑ | Exact match ↑ | F1 ≥ 0.5 | Empty/truncated | Generation cost |
|---|---:|---:|---:|---:|---:|
| Gemini 3.8 Flash | 77.058% | 57.567% | 83.716% | 0 | $2.539025 |
| DeepSeek V4.1 Flash | 70.634% | 44.732% | 78.927% | 6 | $0.459019 |
| GPT-6 Luna | 59.402% | 29.215% | 66.762% | 0 | $0.188533 |
| Haiku 5.5 | 71.185% | 49.138% | 78.927% | 2 | $0.397719 |

F1 sözcük örtüşmesini ölçer; insan değerlendirmesi veya genel doğruluk puanı değildir.

Haiku maliyeti, kullanım bilgisi dönmeyen iki ret için $0,008 rezerv içerir.
