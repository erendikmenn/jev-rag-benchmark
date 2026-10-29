# Güncel paylaşım taslağı — yayımlanmadı

1.044 Türkçe XQuAD sorusu, aynı sabit 20 aday pasaj:

| Model | Recall@5 | nDCG@5 | Reranking maliyeti |
|---|---:|---:|---:|
| Hybrid baseline | %97,605 | %94,314 | $0 ek reranking |
| Jev 1.13 | %99,425 | %98,151 | $0,411 |
| Cohere 3.5 | %99,425 | %98,626 | $1,044 |
| Clef Flash / DeepInfra FP8 | %94.444 | %92.567 | $0.168 |

Jev ile Cohere'in top-5 erişim başarısı eşit; Cohere sıralamada biraz daha iyi.
Clef Flash daha ucuz, fakat 20 soruluk batch biçiminde baseline'ın da gerisinde.
Sonraki küçük tanı testinde pasajları ayrı puanlamak seçilen 6 hatayı düzeltti;
bu, modelin genel yeteneği hakkında kesin hüküm vermemeyi gerektiriyor.

Küçük, 240 pasajlık veri seti; genel liderlik iddiası değil. Fiyatlar bu koşuların
ölçülen API kullanımını gösterir; pilot, tekrar ve altyapı giderleri tabloya dahil değil.
Clef Flash için cevap üretimi tekrar çalıştırılmadı.

Kaynak: https://github.com/erendikmenn/jev-rag-benchmark
