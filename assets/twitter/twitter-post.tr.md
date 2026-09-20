# Twitter/X tek gönderi taslağı — güncel benchmark

Jev kullanarak RAG sistemine farklı bir yaklaşım getirmeyi denediğim ilk çalışmanın sonuçlarını dün paylaşmıştım.

Sonrasında benchmark’ı baştan sona sıkılaştırdım. Apples-to-oranges kalan noktaları kaldırdım: 1.044 tekil soru, aynı hybrid top-20 aday havuzu, aynı top-5 bağlam bütçesi ve cevap modelleri için tamamen aynı dondurulmuş Jev bağlamları.

Güncel sonuç ilkinden de daha iyi çıktı.

Klasik BM25 top-5:
• Recall@5: %93,77 — 979/1.044
• Kaçırılan soru: 65

Hybrid retrieval, reranker yok:
• Recall@5: %97,61 — 1.019/1.044
• nDCG@10: %94,31
• Kaçırılan soru: 25

Hybrid + Jev:
• Recall@5: %99,43 — 1.038/1.044
• nDCG@10: %98,10
• Kaçırılan soru: yalnızca 6

Yani Jev’in izole etkisini aynı aday havuzunda ölçtüğümüzde, raw hybrid sistemin kaçırdığı 25 sorunun 19’unu geri kazandı. Klasik BM25’e göre ise retrieval hatası 65’ten 6’ya düştü: yaklaşık %90,8 daha az hata.

En çarpıcı kısım şu: Hybrid top-20 içinde doğru kaynak 1.039 soruda vardı. Jev bunların 1.038’inde doğru pasajı ilk 5’e taşıdı. Kalan 6 hatanın 5’inde doğru kaynak zaten aday havuzunda yoktu; Jev’in gerçekten kaçırdığı yalnızca 1 soru kaldı.

Aynı benchmark’ta Cohere Rerank 3.5 de %99,43 Recall@5 aldı. Jev aynı recall’a $0,411 ile ulaştı; Cohere maliyeti $1,044 oldu. Yani yaklaşık %60,6 daha ucuz.

Cevap modelini de apples-to-apples karşılaştırdım. İki model de tamamen aynı Jev top-5 bağlamlarını gördü:

Gemini 3.8 Flash:
• Başarılı cevap: 163 — %15,61
• Ortalama F1: %27,27
• Geçerli atıf: %93,87
• Toplam maliyet: $2,868
• Üretim p50: 3,13 sn

DeepSeek V4.1 Flash:
• Başarılı cevap: 194 — %18,58
• Ortalama F1: %29,19
• Geçerli atıf: %95,69
• Toplam maliyet: $1,070
• Üretim p50: 7,64 sn

DeepSeek 31 daha fazla başarılı cevap verdi ve toplam maliyeti %62,7 düşürdü. Bunun bedeli hız oldu: medyanda yaklaşık 2,44 kat daha yavaş.

Bir de cevap sonrası atıf doğrulama katmanı ekledim. 100 soruluk örneklemde finalde kalan 104/104 iddia-kaynak çifti doğrulandı; toplam maliyet yalnızca %3,7 arttı. Desteklenmeyen cevapta sistem uydurmak yerine çekimser kalabildi.

Burada önemli ayrım şu: %99,43 “soruların %99,43’üne doğru cevap verdik” demek değil. Doğru kaynağı üretici modelin önüne ilk 5 bağlam içinde getirme başarısı. Katı cevap başarısı hâlâ %18,58; yani retrieval tarafındaki darboğazı büyük ölçüde çözerken sıradaki geliştirme alanını da net biçimde gördük: cevap üretimi.

Bu benim ilk ciddi açık kaynak benchmark çalışmamdı. İlk paylaşımda eksikler vardı; sonuçları parlatmak yerine metodolojiyi düzelttim, bütün karşılaştırmaları eşitledim ve ayrıntılı raporu repoya ekledim. Bundan sonra da bu tip deneyleri açık biçimde paylaşmaya devam edeceğim.

Kod: https://github.com/erendikmenn/jev-rag-benchmark
Dataset: https://huggingface.co/datasets/google/xquad
Orijinal kaynak: https://github.com/google-deepmind/xquad
İlk paylaşım: https://x.com/ErenAILab/status/2101417298728792134

## Eklenecek görseller

1. `benchmark-summary.png`
2. `benchmark-details.png`
