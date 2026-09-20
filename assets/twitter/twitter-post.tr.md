# Twitter/X tek gönderi taslağı — güncel benchmark

Dostlar, dün paylaştığım Jev + RAG çalışmasının sonuçlarını güncelledim. Artık çok daha iyi ve çok daha adil bir karşılaştırmamız var. Mutlaka incelemenizi ve önerilerinizi paylaşmanızı isterim.

Apples-to-oranges kalan noktaları düzelttim: 1.044 soruda aynı hybrid top-20 aday havuzu, aynı top-5 bağlam bütçesi ve cevap modelleri için aynı dondurulmuş Jev bağlamları kullanıldı.

Klasik BM25 top-5:
• Recall@5: %93,77
• Kaçırılan soru: 65

Hybrid retrieval:
• Recall@5: %97,61
• nDCG@10: %94,31
• Kaçırılan soru: 25

Hybrid + Jev:
• Recall@5: %99,43
• nDCG@10: %98,10
• Kaçırılan soru: yalnızca 6

Jev, aynı aday havuzunda hybrid sistemin kaçırdığı 25 sorunun 19’unu geri kazandı. Klasik BM25’e göre retrieval hatası 65’ten 6’ya düştü. Hybrid top-20’de doğru kaynak bulunan 1.039 sorunun 1.038’inde doğru pasaj ilk 5’e taşındı.

Jev’i Cohere Rerank 3.5 ile de karşılaştırdım. İkisi de %99,43 Recall@5 aldı; Jev’in toplam rerank maliyeti $0,411, Cohere’in ise $1,044 oldu. Aynı recall, yaklaşık %60,6 daha düşük maliyet.

Aynı Jev bağlamlarıyla cevap modellerini karşılaştırdığımda Gemini 163, DeepSeek ise 194 başarılı cevap verdi. DeepSeek toplam maliyeti %62,7 düşürdü ve 31 ek başarılı cevap üretti; fakat medyanda 2,44 kat daha yavaştı.

Cevap sonrası doğrulama katmanında, 100 soruluk örneklemde finalde kalan 104/104 iddia-kaynak çifti doğrulandı. Bunun toplam maliyete etkisi yalnızca %3,7 oldu.

Önemli not: %99,43 cevap doğruluğu değil, doğru kaynağın ilk 5 bağlama girme oranı. Yani retrieval tarafındaki darboğazı büyük ölçüde çözdük; geliştirilmesi gereken asıl bölüm artık cevap üretimi.

Kod: https://github.com/erendikmenn/jev-rag-benchmark
Dataset: https://huggingface.co/datasets/google/xquad
Orijinal kaynak: https://github.com/google-deepmind/xquad

## Eklenecek görseller

1. `benchmark-bars.png`
2. `benchmark-summary.png`
3. `benchmark-details.png`
