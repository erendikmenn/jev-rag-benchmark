# Güncel paylaşım taslağı — yayımlanmadı

1.044 Türkçe XQuAD sorusu, aynı sabit 20 aday pasaj:

| Yapılandırma | Recall@5 | nDCG@5 | Reranking maliyeti |
|---|---:|---:|---:|
| Hybrid baseline | %97,605 | %94,314 | $0 ek reranking |
| Jev 1.13 — batch | %99,425 | %98,151 | $0,411 |
| Cohere 3.5 — native rerank | %99,425 | %98,626 | $1,044 |
| Clef Flash — batch | %94,444 | %92,567 | $0,168 |
| Clef Flash — ayrı pasaj puanlama | %99,521 | %98,270 | $0,196 |

Clef Flash'ı her pasajı ayrı puanlayarak çalıştırmak 53 batch hatasını düzeltti.
Doğru pasajı aday havuzunda bulunan 1.039 sorunun tamamında ilk beşe ulaştı.
Diğer beş sorunun cevabı havuzda yoktu.

Jev'e göre bir ek doğru erişim ve %52,3 daha düşük kullanım maliyeti;
Cohere sıralama kalitesinde hâlâ önde. Model ile birlikte çalıştırma stratejisi de
değişiyor: Jev pointwise tüm sette tekrar çalıştırılmadı. Bir soruluk fark,
genel model üstünlüğü iddiası için yeterli değil.

Jev'de de küçük tanı örnekleminde sıra/çalıştırma biçimi hassasiyeti var.
Cohere'in 18 kontrolünde belge sırasını çevirmek ilk beş kümesini değiştirmedi.
Tüm 1.044 eski kayıtta Jev/Cohere puan-belge eşleşmeleri ve sıralama doğrulandı.

Clef Flash rotası: OpenRouter → DeepInfra FP8.
Küçük, 240 pasajlık veri seti; tanıdan sonra aynı değerlendirme seti tekrar kullanıldı.
Tablo maliyetleri başarılı tam-koşu çağrılarını içerir; rezerv ve tanı maliyetlerini içermez.
Cevap üretimi tekrar çalıştırılmadı. Ayrı pasaj yöntemi sorgu başına 20 API isteği yapıyor.

Kaynak: https://github.com/erendikmenn/jev-rag-benchmark
