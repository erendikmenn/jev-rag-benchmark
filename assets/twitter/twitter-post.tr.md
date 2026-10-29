# Güncel paylaşım taslağı — 29 Ekim 2026

Jev RAG benchmarkını aynı aday metinleri ve sırasıyla 1.044 Türkçe XQuAD sorusunda yeniden çalıştırdık.

Jev ve Cohere Recall@5: %99,43. Cohere nDCG@5'te az farkla önde (%98,63 / %98,15);
Jev reranking maliyetinde yaklaşık %61 daha ucuz ($0,411 / $1,044).

Aynı Jev bağlamı ve kısa cevap promptuyla Gemini F1 %77,06; DeepSeek %70,63.
Gemini bu protokolde daha yüksek sözcüksel kalite, DeepSeek daha düşük üretim maliyeti verdi.
DeepSeek'in altı boş/kesilmiş cevabı sıfır puanla dahil edildi.

Eski cümle-cevap puanlaması, farklı Cohere adayları ve @10 etiketleri yanıltıcıydı.
Önceki kalite önerileri geçersiz; eski raporlar tarihsel kayıt olarak işaretlendi.
Recall cevap doğruluğu değildir; F1 da insan değerlendirmesi değildir.

Kod ve ham sonuçlar: https://github.com/erendikmenn/jev-rag-benchmark

Görseller: benchmark-bars.png, benchmark-summary.png, benchmark-details.png.
Bu bir taslaktır; otomatik olarak sosyal medyada yayımlanmamıştır.
