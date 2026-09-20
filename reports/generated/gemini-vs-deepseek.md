# Gemini 3.8 Flash vs DeepSeek V4.1 Flash

- Eşleştirilmiş soru: **1044**
- İki model de aynı dondurulmuş sorguları, aynı Jev sırasını ve aynı bağlamları kullandı.
- Başarı tanımı: atıflar çıkarıldıktan sonra token F1 ≥ 0.5.

| Metrik | Gemini 3.8 Flash | DeepSeek V4.1 Flash |
|---|---:|---:|
| EM | 0.0010 | 0.0086 |
| F1 | 0.2727 | 0.2919 |
| Başarı | 0.1561 | 0.1858 |
| Geçerli atıf | 0.9387 | 0.9569 |
| Yanıtsız bırakma | 0.0613 | 0.0421 |
| Yanlış cevap | 0.7826 | 0.7720 |

## Eşleştirilmiş farklar (DeepSeek − Gemini)

- Ortalama F1 farkı: **+0.0191** (%95 GA +0.0103, +0.0286).
- Başarı oranı farkı: **+0.0297** (%95 GA +0.0096, +0.0508).
- DeepSeek daha iyi / Gemini daha iyi / eşit: **404 / 278 / 362**.

## Hız ve maliyet

| Model | p50 ms | p95 ms | USD/sorgu | Toplam USD |
|---|---:|---:|---:|---:|
| Gemini 3.8 Flash | 3127.9 | 13114.2 | $0.002354 | $2.457358 |
| DeepSeek V4.1 Flash | 7640.7 | 46536.9 | $0.000631 | $0.659245 |

## Çözümlenen model kimlikleri

- Gemini: `google/gemini-3.8-flash`
- DeepSeek: `deepseek/deepseek-v4.1-flash`
