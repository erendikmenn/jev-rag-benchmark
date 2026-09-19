# XQuAD-TR tam OpenRouter Jev analizi

- Eşleştirilmiş benzersiz soru: **1044**
- Sonuç satırı: **2088** (A ve D)
- Aday liste uyuşmazlığı: **0**
- Aday belgelerde temsil edilen farklı makale başlığı: **48**
- Başarı tanımı: atıflar çıkarıldıktan sonra token F1 ≥ 0.5.

## Ana metrikler

| Metrik | A: normal RAG | D: Jev |
|---|---:|---:|
| nDCG@10 | 0.8917 | 0.9573 |
| Recall@5 | 0.9377 | 0.9665 |
| Kaynakta altın cevap | 0.9397 | 0.9693 |
| EM | 0.0010 | 0.0029 |
| F1 | 0.2224 | 0.2289 |
| Başarı oranı | 0.1130 | 0.1188 |
| Geçerli atıf | 0.8276 | 0.8573 |
| Yanıtsız bırakma | 0.1830 | 0.1657 |
| Yanlış cevap | 0.7040 | 0.7155 |
| Fallback | 0.0000 | 0.0000 |
| Altın cevap aynen içeriliyor | 0.6839 | 0.6944 |
| Altın token recall | 0.6758 | 0.6781 |

## Eşleştirilmiş farklar

- Ortalama F1 farkı: **+0.0066** (%95 GA -0.0000, +0.0139).
- Başarı oranı farkı: **+0.0057** (%95 GA -0.0057, +0.0172).
- F1'e göre Jev daha iyi / normal daha iyi / eşit: **223 / 197 / 624**.
- Başarısız→başarılı / başarılı→başarısız: **23 / 17**.
- Altın kaynağı bağlama kazandırdı / kaybetti: **31 / 0**.
- Yanıtsızdan cevaba / cevaptan yanıtsıza: **62 / 44**.

## Gecikme (p50 / p95 / p99 ms)

| Aşama | A | D |
|---|---:|---:|
| Retrieval | 1.3 / 2.4 / 3.0 | 1.3 / 2.4 / 3.0 |
| Reranking | 0.0 / 0.0 / 0.0 | 541.0 / 741.6 / 920.6 |
| Generation | 1117.0 / 1741.6 / 2560.8 | 1110.9 / 1840.6 / 2721.5 |
| Uçtan uca | 1118.8 / 1742.5 / 2562.0 | 1685.1 / 2443.7 / 3338.9 |

## Maliyet ve güvenilirlik

- Reranking: **$0.393221**; cevap üretimi: **$0.100894**; toplam: **$0.494115**.
- A maliyet/sorgu: **$0.000048**.
- D maliyet/sorgu: **$0.000426**.
- A başarı başına maliyet: **$0.000421** (118 başarı).
- D başarı başına maliyet: **$0.003584** (124 başarı).
- Reranker hatası / generator hatası: **0 / 0**.
- Retry kullanılan satır / toplam retry: **0 / 0**.

## En sık temsil edilen makaleler (aday havuzu)

| Makale | Soru-aday görünümü |
|---|---:|
| Warsaw | 800 |
| Economic_inequality | 682 |
| Martin_Luther | 681 |
| Black_Death | 616 |
| Genghis_Khan | 591 |
| Doctor_Who | 587 |
| United_Methodist_Church | 585 |
| Geology | 570 |
| Victoria_and_Albert_Museum | 539 |
| Super_Bowl_50 | 536 |
| Newcastle_upon_Tyne | 535 |
| European_Union_law | 526 |
| Fresno,_California | 511 |
| Ctenophora | 508 |
| Imperialism | 498 |
| Nikola_Tesla | 486 |
| Jacksonville,_Florida | 484 |
| American_Broadcasting_Company | 481 |
| Teacher | 463 |
| Prime_number | 452 |

## Jev'in en fazla iyileştirdiği cevap örnekleri

### Hangi rekonstrüksiyonlar 1999'daki makalenin bilgilerini desteklemiştir?

- Altın: Jones et al. 1998, Pollack, Huang & Shen 1998, Crowley & Lowery 2000 ve Briffa 2000
- A (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- D (F1 0.824): MBH99 bulguları, Jones et al. 1998, Pollack, Huang & Shen 1998, Crowley & Lowery 2000 ve Briffa 2000 atıflı rekonstrüksiyonları tarafından desteklenmiştir [xquad-tr-1205a55bc8be].
- Fark: **+0.824**

### Hedefte pakete ne olur?

- Altın: orijinal mesaj/veri paket sıralama sayısına göre doğru sırada yeniden birleştirilir
- A (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- D (F1 0.783): Hedefte, orijinal mesajın veya verinin paket sıralama sayısına göre doğru sırada yeniden birleştirilir [xquad-tr-4abaaaf8c66e].
- Fark: **+0.783**

### Çin'de, bu kişi toprağın dağların aşınması ve alüvyon birikimi ile oluştuğu sonucuna varmıştır, ismi nedir?

- Altın: Shen Kuo
- A (F1 0.267): Shen Kuo, toprağın dağların aşınması ve alüvyon birikimi ile oluştuğu sonucuna varan kişidir [xquad-tr-36478d7e8467].
- D (F1 1.000): Shen Kuo [xquad-tr-36478d7e8467].
- Fark: **+0.733**

### Parlamento seçimleri hangi sıklıkta gerçekleşir?

- Altın: beş yılda bir
- A (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- D (F1 0.667): Parlamento seçimleri beş yılda bir gerçekleşir [xquad-tr-c84dc45a4483].
- Fark: **+0.667**

### Parlamento seçimleri hangi sıklıkta yapılır?

- Altın: beş yılda bir
- A (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- D (F1 0.667): Parlamento seçimleri beş yılda bir gerçekleşir [xquad-tr-c84dc45a4483].
- Fark: **+0.667**

### Jacksonville ile birlikte hangi nehir akıyor?

- Altın: St. Johns Nehri'nin
- A (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- D (F1 0.667): Jacksonville, St. Johns Nehri'nin kıyısında bulunmaktadır [xquad-tr-f3632c9a76b7].
- Fark: **+0.667**

### Jacksonville'deki yeni bina projelerinden kim sorumluydu?

- Altın: Belediye Başkanı W. Haydon Burns
- A (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- D (F1 0.667): Jacksonville'deki yeni bina projelerinden Belediye Başkanı W. Haydon Burns sorumluydu [xquad-tr-4a841f83d10a].
- Fark: **+0.667**

### İskoç Parlamento üyesi olmak için kaç yaşında olmak gereklidir?

- Altın: 18 yaş üstünde
- A (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- D (F1 0.500): İskoç Parlamentosu üyeliği için 18 yaş üstünde olmak gerekir [xquad-tr-408e1ceaefad].
- Fark: **+0.500**


## Jev'in en fazla kötüleştirdiği cevap örnekleri

### Rodofit ne anlama gelmektedir?

- Altın: kırmızı alg kloroplast soyu
- A (F1 0.727): Rodofit, kırmızı alg kloroplast soyu anlamına gelmektedir [xquad-tr-4a718c2591c6].
- D (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Fark: **-0.727**

### Kloroplastidan ne anlama gelmektedir?

- Altın: yeşil kloroplast soyu
- A (F1 0.667): Kloroplastidan, yeşil kloroplast soyu anlamına gelmektedir [xquad-tr-4a718c2591c6].
- D (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Fark: **-0.667**

### Hangi kültürler Kubilay'ın yönetiminin parçalarıydı?

- Altın: Han Çinlileri, Hitaylar, Curçenler, Moğollar ve Tibet Budistleri
- A (F1 0.727): Kubilay'ın yönetimi, Han Çinlileri, Hitaylar, Curçenler, Moğollar ve Tibet Budistleri gibi çeşitli kültürleri yansıtmaktı [xquad-tr-1005ea8ca58f].
- D (F1 0.167): Kubilay Han'ın yönetimi, Çinlileri, Hitayları, Curçenleri, Moğolları ve Tibet Budistlerini içeren çeşitli kültürlerden oluşuyordu [xquad-tr-1005ea8ca58f]. Bu bürokrasi, yerli Çin hanedanlarının yanı sıra farklı kültürlerden unsurların bir karışımından meydana geliyordu [xquad-tr-1005ea8ca58f].
- Fark: **-0.561**

### AFC Şampiyonluk oyununda Broncos'u kim yenmiştir?

- Altın: New England Patriots
- A (F1 0.545): AFC Şampiyonluk oyununda Broncos'u New England Patriots yenmiştir [xquad-tr-391129eff6d9].
- D (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Fark: **-0.545**

### Bu reform yanlısı lider nerede eğitim verdi?

- Altın: Paris Üniversitesi'nde
- A (F1 0.444): Lefevre, Paris Üniversitesi'nde profesör olarak görev yaptı. [xquad-tr-294593759fcc]
- D (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Fark: **-0.444**

### Bir türbindeki yaklaşık yoğuşturucu sıcaklığı nedir?

- Altın: 30 °C'dir
- A (F1 0.444): Bir türbindeki yaklaşık yoğuşturucu sıcaklığı 30 °C'dir [xquad-tr-3f408f83fc7e].
- D (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Fark: **-0.444**

### Gelir eşitsizliğinin büyümeye zarar verdiği görüşüne dönmenin ana nedeni neydi?

- Altın: beşerî sermayenin kalkınmadaki artan önemidir
- A (F1 0.357): Gelir eşitsizliğinin büyümeye zarar verdiği görüşüne dönmenin ana nedeni, beşerî sermayenin kalkınmadaki artan önemidir [xquad-tr-63a31756327f]. Bu değişimle birlikte yaygın eğitim, büyümenin sırrı haline gelmiştir [xquad-tr-63a31756327f].
- D (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Fark: **-0.357**

### BSkyB'nin spor portföyü neleri içerir?

- Altın: İngiltere Premier League Futbol
- A (F1 0.353): BSkyB'nin spor portföyü, İngiltere Premier League Futbol'u da içeren çeşitli spor içeriklerini barındırır [xquad-tr-3ecd8572f796].
- D (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Fark: **-0.353**


## Yorumlama

- Retrieval kazanımı cevap başarısına bire bir taşınmaz; cevaplayıcı doğru kaynak mevcutken de uzun, eksik veya yanlış odaklı yanıt üretebilir.
- EM tam metin eşleşmesidir ve kaynaklı tam cümleleri sert biçimde cezalandırır; F1 ve kaynak desteği birlikte okunmalıdır.
- Bu çalışma tüm benzersiz XQuAD-TR test sorularını kapsar; sonuç başka alanlara otomatik genellenmemelidir.
