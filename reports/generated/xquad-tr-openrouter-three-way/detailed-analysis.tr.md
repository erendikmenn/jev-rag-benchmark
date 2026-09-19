# XQuAD-TR üçlü üretici karşılaştırması

- Eşleştirilmiş benzersiz soru: **1044**
- Toplam sonuç satırı: **3132**
- G kolu, D kolunun dondurulmuş Jev bağlamlarını aynen kullanır; D→G farkı yalnızca üretici modeldir.
- Başarı tanımı: atıflar çıkarıldıktan sonra token F1 ≥ 0.5.

## Ana sonuçlar

| Metrik | Normal + Qwen | Jev + Qwen | Jev + Gemini 3.8 medium |
|---|---:|---:|---:|
| nDCG@10 | 0.8917 | 0.9573 | 0.9573 |
| Recall@5 | 0.9377 | 0.9665 | 0.9665 |
| Kaynakta altın cevap | 0.9397 | 0.9693 | 0.9693 |
| EM | 0.0010 | 0.0029 | 0.0010 |
| F1 | 0.2224 | 0.2289 | 0.2651 |
| Geçerli atıf | 0.8276 | 0.8573 | 0.9033 |
| Yanıtsız bırakma | 0.1830 | 0.1657 | 0.0967 |
| Yanlış cevap | 0.7040 | 0.7155 | 0.7414 |
| Başarı oranı | 0.1130 | 0.1188 | 0.1619 |
| Altın cevap aynen içeriliyor | 0.6839 | 0.6944 | 0.7720 |
| Altın token recall | 0.6758 | 0.6781 | 0.7522 |

## D→G: yalnızca üretici model farkı

- Ortalama F1 farkı: **+0.0361** (%95 GA +0.0275, +0.0451).
- Başarı oranı farkı: **+0.0431** (%95 GA +0.0239, +0.0632).
- Gemini daha iyi / Qwen daha iyi / eşit F1: **424 / 279 / 341**.
- Başarısız→başarılı / başarılı→başarısız: **72 / 27**.
- Yanlış cevabı düzeltti / yeni yanlış cevap üretti: **81 / 108**.
- Yanıtsızdan cevaba / cevaptan yanıtsıza: **94 / 22**.

## Gecikme (p50 / p95 / p99 ms)

| Aşama | Normal + Qwen | Jev + Qwen | Jev + Gemini |
|---|---:|---:|---:|
| Generation | 1117.0 / 1741.6 / 2560.8 | 1110.9 / 1840.6 / 2721.5 | 2885.5 / 6946.2 / 10125.3 |
| Uçtan uca | 1118.8 / 1742.5 / 2562.0 | 1685.1 / 2443.7 / 3338.9 | 3451.9 / 7551.2 / 10647.8 |

## Maliyet

- Normal RAG + Qwen3.7 Flash: **$0.000048/sorgu**, **$0.000421/başarı** (118 başarı).
- Jev + Qwen3.7 Flash: **$0.000426/sorgu**, **$0.003584/başarı** (124 başarı).
- Jev + Gemini 3.8 Flash (medium): **$0.002510/sorgu**, **$0.015508/başarı** (169 başarı).
- Nihai G satırlarının Gemini üretim gideri: **$2.227552**.
- Bu tutar yalnızca nihai tutulan cevapları kapsar; token-sınırı ayarı sırasında atılan ve yeniden çalıştırılan deneme çağrıları dahil değildir.
- Jev + Gemini kolunun üretimde varsayımsal toplam gideri: **$2.620773** (Jev maliyeti önceki koşudan kopyalanmıştır; ikinci kez ödenmemiştir).

## Gemini'nin en fazla iyileştirdiği örnekler

### Rodofit ne anlama gelmektedir?

- Altın: kırmızı alg kloroplast soyu
- Jev + Qwen (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Jev + Gemini (F1 0.727): Rodofit, kırmızı alg kloroplast soyu anlamına gelmektedir [xquad-tr-4a718c2591c6].
- Fark: **+0.727**

### Gelir eşitsizliğinin büyümeye zarar verdiği görüşüne dönmenin ana nedeni neydi?

- Altın: beşerî sermayenin kalkınmadaki artan önemidir
- Jev + Qwen (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Jev + Gemini (F1 0.714): Bu değişimin temel nedeni, beşerî sermayenin kalkınmadaki artan önemidir [xquad-tr-63a31756327f].
- Fark: **+0.714**

### Hangi rapor doğru tarihi içeriyordu?

- Altın: "Küresel ve Bölgesel Ölçekte geçmişte ve günümüzde Kar ve Buz Farklılıkları"
- Jev + Qwen (F1 0.000): WWF raporu, 2035 tarihini IPCC tarafından doğru şekilde alıntılamıştır [xquad-tr-5550c7743e65].
- Jev + Gemini (F1 0.710): Doğru tarihi içeren kaynak, bir ICSI raporu olan "Küresel ve Bölgesel Ölçekte geçmişte ve günümüzde Kar ve Buz Farklılıkları" raporuydu [xquad-tr-5550c7743e65].
- Fark: **+0.710**

### 1970'lerde oynanan oyunun adı neydi?

- Altın: Seven Keys to Doomsday'da Dalekler ve Doktor Kim'de
- Jev + Qwen (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Jev + Gemini (F1 0.700): 1970'lerin başında sahnelenen oyun Seven Keys to Doomsday'da Dalekler ve Doktor Kim'dir [xquad-tr-138af7162b1d].
- Fark: **+0.700**

### Fransız Protestanları, Yeni Dünyadaki dini inançlarını nasıl değiştirdi?

- Altın: daha fazla sayıda üyesi olan diğer Protestan mezheplerine katıldı
- Jev + Qwen (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Jev + Gemini (F1 0.667): Kuzey Amerika'daki Fransız Protestan cemaatlerinin veya bireylerinin çoğu, nihayetinde daha fazla sayıda üyesi olan diğer Protestan mezheplerine katıldı [xquad-tr-dd26c002354e].
- Fark: **+0.667**

### Kloroplastidan ne anlama gelmektedir?

- Altın: yeşil kloroplast soyu
- Jev + Qwen (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Jev + Gemini (F1 0.667): Kloroplastidan, yeşil kloroplast soyu anlamına gelmektedir [xquad-tr-4a718c2591c6].
- Fark: **+0.667**

### John Sheepshanks’in müze için önemli vasiyetinin hedefi neydi?

- Altın: “Bir Ulusal İngiliz Sanat Galerisi” oluşturmak
- Jev + Qwen (F1 0.000): John Sheepshanks’in vasiyeti, müzeye Constable'ın eserlerinin gelmesini sağlamayı hedeflemiştir [xquad-tr-655fdb0366c0].
- Jev + Gemini (F1 0.625): John Sheepshanks'in bağışının hedefi, “Bir Ulusal İngiliz Sanat Galerisi” oluşturmaktı [xquad-tr-655fdb0366c0].
- Fark: **+0.625**

### Danışmanlık yapan eczacıların artarak hastalarla doğrudan çalışmasının ana nedeni nedir?

- Altın: birçok yaşlı insan günümüzde çok sayıda ilaç kullanıyor, ancak kurumsal ortamların
- Jev + Qwen (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Jev + Gemini (F1 0.621): Bunun ana nedeni, günümüzde birçok yaşlı insanın çok sayıda ilaç kullanması ancak kurumsal ortamların dışında yaşamaya devam etmesidir [xquad-tr-3323eafeea1c].
- Fark: **+0.621**


## Gemini'nin en fazla kötüleştirdiği örnekler

### Çin'de, bu kişi toprağın dağların aşınması ve alüvyon birikimi ile oluştuğu sonucuna varmıştır, ismi nedir?

- Altın: Shen Kuo
- Jev + Qwen (F1 1.000): Shen Kuo [xquad-tr-36478d7e8467].
- Jev + Gemini (F1 0.286): Bu kişi bilge Shen Kuo'dur [xquad-tr-36478d7e8467].
- Fark: **-0.714**

### Ren Vadisi yakınındaki Orta Çağ’dan kalma kaleleri bulunan bölgeye ne ad verilir?

- Altın: Romantik Ren
- Jev + Qwen (F1 0.500): Bu bölgeye "Romantik Ren" adı verilir [xquad-tr-752ffa1f6ada].
- Jev + Gemini (F1 0.000): Orta Çağ’dan kalan 40’dan fazla kale ve hisara ev sahipliği yapan bu bölge “Romantik Ren” olarak bilinmektedir [xquad-tr-752ffa1f6ada].
- Fark: **-0.500**

### Hangi kamu politikası okulu kendi evini Ludiwig Mies van der Rohe'nin tasarladığı binada kurmuştur?

- Altın: Harris Kamu Politikası Çalışmaları Okulu'nun
- Jev + Qwen (F1 0.476): Ludwig Mies van der Rohe tarafından tasarlanan bina, Harris Kamu Politikası Çalışmaları Okulu'nun yuvası haline gelmiştir [xquad-tr-7f444e38fda5].
- Jev + Gemini (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Fark: **-0.476**

### Bazı büyük eczacılık yönetim şirketleri nelerdir?

- Altın: Omnicare, Kindred Healthcare ve PharMerica
- Jev + Qwen (F1 1.000): Omnicare, Kindred Healthcare ve PharMerica [xquad-tr-3323eafeea1c].
- Jev + Gemini (F1 0.588): Büyük eczacılık yönetim şirketleri arasında öncelikle Omnicare, Kindred Healthcare ve PharMerica bulunmaktadır [xquad-tr-3323eafeea1c].
- Fark: **-0.412**

### IPCC'nin ilk başkanı kimdi?

- Altın: Bert Bolin
- Jev + Qwen (F1 0.400): IPCC'nin ilk başkanı 1988'de seçilen Bert Bolin idi [xquad-tr-c0272f2104fa].
- Jev + Gemini (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Fark: **-0.400**

### Amazon'da tipik bir yılda kaç ton karbon emilir?

- Altın: 1,5 gigaton
- Jev + Qwen (F1 0.400): Tipik bir yılda Amazon 1,5 gigaton karbondioksit emmektedir [xquad-tr-102947b53d99].
- Jev + Gemini (F1 0.000): Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.
- Fark: **-0.400**

### Jacksonville'deki yeni bina projelerinden kim sorumluydu?

- Altın: Belediye Başkanı W. Haydon Burns
- Jev + Qwen (F1 0.667): Jacksonville'deki yeni bina projelerinden Belediye Başkanı W. Haydon Burns sorumluydu [xquad-tr-4a841f83d10a].
- Jev + Gemini (F1 0.276): Jacksonville'deki yeni kamu bina projelerini finanse etmek için harcamaları Jacksonville şehri hükümeti arttırmış ve Belediye Başkanı W. Haydon Burns'ün dönemi bu projelerin inşasıyla sonuçlanmıştır [xquad-tr-4a841f83d10a].
- Fark: **-0.391**

### Jacksonville hangi vilayete bağlı?

- Altın: Duval Vilayetinin
- Jev + Qwen (F1 0.571): Jacksonville, Duval Vilayetinin vilayet merkezidir [xquad-tr-95526a3d3722].
- Jev + Gemini (F1 0.200): Jacksonville, Duval Vilayeti'ne bağlıdır ve bu vilayetin merkezidir [xquad-tr-95526a3d3722].
- Fark: **-0.371**


## Yorumlama

- D ve G aynı sorgu, aynı adaylar, aynı Jev sırası ve aynı beş bağlamı kullanır; üretici karşılaştırması kontrollüdür.
- Gemini medium gizli reasoning tokenları kullanır; görünen cevap kısa olsa da completion maliyeti bunları içerir.
- XQuAD kısa altın cevapları nedeniyle EM kaynaklı tam cümleleri sert cezalandırır; F1, başarı ve yanlış cevap oranı birlikte okunmalıdır.
