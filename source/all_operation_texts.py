operation_texts = [
r"İşlem Seç",
r"""Gri Dönüşüm

Bu işlem, renkli bir görüntüyü gri tonlamalı bir görüntüye dönüştürür.

Her pikselin kırmızı, yeşil ve mavi kanallarının ağırlıklı toplamını hesaplayarak bunu yapar. Kullanılan formül:

    `0.2989 * R + 0.587 * G + 0.114 * B`.
""",
r"""İkili Dönüşüm

Bu işlem, gri tonlamalı bir görüntüyü ikili (siyah-beyaz) bir görüntüye dönüştürür. Bunu, her pikselin yoğunluğunu bir eşik değeriyle karşılaştırarak yapar.

Yoğunluk eşik değerine eşit veya büyükse piksel beyaz (255) yapılır; değilse siyah (0) yapılır.
""",
r"""Görüntü Döndürme

Bu işlem, bir görüntüyü istenilen herhangi bir açıyla döndürmemize olanak tanır.

bounding_box
Görüntünün yükseklik, genişlik ve döndürme açısı değerleri kullanılarak çerçevenin boyutu hesaplanır ve bu boyuta göre sıfırlarla dolu bir matris oluşturulur:

    new_w = ceil(w * abs(sin(rad)) + h * abs(cos(rad)))
    new_h = ceil(h * abs(sin(rad)) + w * abs(cos(rad)))

place_image_in_bounding_box
Görüntünün yükseklik ve genişlik bilgileri ile çerçevenin boyutları kullanılarak çerçevenin merkez noktası hesaplanır. Daha sonra, çerçevenin merkez noktasından görüntünün yükseklik ve genişlik değerleri çıkarılarak görüntünün çerçeve içindeki sol üst pikselinin koordinatları elde edilir. Bu pikselden başlayarak görüntü çerçeve içine piksel piksel yerleştirilir.

rotate_image
Her piksel, görüntünün çerçeve içine yerleştirilmesiyle elde edilen merkez koordinatları kullanılarak döndürülür:

    x' = x * cos(angle) - y * sin(angle)
    y' = x * sin(angle) + y * cos(angle) 

Not: new_x ve new_y döndürme matris formülü, koordinat sistemindeki (0,0) orijin noktasına göre döndürme yapar. (0,0) noktasını elde etmek için, görüntünün çerçeve içine yerleştirilmesiyle elde edilen merkez noktası, i ve j’den x, y değerleri çıkarılarak döndürülür. Daha sonra görüntüyü çerçevenin ortasına getirmek için, çerçeveye yerleştirilen görüntünün merkez koordinatları eklenir.

Döndürülmüş pikselin görüntüdeki üst sol (x0, y0) ve alt sağ (x1, y1) koordinatları bulunur. Daha sonra bu görüntünün etrafındaki 4 pikselin (a, b, c, d) ağırlıkları hesaplanır:

    (x0, y0) pikseli için ağırlık c * d
    (x1, y0) pikseli için ağırlık a * d
    (x0, y1) pikseli için ağırlık c * b
    (x1, y1) pikseli için ağırlık a * b

Daha sonra bu 4 çevre pikselin ağırlıkları kullanılarak her pikselin RGB değeri Bilineer Enterpolasyon yöntemiyle hesaplanır.

    f(x, y) ≈ (1 - a)(1 - b) f(x0, y0) + a(1 - b) f(x1, y0) + (1 - a)b f(x0, y1) + ab f(x1, y1)
""",
r"""Görüntü Kırpma

Bu işlem, bir görüntünün istenilen iki koordinat arasındaki kısmını kırpmak için kullanılır.

Bu işlem, iki koordinattan birini sol üst köşe ve diğerini sağ alt köşe kabul ederek bir dörtgen çizer ve bu kısmı döndürür.
""",
r"""Görüntü Yakınlaştırma

Bu işlem, görüntünün En Yakın Komşu Enterpolasyonu ile yakınlaştırılmasını sağlar.
""",
r"""Görüntü Uzaklaştırma

Bu işlem, görüntünün En Yakın Komşu Enterpolasyonu ile uzaklaştırılmasını sağlar.
""",
r"""RGB'den NTSC Dönüşüm

Bu işlem, RGB renk uzayındaki bir görüntüyü NTSC (National Television System Committee) renk uzayına dönüştürür.
""",
r"""RGB'den YCbCr Dönüşüm

Bu işlem, RGB renk uzayındaki bir görüntüyü YCbCr (Luminance, Chrominance) renk uzayına dönüştürür.
""",
r"""Histogram Germe

Bu işlem, görüntüdeki piksel yoğunluklarını belirli bir aralığa gererek kontrastı arttırmayı amaçlar.

İlk olarak, görüntüdeki minimum ve maksimum yoğunluk değerleri (c ve d) bulunur. Daha sonra, bu değerler arasındaki yoğunlukları, belirtilen aralığa (genellikle 0 ile 255 arasında) yerleştirmek için bir dönüşüm yapılır.

Formül şu şekildedir:

    P_{çıkış} = (P_{giriş} - c) * ((b - a) / (d - c)) + a

Burada:
- P_{giriş}: Giriş piksel değeri,
- P_{çıkış}: Çıkış piksel değeri,
- c ve d: Görüntüdeki minimum ve maksimum yoğunluk değerleri,
- a ve b: Çıkış aralığının minimum ve maksimum değerleri (genellikle 0 ve 255).

Sonuçta, kontrastı artırılmış ve daha geniş bir yoğunluk aralığına yayılmış bir görüntü elde edilir.
""",
r"""Histogram Genişletme

Bu işlem, görüntüdeki piksel değerlerini mevcut minimum ve maksimum yoğunluk aralığından alarak, tüm gri ton aralığına (0 ile 255 arası) yayar.

Amaç, görüntünün kontrastını artırmak ve düşük dinamik aralığa sahip görüntülerde detayları daha görünür hale getirmektir.

İşlem şu adımları izler:
- Görüntü gri tonlamaya dönüştürülür (eğer değilse),
- Minimum (min_val) ve maksimum (max_val) piksel değerleri hesaplanır,
- Her piksel değeri, şu formül ile yeniden ölçeklenir:

    P_{çıkış} = ((P_{giriş} - min_val) / (max_val - min_val)) * 255

Burada:
- P_{giriş}: Orijinal piksel değeri,
- P_{çıkış}: Yeni (genişletilmiş) piksel değeri.

Sonuç olarak, tüm piksel değerleri 0 ile 255 arasına yayılır ve daha canlı, kontrastlı bir görüntü elde edilir.
""",
r"""Aritmetik İşlemler Toplama

bilgiler...
""",
r"""Aritmetik İşlemler Bölme

bilgiler...
""",
r"""Kontrast Artırma/Azaltma

bilgiler...
""",
r"""Konvolüsyon İşlemi Mean

bilgiler...
""",
r"""Eşikleme (Thresholding)

bilgiler...
""",
r"""Kenar Tespiti Prewitt

bilgiler...
""",
r"""Gürültü Ekleme (Salt & Pepper)

bilgiler...
""",
r"""Filtre (Mean)

bilgiler...
""",
r"""Filtre (Median)

bilgiler...
""",
r"""Filtre (Unsharp)

bilgiler...
""",
r"""Morfolojik İşlemler Genişleme

bilgiler...
""",
r"""Morfolojik İşlemler Aşınma

bilgiler...
""",
r"""Morfolojik İşlemler Açma

bilgiler...
""",
r"""Morfolojik İşlemler Kapama

bilgiler...
""",
]