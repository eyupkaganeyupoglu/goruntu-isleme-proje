operation_texts = [
r"İşlem Seç",
r"""Gri Tonlamaya Dönüştürme
`image1`

Bu fonksiyon, RGB bir görüntüyü gri tonlamalı bir görüntüye çevirir.

Her pikselin kırmızı, yeşil ve mavi bileşenlerinden ağırlıklı ortalama alınarak gri ton değeri hesaplanır ve ilgili konuma atanır. Aşağıdaki formül kullanılır:

    Gri = 0.2989 * R + 0.587 * G + 0.114 * B
""",
r"""İkili (Binary) Görüntüye Dönüştürme
`image1`

Bu fonksiyon, gri tonlamalı bir görüntüyü eşikleme yöntemiyle siyah-beyaz (binary) formata çevirir.

Renkli görüntü önce gri tonlamaya dönüştürülür. Her pikselin değeri eşik (threshold) ile karşılaştırılır. Piksel değeri eşik değerinden büyük veya eşitse beyaz (255), küçükse siyah (0) olarak atanır.
""",
r"""Görüntüyü Döndürme
`image1`

Bu fonksiyon, verilen açıya göre bir görüntüyü döndürür.

İstenilen açıda döndürülmüş görüntünün sığabileceği bir bounding box (sınırlayıcı kutu) hesaplamak için aşağıdaki formül kullanılır:

    new_w = |w * cos(θ)| + |h * sin(θ)|
    new_h = |h * cos(θ)| + |w * sin(θ)|

Orijinal görüntü, bounding box'un merkezine yerleştirilir.

Her bir pikselin yeni konumu hesaplamak için aşağıdaki formül kullanılır:

    x' =  (x - x_center) * cos(theta) + (y - y_center) * sin(theta) + x_center
    y' = -(x - x_center) * sin(theta) + (y - y_center) * cos(theta) + y_center

theta: döndürme açısını radian cinsinden ifade eder.

Döndürme işlemiyle elde edilen yeni koordinatlar, en yakın alt ve üst tam sayılara yuvarlanarak görüntü sınırları içinde kalmaları sağlanır.

Her pikselin rengi, bu dört komşu pikselin değerleri kullanılarak, her iki eksendeki interpolasyon farklarına göre ağırlıklı biçimde hesaplanan bilinear interpolasyon yöntemiyle belirlenir.

    P = (1 - a) * (1 - b) * Q11 +
             a  * (1 - b) * Q21 +
        (1 - a) * b       * Q12 +
             a  * b       * Q22  

Burada:
- P: Yeni pikselin değeri
- Q11: Sol üst komşu pikselin değeri
- Q21: Sağ üst komşu pikselin değeri
- Q12: Sol alt komşu pikselin değeri
- Q22: Sağ alt komşu pikselin değeri
- a: x eksenindeki interpolasyon farkı
- b: y eksenindeki interpolasyon farkı
""",
r"""Görüntü Kırpma
`image1`

Bu fonksiyon, belirtilen koordinatlar arasında kalan dikdörtgen bölgeyi kırparak yeni bir görüntü oluşturur.
""",
r"""Yakınlaştırma (Zoom In)
`image1`

Bu fonksiyon, bir görüntüyü belirtilen oranda büyüterek yakınlaştırır.

Bu işlemi, Nearest Neighbor İnterpolasyon ile,  yeni bir piksel konumunu belirlerken, orijinal görüntüdeki en yakın komşu pikselin değerini kullanarak yapar.
""",
r"""Uzaklaştırma (Zoom Out)
`image1`

Bu fonksiyon, bir görüntüyü belirtilen oranda küçülterek uzaklaştırır.

Bu işlemi, Bilinear İnterpolasyon ile, yeni bir piksel konumunu belirlerken, orijinal görüntüdeki dört komşu pikselin değerlerinin ağırlıklı ortalamasını alarak yapar.

""",
r"""RGB'den NTSC'ye Dönüşüm
`image1`

Bu fonksiyon, RGB bir görüntüyü NTSC formatına çevirir.

Daha hassas hesaplamalar yapabilmek için görüntü float64 formatına dönüştürülür.

Görüntünün RGB değerleri ile NTSC dönüşüm matrisinin transpozesi çarpılarak NTSC değerleri hesaplanır. 

    NTSC dönüşüm matrisi:
        [0.299,  0.587,  0.114]
        [0.596, -0.275, -0.321]
        [0.212, -0.523,  0.311]
""",
r"""RGB'den YCbCr'ye Dönüşüm
`image1`

Bu fonksiyon, RGB bir görüntüyü YCbCr formatına çevirir.

Görüntü veri tipine bağlı olarak, YCbCr dönüşümüne uygulanacak delta değeri belirlenir:
- uint8 için delta 128,
- uint16 için delta 32768,
- Diğer veri türleri için delta 0.5 olarak belirlenir.

Y (parlaklık) değeri hesaplanır:

    Y = 0.299 * R + 0.587 * G + 0.114 * B

Kırmızı farkı (Cr) değeri hesaplanır:

    Cr = (R - Y) * 0.713 + delta

Mavi farkı (Cb) değeri hesaplanır:

    Cb = (B - Y) * 0.564 + delta

Sonuç görüntüsü, delta değerine göre uygun veri tipine dönüştürülür ve değerler, belirli bir aralığa sıkıştırılır.
""",
r"""Histogram Germe
`image2`

Bu fonksiyon, görüntüdeki parlaklık değerlerinin histogramını gererek daha geniş bir aralığa yayılmasını sağlar.

Eğer görüntü renkli ise, önce gri tonlamaya dönüştürülür. Ardından, gri tonlamalı görüntüdeki minimum ve maksimum piksel değerleri hesaplanır. Bu değerler, yeni bir aralığa (0 ile 255 arasında) dönüştürülür.

    Gri = (gray - c) * ((b - a) / (d - c)) + a

Burada:
- c: Görüntüdeki en küçük piksel değeri
- d: Görüntüdeki en büyük piksel değeri
- a: Yeni minimum değer (0)
- b: Yeni maksimum değer (255)

Sonuç olarak, kontrastı artırılmış bir görüntü elde edilir.
""",
r"""Histogram Genişletme
`image2`

Bu fonksiyon, görüntüdeki parlaklık değerlerinin histogramını genişleterek daha fazla çeşitlilik sağlar. Bu işlem, görüntünün kontrastını artırmak için kullanılır.

Eğer görüntü renkli ise, önce gri tonlamaya dönüştürülür. Ardından, gri tonlamalı görüntüdeki minimum ve maksimum piksel değerleri hesaplanır. Bu değerler kullanılarak her pikselin değeri 0 ile 255 arasına normalize edilir.

    Genişletilmiş = ((gray - min_val) / (max_val - min_val)) * 255

Burada:
- min_val: Görüntüdeki en küçük piksel değeri
- max_val: Görüntüdeki en büyük piksel değeri

Eğer maksimum ve minimum değerler aynıysa (görüntüdeki tüm pikseller aynı değere sahipse), dönüşüm yapılmaz ve orijinal görüntü döndürülür. 

Sonuç olarak, her piksel değeri 0 ile 255 arasına sıkıştırılarak kontrastı artırılmış bir görüntü elde edilir.
""",
r"""Aritmetik İşlemler Toplama (image3-tom, image3-jerry, All)
`image3-tom, image3-jerry, All`

Bu fonksiyon, iki görüntü arasındaki piksel değerlerini toplar ve sonuçları birleştirerek yeni bir görüntü oluşturur. İki görüntü de aynı boyutlarda olmalıdır.

İlk olarak, görüntülerin boyutlarının aynı olup olmadığı kontrol edilir. Eğer boyutlar farklıysa, bir hata mesajı döndürülür.

    Sonuç = (image1[i, j, c] + image2[i, j, c]) // 2

Burada:
- image1 ve image2: Aynı boyutta ve renk kanallarına sahip iki giriş görüntüsü
- i, j: Piksel koordinatları
-`c: Renk kanalı (kırmızı, yeşil, mavi)

Her pikselin renk kanalları arasındaki değerler toplanır ve toplamın ortalaması alınarak sonucun her piksel değeri hesaplanır.
""",
r"""Aritmetik İşlemler Bölme
`All`

Bu fonksiyon, iki görüntü arasındaki piksel değerlerini böler ve sonuçları birleştirerek yeni bir görüntü oluşturur. İki görüntü de aynı boyutlarda olmalıdır.

İlk olarak, görüntülerin boyutlarının aynı olup olmadığı kontrol edilir. Eğer boyutlar farklıysa, bir hata mesajı döndürülür.

Her piksel için, `image1[i, j, c]` değerinin `image2[i, j, c]` değerine bölümü yapılır. Eğer `image2`'deki değer 0 ise, bölme işleminden kaçınılır ve sonuç piksel değeri 255 olarak atanır (bölme hatası).

    Sonuç = min(image1[i, j, c] // image2[i, j, c], 255)

Burada:
- image1 ve image2: Aynı boyutta ve renk kanallarına sahip iki giriş görüntüsü
- i, j: Piksel koordinatları
- c: Renk kanalı (kırmızı, yeşil, mavi)
""",
r"""Kontrast Artırma/Azaltma
`image1, All`

Bu fonksiyon, bir görüntünün kontrastını artırmak ya da azaltmak için kullanılır. Kontrast, görüntüdeki parlaklık değerlerinin dağılımını kontrol eden bir parametredir. `alpha` ve `beta` parametreleri ile kontrast değiştirilebilir.

Fonksiyon, görüntüyü önce `float32` formatına dönüştürür, çünkü kontrast işlemi sırasında tam sayı değeri kayıpları olabilir. Daha sonra her bir pikselin değeri, aşağıdaki formüle göre dönüştürülür:

    Sonuç = alpha * (Piksel Değeri - 128) + 128 + beta

Burada:
- alpha: Kontrastı artıran bir çarpan.
- beta: Görüntünün parlaklığını değiştiren bir kaydırma faktörü.
- 128: RGB görüntülerinin ortalama parlaklık değeri olarak kabul edilen merkez değerdir.

Son olarak, hesaplanan değerler `0` ile `255` arasında sıkıştırılır ve `uint8` formatına dönüştürülür. Böylece, görüntüdeki her pikselin değeri geçerli renk aralığında kalır.
""",
r"""Konvolüsyon İşlemi Mean
`image1, image5`

Bu fonksiyon, bir görüntü üzerinde mean filter konvolüsyon işlemi uygular. Mean filter, görüntüdeki her bir pikselin değerini çevresindeki piksellerin ortalaması ile değiştirir. Bu, görüntüdeki gürültüyü azaltmaya ve detayları yumuşatmaya yardımcı olur.

Görüntü önce sınırları genişletilerek bir kenar yansıması (reflection) ile doldurulur. Böylece, görüntünün kenarlarında da konvolüsyon işlemi yapılabilir.

Her piksel için 3x3'lük bir pencere seçilir ve bu pencerenin ortalama değeri hesaplanır. Sonuç olarak, her bir pikselin yeni değeri, bu ortalama değer ile değiştirilir.

İşlem her renk kanalı için ayrı ayrı uygulanır.

Sonuç görüntüsü, uint8 formatına dönüştürülür ve 0 ile 255 arasındaki piksel değerleriyle döndürülür.
""",
r"""Eşikleme (Thresholding)
`image1`

Bu fonksiyon, gri tonlamalı bir görüntü üzerinde eşikleme (thresholding) işlemi yapar. Eşikleme, her pikselin değeri ile belirli bir eşik değeri karşılaştırılarak, belirli bir değerin üzerinde olan piksellerin beyaz (255) ve altında kalan piksellerin siyah (0) yapılması işlemidir. Bu, genellikle görüntüdeki önemli nesnelerin veya kenarların ayrılmasında kullanılır.
""",
r"""Kenar Tespiti Prewitt
`image1, image5`

Bu fonksiyon, Prewitt kenar algılama filtresi kullanarak bir görüntüdeki kenarları algılar. Prewitt filtresi, görüntüdeki yatay ve dikey kenarları belirlemek için kullanılan bir filtreleme tekniğidir. Bu işlem, görüntüyü yatay ve dikey yönde türevler alarak görüntüdeki keskin değişim noktalarını bulur.

Öncelikle, renkli bir görüntü gri tonlamaya dönüştürülür. Daha sonra, Prewitt operatörünü temsil eden iki kernel (filtre) kullanılır:
- `kernel_x`: Yatay kenarları algılamak için
- `kernel_y`: Dikey kenarları algılamak için

Bu filtreler, her pikselin çevresindeki 3x3'lik bölge üzerinde uygulanır. Elde edilen yatay ve dikey gradyanlar, her pikselin kenar gücünü oluşturur.

Prewitt Operatörü için kullanılan kernel'ler:

    kernel_x:
        [-1,  0,  1]
        [-1,  0,  1]
        [-1,  0,  1]

    kernel_y:
        [ 1,  1,  1]
        [ 0,  0,  0]
        [-1, -1, -1]

Her bir pikselin kenar gücü, bu iki kernel ile hesaplanır ve sonuç olarak, kenarların net bir şekilde ortaya çıktığı yeni bir görüntü elde edilir. 

Formül:

    gx = Σ(kernel_x * region)
    gy = Σ(kernel_y * region)
    gradient = min(√(gx^2 + gy^2), 255)

Burada:
- `gx` ve `gy`: Yatay ve dikey gradyanlar
- `region`: 3x3'lük pencere
- `gradient`: Hesaplanan kenar gücü
""",
r"""Gürültü Ekleme (Salt & Pepper)
`image1`

Bu fonksiyon, görüntüye tuz ve karabiber (Salt & Pepper) gürültüsü ekler. Tuz ve karabiber gürültüsü, görüntüde rastgele seçilen piksellere tamamen beyaz (tuz) veya siyah (karabiber) değerlerinin atanması ile oluşur.

Fonksiyon, aşağıdaki adımları takip eder:

1. Görüntünün boyutları belirlenir.
2. Ekleme işlemi yapılacak gürültü miktarı (amount) belirlenir. Burada, her bir pikselin tuz veya karabiber olarak değiştirilme olasılığı %5 (0.05) olarak ayarlanmıştır.
3. Görüntüye eklenmesi gereken tuz ve karabiber piksel sayıları hesaplanır:
    - num_salt: Tuz gürültüsü eklemek için gereken piksel sayısı
    - num_pepper: Karabiber gürültüsü eklemek için gereken piksel sayısı
4. Rastgele olarak tuz ve karabiber pikselleri, görüntü üzerinde seçilir ve ilgili piksellere sırasıyla beyaz (255, 255, 255) ve siyah (0, 0, 0) değerleri atanır.
""",
r"""Ortalama Filtreleme ile Gürültü Giderme
`image1`

Bu fonksiyon, tuz ve karabiber (salt and pepper) gürültüsü eklenmiş bir görüntüye, ortalama filtreleme (mean filtering) uygulayarak gürültüyü gidermeye çalışır.

1. İlk olarak, `add_noise_salt_and_pepper` fonksiyonu ile görüntüye tuz ve karabiber gürültüsü eklenir.
2. Görüntüye ortalama filtre uygulamak için, her pikselin çevresindeki 3x3 komşu piksellerin ortalaması alınır. 
3. Ancak, eğer piksellerin değeri 0 (karabiber) veya 255 (tuz) ise, bu piksellerin bulunduğu komşuluk dikkate alınmaz. Yani, sadece geçerli (0 ve 255 dışında) piksel değerleri ortalamaya katılır.
4. Eğer tüm komşuluk pikselleri 0 veya 255'ten oluşuyorsa, bu durumda komşuluktaki tüm piksellerin ortalaması kullanılır.

Bu işlem, görüntüdeki tuz ve karabiber gürültüsünü gidermeye yardımcı olabilir, çünkü gürültülü pikseller çevresindeki piksellerin ortalaması ile değiştirilir.
""",
r"""Medyan Filtreleme ile Gürültü Giderme
`image1`

Bu fonksiyon, tuz ve karabiber (salt and pepper) gürültüsü eklenmiş bir görüntüye, medyan filtreleme (median filtering) uygulayarak gürültüyü gidermeye çalışır.

1. İlk olarak, `add_noise_salt_and_pepper` fonksiyonu ile görüntüye tuz ve karabiber gürültüsü eklenir.
2. Görüntüde her pikselin çevresindeki 3x3 komşu piksellerin medyan değeri hesaplanır. 
3. Eğer bir pikselin değeri 0 (karabiber) veya 255 (tuz) ise, bu piksellerin bulunduğu komşuluk dikkate alınmaz. Yalnızca geçerli (0 ve 255 dışında) piksellerin medyanı alınır.
4. Eğer tüm komşuluk pikselleri 0 veya 255'ten oluşuyorsa, komşuluktaki tüm piksellerin medyanı hesaplanır.

Bu işlem, görüntüdeki tuz ve karabiber gürültüsünü gidermeye yardımcı olabilir, çünkü gürültülü pikseller çevresindeki piksellerin medyan değeri ile değiştirilir.
""",
r"""Filtre Unsharp 
`image5`

Bu fonksiyon, görüntüyü keskinleştirmek için Unsharp Masking yöntemini uygular.

Fonksiyon, aşağıdaki adımları takip eder:
1. Orijinal görüntü üzerinde bir bulanıklaştırma işlemi uygulanır. Bu amaçla, ortalama filtre kullanılır.
2. Orijinal görüntü ile bulanıklaştırılmış görüntü arasındaki fark (mask) hesaplanır. Bu fark, görüntüdeki ayrıntıları ve kenarları temsil eder.
3. Mask, orijinal görüntüye eklenir ve görüntünün keskinliği artırılır. `amount` parametresi, maskenin orijinal görüntüye ne kadar eklenmesi gerektiğini belirler.

Sonuç olarak, görüntüdeki kenarlar ve ayrıntılar daha belirgin hale gelir.
""",
r"""Morfolojik İşlemler Genişleme
`image4`

Bu fonksiyon, morfolojik genişleme (dilation) işlemi uygulayarak bir ikili (binary) görüntüyü genişletir.

Genişleme işlemi, ikili görüntüdeki her beyaz pikselin çevresindeki komşu pikselleri de beyaz yaparak görüntüyü genişletir.
""",
r"""Morfolojik İşlemler Aşınma
`image4`

Bu fonksiyon, morfolojik aşınma (erosion) işlemi uygulayarak bir ikili (binary) görüntüyü daraltır.

Aşınma işlemi, ikili görüntüdeki her siyah pikselin çevresindeki komşu pikselleri de siyah yaparak görüntüyü daraltır.
""",
r"""Morfolojik İşlemler Açma
`image4`

Bu fonksiyon, morfolojik açma (opening) işlemi uygulayarak bir ikili (binary) görüntüyü açar.

Açma işlemi, önce aşınma (erosion) işlemi ardından genişleme (dilation) işlemi uygular.
""",
r"""Morfolojik İşlemler Kapama
`image4`

Bu fonksiyon, morfolojik kapama (closing) işlemi uygulayarak bir ikili (binary) görüntüyü kapatır.

Kapama işlemi, önce genişleme (dilation) işlemi ardından aşınma (erosion) işlemi uygular.
""",
]