# Bilgisayar Ağları - Rota Optimizasyon Projesi

BSM307/317 dönem projesi kapsamında üç farklı rota bulma algoritmasının (Genetik Algoritma, Karınca Kolonisi Optimizasyonu, Q-Learning) aynı ağ topolojisi üzerinde performans karşılaştırması yapılmaktadır.

## 📋 Proje Özeti

Bu proje, ağ topolojisinde optimal rota bulma problemini çözmek için üç farklı algoritmanın etkinliğini karşılaştırır:
- **Genetik Algoritma (GA)**: Evrimsel optimizasyon yaklaşımı
- **Karınca Kolonisi Optimizasyonu (ACO)**: Doğa esinlenmeli sürü zekası algoritması  
- **Q-Learning**: Pekiştirmeli öğrenme tabanlı yaklaşım

## 🛠️ Sistem Gereksinimleri

### Python Sürümü
- **Python 3.10+** (önerilen)

### Gerekli Kütüphaneler
```bash
pip install pandas networkx matplotlib numpy PySide6
```

### Veri Dosyaları
Proje kökünde aşağıdaki CSV dosyaları bulunmalıdır:
- `BSM307_317_Guz2025_TermProject_NodeData.csv` - Düğüm verileri
- `BSM307_317_Guz2025_TermProject_EdgeData.csv` - Bağlantı verileri  
- `BSM307_317_Guz2025_TermProject_DemandData.csv` - Talep verileri

## 🚀 Kurulum ve Çalıştırma

### 1. Projeyi İndirin
```bash
git clone <repository-url>
cd bilgisayarAglari
```

### 2. Bağımlılıkları Yükleyin
```bash
# Tüm gerekli kütüphaneleri yükle
pip install pandas networkx matplotlib numpy PySide6

# Alternatif olarak virtual environment kullanın
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# veya
.venv\Scripts\activate     # Windows

pip install pandas networkx matplotlib numpy PySide6
```

### 3. Veri Dosyalarını Kontrol Edin
Proje kökünde CSV dosyalarının bulunduğundan emin olun.

## 🎮 Kullanım Seçenekleri

### A) Grafik Arayüz (GUI) - Önerilen
```bash
python main_gui.py
```
- Kullanıcı dostu arayüz
- Algoritma parametrelerini kolayca ayarlama
- Sonuçları görsel olarak karşılaştırma
- Gerçek zamanlı performans izleme

### B) Komut Satırı - Tek Algoritma
```bash
# Genetik Algoritma
python genetik_proje.py

# Karınca Kolonisi Optimizasyonu  
python karinca.py

# Q-Learning
python q_learning.py
```

### C) Otomatik Deney Düzeneği - Toplu Test
```bash
python deney_duzenegi.py --demands 20 --repeats 5 --algorithms ga aco qlearning --weights 0.4 0.4 0.2
```

## 🧪 Deney Düzeneği (deney_duzenegi.py)

Toplu deneyleri otomatikleştirmek için geliştirilmiş kapsamlı test sistemi.

### Temel Kullanım
```bash
python deney_duzenegi.py \
  --demands 20 \
  --repeats 5 \
  --algorithms ga aco qlearning \
  --weights 0.4 0.4 0.2 \
  --seed 42
```

### Parametreler

#### Temel Parametreler
- `--demands N`: Demand CSV'den kaç satır okunacağı (varsayılan: 10)
- `--demand-offset N`: Başlangıç satır indeksi (varsayılan: 0)
- `--repeats N`: Her algoritma için tekrar sayısı (min 5 önerilir)
- `--algorithms`: Çalıştırılacak algoritmalar (`ga`, `aco`, `qlearning`)
- `--weights W1 W2 W3`: Gecikme/Güvenilirlik/Kaynak ağırlıkları
- `--seed N`: Tekrarlanabilirlik için rastgele tohum değeri

#### Algoritma Özel Parametreleri
```bash
# Genetik Algoritma
--ga-pop 50          # Popülasyon boyutu
--ga-gen 100         # Nesil sayısı
--ga-mut 0.1         # Mutasyon oranı
--ga-cross 0.8       # Çaprazlama oranı

# Karınca Kolonisi
--aco-ants 30        # Karınca sayısı
--aco-iter 50        # İterasyon sayısı
--aco-alpha 1.0      # Feromon ağırlığı
--aco-beta 2.0       # Sezgisel bilgi ağırlığı
--aco-evap 0.5       # Buharlaşma oranı

# Q-Learning
--ql-episodes 1000   # Eğitim bölümü sayısı
--ql-alpha 0.1       # Öğrenme oranı
--ql-gamma 0.9       # İndirim faktörü
--ql-epsilon 0.1     # Keşif oranı
```

#### Çıktı Kontrolü
```bash
--output rapor.txt   # Özel rapor dosyası adı
--parallel           # Paralel işlem (hızlandırma)
```

### Örnek Komutlar

#### Hızlı Test
```bash
python deney_duzenegi.py --demands 5 --repeats 3 --algorithms ga aco
```

#### Kapsamlı Analiz
```bash
python deney_duzenegi.py \
  --demands 50 \
  --repeats 10 \
  --algorithms ga aco qlearning \
  --weights 0.3 0.4 0.3 \
  --seed 42 \
  --parallel \
  --output kapsamli_analiz.txt
```

#### Algoritma Karşılaştırması
```bash
python deney_duzenegi.py \
  --demands 20 \
  --repeats 7 \
  --algorithms ga aco qlearning \
  --ga-pop 100 --ga-gen 200 \
  --aco-ants 50 --aco-iter 100 \
  --ql-episodes 2000
```

## 🔄 Tekrarlanabilirlik (Seed Bilgisi)

Tüm algoritmalar Python'un `random` modülünü kullanır. Aynı seed değeriyle çalıştırıldığında sonuçlar tamamen tekrarlanabilir olur.

### Seed Kullanımı
```bash
# Sabit sonuçlar için
python deney_duzenegi.py --seed 42 --demands 20 --repeats 5

# Farklı seed değerleri
python deney_duzenegi.py --seed 123 --demands 20 --repeats 5
python deney_duzenegi.py --seed 456 --demands 20 --repeats 5
```

### Seed Olmadan
```bash
# Her çalıştırmada farklı sonuçlar
python deney_duzenegi.py --demands 20 --repeats 5
```

**Not**: Seed verilmezse sistem saatine göre rastgele değerler üretilir. Q-Learning için ek numpy rastgeleliği de aynı seed ile kontrol edilir.

## 📊 Rapor Analizi

### Rapor Dosyası Formatı
Deney düzeneği `deney_detay_YYYYMMDD_HHMMSS.txt` formatında detaylı rapor üretir.

### Rapor İçeriği
Her deney kombinasyonu için:
- **Başlangıç/Hedef**: Kaynak ve hedef düğümler
- **Talep**: İstenen bant genişliği
- **Algoritma Performansı**: 
  - Başarı oranı
  - Ortalama çalışma süresi
  - Ortalama maliyet
  - Standart sapma
  - En iyi/kötü sonuçlar
- **QoS Metrikleri**:
  - Gecikme (ms)
  - Güvenilirlik
  - Darboğaz bant genişliği
  - Toplam maliyet
- **Hata Analizi**: Başarısız tekrarların gerekçeleri

### Rapor Örneği
```
=== DENEY: Kaynak=15, Hedef=89, Talep=45.2 Mbps ===

Genetik Algoritma:
  ✓ Başarı: 8/10 (%80)
  ⏱ Ortalama Süre: 2.34s (±0.45s)
  💰 Ortalama Maliyet: 156.78 (±23.45)
  🏆 En İyi: 134.56, En Kötü: 189.23

Karınca Kolonisi:
  ✓ Başarı: 9/10 (%90)  
  ⏱ Ortalama Süre: 1.87s (±0.32s)
  💰 Ortalama Maliyet: 142.33 (±18.67)
  🏆 En İyi: 128.45, En Kötü: 167.89
```

## 📁 Proje Yapısı

```
bilgisayarAglari/
├── 📄 README.md                    # Bu dosya
├── 🐍 ag.py                        # Ağ topolojisi oluşturucu
├── 🧬 genetik_proje.py             # Genetik Algoritma
├── 🐜 karinca.py                   # Karınca Kolonisi Optimizasyonu
├── 🧠 q_learning.py                # Q-Learning Algoritması
├── 🔬 deney_duzenegi.py            # Otomatik deney sistemi
├── 🖥️ main_gui.py                  # Grafik kullanıcı arayüzü
├── 📊 BSM307_317_Guz2025_TermProject_NodeData.csv
├── 📊 BSM307_317_Guz2025_TermProject_EdgeData.csv
├── 📊 BSM307_317_Guz2025_TermProject_DemandData.csv
├── 📁 data/                        # Ek veri dosyaları
├── 📁 rapor/                       # Rapor çıktıları
└── 📁 __pycache__/                 # Python cache dosyaları
```

## 🎯 Algoritma Detayları

### Genetik Algoritma (genetik_proje.py)
- **Popülasyon tabanlı**: Çoklu çözüm adayı
- **Evrimsel operatörler**: Seçim, çaprazlama, mutasyon
- **Elitizm**: En iyi bireyleri koruma
- **Parametreler**: Popülasyon boyutu, nesil sayısı, mutasyon/çaprazlama oranları

### Karınca Kolonisi (karinca.py)  
- **Feromon tabanlı**: Deneyim paylaşımı
- **Sezgisel bilgi**: Yerel optimizasyon
- **Dinamik arama**: Keşif/sömürü dengesi
- **Parametreler**: Karınca sayısı, feromon ağırlıkları, buharlaşma oranı

### Q-Learning (q_learning.py)
- **Pekiştirmeli öğrenme**: Deneme-yanılma
- **Q-tablosu**: Durum-eylem değerleri
- **Epsilon-greedy**: Keşif stratejisi
- **Parametreler**: Öğrenme oranı, indirim faktörü, keşif oranı

## 🔧 Sorun Giderme

### Yaygın Hatalar

#### 1. Modül Bulunamadı
```bash
ModuleNotFoundError: No module named 'pandas'
```
**Çözüm**: Gerekli kütüphaneleri yükleyin
```bash
pip install pandas networkx matplotlib numpy PySide6
```

#### 2. CSV Dosyası Bulunamadı
```bash
FileNotFoundError: BSM307_317_Guz2025_TermProject_NodeData.csv
```
**Çözüm**: CSV dosyalarının proje kökünde olduğundan emin olun

#### 3. GUI Açılmıyor
```bash
ImportError: No module named 'PySide6'
```
**Çözüm**: PySide6 yükleyin
```bash
pip install PySide6
```

#### 4. Performans Sorunları
- Büyük ağlarda algoritma parametrelerini azaltın
- `--parallel` seçeneğini kullanın
- Demand sayısını düşürün

### Performans İpuçları

1. **Hızlı Test**: Az demand, az tekrar
2. **Kapsamlı Analiz**: Paralel işlem kullanın
3. **Bellek Optimizasyonu**: Büyük ağlarda parametreleri ayarlayın
4. **Sonuç Karşılaştırması**: Aynı seed değerini kullanın

## 📈 Sonuç Değerlendirmesi

### Başarı Kriterleri
- **Bant Genişliği**: Talep edilen değeri karşılama
- **Gecikme**: Minimum toplam gecikme
- **Güvenilirlik**: Yüksek güvenilirlik değeri
- **Kaynak Kullanımı**: Optimal kaynak tüketimi

### Karşılaştırma Metrikleri
- **Başarı Oranı**: Geçerli çözüm bulma yüzdesi
- **Çalışma Süresi**: Algoritma performansı
- **Çözüm Kalitesi**: Maliyet fonksiyonu değeri
- **Kararlılık**: Standart sapma değerleri

## 👥 Katkıda Bulunma

Bu proje BSM307/317 dönem projesi kapsamında geliştirilmiştir. Katkılar ve öneriler için:

1. Projeyi fork edin
2. Yeni özellik dalı oluşturun (`git checkout -b yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Dalınızı push edin (`git push origin yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir. Akademik kullanım için serbesttir.

---

**Son Güncelleme**: 31 Aralık 2025  
**Proje Durumu**: Aktif Geliştirme  
**Python Sürümü**: 3.10+