# bilgisayarAglari

BSM307 / 317 dönem projesi için üç farklı rota bulma algoritmasının (Genetik Algoritma, Karınca Kolonisi Optimizasyonu, Q-Learning) aynı ağ topolojisi üzerinde kıyaslanması ve raporlanması bu depo üzerinden gerçekleştiriliyor. `ag.py` dosyası CSV’deki düğüm/kenar verilerini okuyup `networkx` grafını oluşturur; diğer dosyalar algoritmaları ve otomasyon araçlarını içerir.

## Gerekli Araçlar
- Python 3.10+ (standart kütüphaneler + `pandas`, `networkx`, `matplotlib`)
- Proje kökünde sağlanan `BSM307_317_Guz2025_TermProject_*.csv` dosyaları

Virtualenv ihtiyacı yoksa doğrudan sistem Python’ı ile de çalıştırabilirsiniz. Eksik paket olursa:

```bash
python3 -m pip install pandas networkx matplotlib
```

## Deney Düzeneği (deney_duzenegi.py)
Toplu deneyleri otomatikleştirmek için `deney_duzenegi.py` betiğini kullanın. Betik `BSM307_317_Guz2025_TermProject_DemandData.csv` içindeki satırları `(S,D,B)` kombinasyonlarına çevirir, her kombinasyon için seçtiğiniz algoritmaları çalıştırır ve sonuçları zaman damgalı bir rapora yazar.

### Örnek Çalıştırma
```bash
python3 deney_duzenegi.py \
  --demands 20 \
  --repeats 5 \
  --algorithms ga aco qlearning \
  --weights 0.4 0.4 0.2
```

Komut parametreleri:
- `--demands` / `--demand-offset`: Demand CSV’den kaç satır ve hangi başlangıç indexinden okunacağını belirler.
- `--repeats`: Her algoritma için kaç kez tekrar yapılacağı (ortalama, std, en iyi/kötü için minimum 5 önerilir).
- `--algorithms`: `ga`, `aco`, `qlearning` anahtarlarından dilediğiniz alt küme.
- `--weights`: Gecikme / Güvenilirlik / Kaynak kullanım ağırlıkları (betik normalize eder).
- Algoritma özel parametreleri (`--ga-pop`, `--aco-ants`, `--ql-episodes` vb.) ayrıntılı ince ayar sağlar.
- `--output`: Varsayılan isim yerine özel rapor dosyası tanımlamak için.

Betik çıktısı `deney_detay_YYYYMMDD_HHMMSS.txt` formatında rapor üretir; başarılı ve başarısız her tekrarın metriklerini, süreleri ve gerekçelerini bu dosyada bulabilirsiniz.

## Seed (Tekrarlanabilirlik) Bilgisi
Tüm algoritmalar Python’un `random` modülünü kullanır. Aynı seed’le koşulduğunda betik aynı sırada aynı rastgele kararları vereceğinden sonuçlar tekrarlanabilir olur.

```bash
python3 deney_duzenegi.py --seed 42 --demands 20 --repeats 5
```

Seed verilmezse sistem saatine göre farklı sonuçlar üretilir. Q-Learning tarafında ek bir numpy rastgeleliği olmadığı için tek `--seed` parametresi yeterlidir; farklı seed değerleri algoritmaların iç çeşitliliğini değiştirir.

## Raporu Okuma
Rapor dosyasında her deney için aşağıdaki blok bulunur:
- Başlangıç / hedef düğümler ve talep edilen bant genişliği
- Algoritma bazında başarı sayısı, ortalama süre, ortalama maliyet, standart sapma, en iyi/kötü sonuçlar
- Geçerli yolların gecikme (ms), güvenilirlik, darboğaz bant genişliği ve maliyet bilgileri
- Talebi karşılamayan veya çökmüş tekrarların gerekçeleri

Son bölümde her algoritmanın kaç kombinasyonda en az bir geçerli çözüm üretebildiği özetlenir. Bu raporu doğrudan proje teslimine ekleyebilir veya Excel’e aktararak grafik oluşturabilirsiniz.
