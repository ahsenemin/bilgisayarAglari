import random
import math
import time
import matplotlib.pyplot as plt
import networkx as nx

# Proje için gerekli temel kütüphaneler:
# random: Genetik Algoritma'nın rastgelelik temelli işlemleri (mutasyon, populasyon başlangıcı) için.
# math: Güvenilirlik maliyetini hesaplarken doğal logaritma (math.log) kullanmak için.
# time: Algoritmanın çalışma süresini (performans) ölçmek için.
# matplotlib.pyplot, networkx: Ağ yapısı ve görselleştirme için.

# Arkadaşının hazırladığı 'ag.py' dosyasından oluşturulan Graf (G) nesnesini içe aktarır.
# Bu graf, tüm düğüm ve bağlantı özelliklerini (Gecikme, Güvenilirlik, Bant Genişliği) içerir.
from ag import G

# ==============================================================================
# 1. Genetik Algoritma Sınıfı (Meta-Sezgisel Çözücü)
# ==============================================================================
class GenetikAlgoritma:
    """QoS Odaklı Çok Amaçlı Rotalama Problemini çözen Meta-Sezgisel Algoritma."""
    
    def __init__(self, graf, kaynak, hedef, pop_size=100, mutasyon_orani=0.1, nesil=100, agirliklar=None):
        """Sınıf başlatıcısı. Algoritmanın başlangıç ayarlarını yapar."""
        self.graph = graf           # Ağ topolojisi
        self.kaynak = kaynak        # Başlangıç Düğümü (Source)
        self.hedef = hedef          # Bitiş Düğümü (Destination)
        self.pop_size = pop_size    # Popülasyon Büyüklüğü (Her nesildeki rota sayısı)
        self.mutation_rate = mutasyon_orani # Mutasyon yapma ihtimali (Örn: %10)
        self.generations = nesil    # Nesil Sayısı (Algoritmanın kaç döngü çalışacağı)
        
        # Proje Raporuna uygun Ağırlıklar: [Gecikme, Güvenilirlik, Kaynak]
        self.weights = agirliklar if agirliklar else [0.33, 0.33, 0.33]

    # --- HESAPLAMA FONKSİYONLARI (Proje Metrikleri) ---
    
    def calculate_path_delay(self, path):
        """Toplam Gecikmeyi hesaplar (Toplamsal metrik, Minimizasyon)."""
        total_delay = 0
        
        # Bağlantı (Link) Gecikmeleri
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            total_delay += self.graph[u][v].get('delay', 0)
            
        # Düğüm (Node) İşlem Süreleri (Kaynak ve Hedef hariç ara düğümler)
        for node in path[1:-1]:
            total_delay += self.graph.nodes[node].get('processing_delay', 0)
            
        return total_delay

    def calculate_path_reliability_cost(self, path):
        """Güvenilirlik Maliyeti: Maksimize edilmesi gereken güvenilirliği, minimize edilecek maliyete çevirir."""
        total_cost = 0
        
        # Formül: -log(Güvenilirlik) toplamı. Güvenilirlik azaldıkça maliyet artar.
        
        # Bağlantı Güvenilirliği Maliyeti
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            r = self.graph[u][v].get('reliability', 0.99)
            if r <= 0: r = 0.0001
            total_cost += -math.log(r)
            
        # Düğüm Güvenilirliği Maliyeti (Tüm düğümler dahil)
        for node in path:
            r = self.graph.nodes[node].get('reliability', 0.99)
            if r <= 0: r = 0.0001
            total_cost += -math.log(r)
            
        return total_cost

    def calculate_resource_usage(self, path):
        """Kaynak Kullanımı Maliyeti (1000/BW toplamı, Minimizasyon)."""
        total_resource = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            bw = self.graph[u][v].get('bandwidth', 100)
            if bw <= 0: bw = 1
            # Düşük bant genişliği = Yüksek maliyet
            total_resource += (1000.0 / bw)
        return total_resource

    def toplam_maliyet_hesapla(self, path):
        """Çok Amaçlı Maliyet Fonksiyonu (Weighted Sum Method)."""
        try:
            d = self.calculate_path_delay(path)
            r = self.calculate_path_reliability_cost(path)
            res = self.calculate_resource_usage(path)
            
            # Formül: Wd*D + Wr*R_maliyet*100 + Wc*C
            # Ağırlıklar ile çarpılıp toplanarak tek bir maliyet skoru elde edilir.
            return (self.weights[0] * d) + (self.weights[1] * r * 100) + (self.weights[2] * res)
        except:
            return float('inf') # Geçersiz rotaları eler

    def uygunluk(self, path):
        """Fitness Fonksiyonu: Maliyet ne kadar düşükse, uygunluk (puan) o kadar yüksektir."""
        cost = self.toplam_maliyet_hesapla(path)
        return 1.0 / (cost + 1e-9)

    # --- GENETİK ALGORİTMA OPERATÖRLERİ ---
    
    def rastgele_yol_bul(self):
        """Başlangıç popülasyonu için rastgele geçerli bir yol (kromozom) üretir."""
        try:
            path = [self.kaynak]
            curr = self.kaynak
            visited = {self.kaynak}
            while curr != self.hedef:
                neighbors = [n for n in self.graph.neighbors(curr) if n not in visited]
                if not neighbors: return None
                curr = random.choice(neighbors)
                path.append(curr)
                visited.add(curr)
                if len(path) > 50: return None
            return path
        except:
            return None

    def populasyon_olustur(self):
        """Belirlenen popülasyon büyüklüğüne ulaşana kadar rastgele yollar dener."""
        populasyon = []
        tries = 0
        while len(populasyon) < self.pop_size and tries < self.pop_size * 10:
            yol = self.rastgele_yol_bul()
            if yol: populasyon.append(yol)
            tries += 1
        return populasyon

    def caprazlama(self, p1, p2):
        """Çaprazlama (Crossover): İki iyi rotanın genlerini (düğüm dizilerini) birleştirir."""
        # Genetik Algoritmanın en iyi genleri birleştirme prensibi. 
        common = [n for n in p1 if n in p2 and n != self.kaynak and n != self.hedef]
        if not common: return p1
        
        node = random.choice(common) # Ortak düğüm seçilir
        idx1 = p1.index(node)
        idx2 = p2.index(node)
        
        # Yeni rota: P1'in başı + P2'nin sonu
        new_path = p1[:idx1] + p2[idx2:]
        
        # Döngü kontrolü: Rotada aynı düğüm tekrar kullanılmış mı?
        if len(new_path) == len(set(new_path)): return new_path
        return p1

    def mutasyon(self, path):
        """Mutasyon (Mutation): Rotanın bir kısmını rastgele değiştirerek çeşitliliği artırır."""
        # Algoritmanın yerel optimuma takılıp kalmasını engeller. 
        if random.random() < self.mutation_rate and len(path) > 2:
            try:
                cut_idx = random.randint(1, len(path)-2) # Rastgele kesme noktası
                node = path[cut_idx]
                
                # Bu noktadan hedefe doğru yeni, rastgele bir yol segmenti oluşturulur.
                curr = node
                new_segment = []
                visited = set(path[:cut_idx+1])
                # ... yeni segment oluşturma mantığı ...
                
                return path[:cut_idx+1] + new_segment # Mutasyona uğramış yeni yolu döndür
            except:
                pass
        return path

    def calistir(self):
        """Genetik Algoritma'nın Nesil Döngüsünü başlatır."""
        start_time = time.time()
        populasyon = self.populasyon_olustur()
        en_iyi_yol = None
        en_iyi_skor = float('inf')

        if not populasyon:
            return None, 0, 0

        print(f"🧬 Algoritma Çalışıyor... ({self.generations} Nesil hesaplanacak)")

        # Ana Nesil Döngüsü
        for i in range(self.generations):
            if not populasyon: break
            
            # Elitism için neslin en iyisini bul
            gen_best = max(populasyon, key=self.uygunluk)
            gen_cost = self.toplam_maliyet_hesapla(gen_best)
            
            # Genel en iyi çözümü güncelle
            if gen_cost < en_iyi_skor:
                en_iyi_skor = gen_cost
                en_iyi_yol = gen_best
            
            yeni_pop = []
            yeni_pop.append(gen_best) # Elitism: En iyi bireyi yeni nesle direk aktar
            
            # Yeni nesli üret
            while len(yeni_pop) < self.pop_size:
                parent1 = random.choice(populasyon)
                parent2 = random.choice(populasyon)
                
                child = self.caprazlama(parent1, parent2)
                child = self.mutasyon(child)
                yeni_pop.append(child)
                
            populasyon = yeni_pop

        end_time = time.time()
        sure = end_time - start_time
        return en_iyi_yol, en_iyi_skor, sure

# --- GÖRSELLEŞTİRME ---
def rotayi_ciz(graf, yol, kaynak, hedef):
    """Bulunan rotayı ağ grafiği üzerinde çizer."""
    if not yol: return
    print("🎨 Grafik çiziliyor, lütfen bekleyin...")
    # ... (Görselleştirme kodları) ...
    plt.show()

# --- ANA PROGRAM (Burada çalışır) ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   GENETİK ALGORİTMA ROTA BULUCU")
    print("="*50)
    
    try:
        # 1. Kullanıcıdan Girdi Alınır
        print("Lütfen 0 ile 249 arasında düğüm numaraları girin.")
        k = int(input("👉 Başlangıç Düğümü (Kaynak): "))
        h = int(input("👉 Bitiş Düğümü (Hedef): "))

        # Düğüm kontrolü
        if k not in G.nodes or h not in G.nodes:
            print("\n❌ HATA: Girdiğiniz düğüm numarası ağda yok!")
        else:
            # 2. Algoritmayı Başlat
            # GA parametreleri (Popülasyon 100, Nesil 200 olarak ayarlandı)
            agirliklar = [0.4, 0.4, 0.2] 
            POP_SIZE = 100
            GENERATIONS = 200 # Ana kod bloğundaki bu değerin kullanıldığını belirtmek için 100'den 200'e güncellendi
            
            ga = GenetikAlgoritma(G, k, h, pop_size=POP_SIZE, nesil=GENERATIONS, agirliklar=agirliklar)
            yol, maliyet, sure = ga.calistir()
            
            # 3. Sonuçları Yazdır
            if yol:
                print("\n" + "-"*30)
                print("✅ SONUÇ BULUNDU")
                print("-"*30)
                print(f"⏱️  Hesaplama Süresi: {sure:.4f} saniye")
                print(f"🛣️  Rota: {yol}")
                print(f"💰 Toplam Maliyet Skoru: {maliyet:.4f}")
                
                # Metrik Detaylarını Hesapla
                d = ga.calculate_path_delay(yol)
                r = ga.calculate_path_reliability_cost(yol)
                c = ga.calculate_resource_usage(yol)
                
                print("\n📊 Metrik Detayları:")
                print(f"   • Toplam Gecikme:      {d:.2f} ms")
                print(f"   • Güvenilirlik Maliyeti: {r:.4f}")
                print(f"   • Kaynak Kullanımı:    {c:.2f}")
                print("="*50)
                
                # 4. Grafiği Çiz
                rotayi_ciz(G, yol, k, h)
            else:
                print("\n❌ Rota bulunamadı.")

    except ValueError:
        print("\n❌ Lütfen geçerli bir sayı giriniz.")
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")