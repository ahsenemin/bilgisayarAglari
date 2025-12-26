import random  # Rastgelelik temelli işlemler (popülasyon üretimi, mutasyon) için.
import math    # Güvenilirlik maliyeti hesaplarken logaritma (math.log) kullanmak için.
import time    # Algoritmanın çalışma süresini (performans) ölçmek için.
import matplotlib.pyplot as plt # Sonucu görselleştirmek (grafik çizmek) için.
import networkx as nx           # Ağ topolojisini yönetmek ve çizmek için.

# Arkadaşının hazırladığı 'ag.py' dosyasından oluşturulan Graf (G) nesnesini içe aktarır.
# Bu graf düğümlerin koordinatlarını, bağlantıları ve (delay, reliability, bandwidth) gibi verileri tutar.
from ag import G

# ==============================================================================
# 1. Genetik Algoritma Sınıfı (Meta-Sezgisel Çözücü)
# ==============================================================================
class GenetikAlgoritma:
    """QoS (Hizmet Kalitesi) Odaklı Çok Amaçlı Rotalama Problemini çözen Evrimsel Algoritma."""
    
    def __init__(self, graf, kaynak, hedef, pop_size=100, mutasyon_orani=0.1, nesil=100, agirliklar=None):
        """
        BAŞLATICI: Algoritmanın genetik parametrelerini ve hedeflerini tanımlar.
        """
        self.graph = graf           # Üzerinde çalışılacak NetworkX graf nesnesi.
        self.kaynak = kaynak        # Başlangıç noktası (Kullanıcının seçtiği).
        self.hedef = hedef          # Bitiş noktası (Algoritmanın ulaşmaya çalıştığı sabit hedef).
        self.pop_size = pop_size    # Her bir nesilde kaç farklı yol (birey) deneneceği.
        self.mutation_rate = mutasyon_orani # Bir yolun rastgele değişime uğrama olasılığı (%10).
        self.generations = nesil    # Evrim sürecinin kaç döngü (nesil) devam edeceği.
        
        # Ağırlıklar: [Gecikme, Güvenilirlik, Kaynak] -> Hangi kriterin daha önemli olduğunu belirler.
        self.weights = agirliklar if agirliklar else [0.33, 0.33, 0.33]

    # --- HESAPLAMA FONKSİYONLARI (Fitness Metrikleri) ---
    
    def calculate_path_delay(self, path):
        """
        Gecikme (Delay): Toplamsal bir metriktir. Yol üzerindeki tüm bağlantıların ve 
        düğümlerin gecikme değerleri toplanır. Hedef: Minimizasyon.
        """
        total_delay = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            # İki düğüm arasındaki bağlantı gecikmesini al.
            total_delay += self.graph[u][v].get('delay', 0)
        # Yol üzerindeki ara düğümlerin işlem gecikmelerini (processing delay) ekle.
        for node in path[1:-1]:
            total_delay += self.graph.nodes[node].get('processing_delay', 0)
        return total_delay

    def calculate_path_reliability_cost(self, path):
        """
        Güvenilirlik (Reliability): Çarpımsal bir metriktir (R1 * R2 * ...).
        Ancak çarpımla uğraşmak yerine, her bir değerin negatif logaritmasını alarak 
        problemi toplamsal bir minimizasyon problemine çeviriyoruz: -log(R).
        Güvenilirlik 1'e ne kadar yakınsa, maliyet o kadar küçük çıkar.
        """
        total_cost = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            r = self.graph[u][v].get('reliability', 0.99)
            if r <= 0: r = 0.0001 # Logaritma hatası almamak için 0 kontrolü.
            total_cost += -math.log(r)
        # Düğümlerin de kendi güvenilirliklerini hesaba kat.
        for node in path:
            r = self.graph.nodes[node].get('reliability', 0.99)
            if r <= 0: r = 0.0001
            total_cost += -math.log(r)
        return total_cost

    def calculate_resource_usage(self, path):
        """
        Kaynak Kullanımı (BW): Bant genişliği düşük yolları cezalandırmak için 
        1000/Bandwidth formülü kullanılır. Düşük hız = Yüksek maliyet.
        """
        total_resource = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            bw = self.graph[u][v].get('bandwidth', 100)
            if bw <= 0: bw = 1
            total_resource += (1000.0 / bw)
        return total_resource

    def toplam_maliyet_hesapla(self, path):
        """
        Çok Amaçlı Maliyet: Gecikme, Güvenilirlik ve Kaynak değerlerini ağırlıklı olarak toplar.
        ÖNEMLİ: Eğer bir yol hedefte bitmiyorsa ona sonsuz (inf) maliyet vererek elenmesini sağlar.
        """
        try:
            # HEDEF KONTROLÜ: Yolun son elemanı kullanıcının girdiği hedef değilse geçersizdir.
            if not path or path[-1] != self.hedef:
                return float('inf') # Sonsuz maliyet = En kötü yol.
            
            d = self.calculate_path_delay(path)
            r = self.calculate_path_reliability_cost(path)
            res = self.calculate_resource_usage(path)
            
            # Formül: (W1 * Delay) + (W2 * Reliability_Cost * 100) + (W3 * Resource_Cost)
            return (self.weights[0] * d) + (self.weights[1] * r * 100) + (self.weights[2] * res)
        except:
            return float('inf') # Hata durumunda (bağlantı kopukluğu vb.) yolu ele.

    def uygunluk(self, path):
        """
        Fitness (Uygunluk): Genetik algoritma büyük değerleri sever. 
        Biz maliyeti düşürmeye çalıştığımız için 1 / Maliyet yaparak ters orantı kuruyoruz.
        """
        cost = self.toplam_maliyet_hesapla(path)
        return 1.0 / (cost + 1e-9) # 0'a bölünme hatasını engellemek için küçük bir pay eklenir.

    # --- GENETİK ALGORİTMA OPERATÖRLERİ ---
    
    def rastgele_yol_bul(self):
        """
        BAŞLANGIÇ POPÜLASYONU: Rastgele ama geçerli (komşuluk ilişkisine uygun) yollar üretir.
        Hedefe ulaşana kadar düğüm düğüm rastgele zıplar.
        """
        try:
            path = [self.kaynak]
            curr = self.kaynak
            visited = {self.kaynak}
            while curr != self.hedef:
                # Daha önce uğranmamış komşu düğümleri listele.
                neighbors = [n for n in self.graph.neighbors(curr) if n not in visited]
                if not neighbors: return None # Çıkmaz sokağa girerse yolu iptal et.
                
                curr = random.choice(neighbors) # Rastgele bir komşu seç.
                path.append(curr)
                visited.add(curr)
                if len(path) > 100: return None # Çok uzarsa sonsuz döngüyü engelle.
            return path
        except:
            return None

    def populasyon_olustur(self):
        """İstenen popülasyon boyutuna ulaşana kadar 'rastgele_yol_bul' fonksiyonunu çalıştırır."""
        populasyon = []
        tries = 0
        while len(populasyon) < self.pop_size and tries < self.pop_size * 20:
            yol = self.rastgele_yol_bul()
            if yol: populasyon.append(yol)
            tries += 1
        return populasyon

    def caprazlama(self, p1, p2):
        """
        Çaprazlama (Crossover): İki iyi ebeveyn yolun genlerini birleştirir.
        Eğer iki yolda da ortak bir düğüm varsa, o noktada yolları kesip birbirine bağlar.
        """
        # Kaynak ve hedef hariç ortak düğümleri bul.
        common = [n for n in p1 if n in p2 and n != self.kaynak and n != self.hedef]
        if not common: return p1 # Ortak nokta yoksa birinci ebeveyni koru.
        
        node = random.choice(common) # Birleşme noktası seç.
        idx1 = p1.index(node)
        idx2 = p2.index(node)
        
        # Yeni yol: P1'in başı + P2'nin sonu.
        new_path = p1[:idx1] + p2[idx2:]
        
        # Döngü kontrolü ve hedefe varış kontrolü.
        if len(new_path) == len(set(new_path)) and new_path[-1] == self.hedef:
            return new_path
        return p1

    def mutasyon(self, path):
        """
        Mutasyon (Mutation): Çeşitliliği sağlamak için yolu rastgele bir noktadan koparır 
        ve o noktadan hedefe yeni bir 'rastgele yürüyüş' yaparak bağlar.
        """
        if random.random() < self.mutation_rate and len(path) > 2:
            try:
                # Rastgele bir kopma (mutasyon) noktası seç.
                cut_idx = random.randint(1, len(path)-2)
                node = path[cut_idx]
                
                curr = node
                new_segment = []
                visited = set(path[:cut_idx+1])
                
                # Bu noktadan hedefe (self.hedef) kadar yeni bir yol parçası ör.
                for _ in range(50):
                    if curr == self.hedef: break
                    neighbors = [n for n in self.graph.neighbors(curr) if n not in visited]
                    if not neighbors: return path # Bağlantı kurulamazsa orijinal yolu koru.
                    
                    curr = random.choice(neighbors)
                    new_segment.append(curr)
                    visited.add(curr)
                
                # Sadece yeni parça hedefe ulaşıyorsa mutasyonu gerçekleştir.
                if new_segment and new_segment[-1] == self.hedef:
                    return path[:cut_idx+1] + new_segment
            except:
                pass
        return path

    def calistir(self):
        """
        GENETİK DÖNGÜ: Seçim, Çaprazlama ve Mutasyon işlemlerini nesiller boyu sürdürür.
        """
        start_time = time.time()
        populasyon = self.populasyon_olustur()
        en_iyi_yol = None
        en_iyi_skor = float('inf')

        if not populasyon: return None, 0, 0

        print(f"🧬 Genetik Algoritma Başlatıldı... (Hedef: {self.hedef})")

        for i in range(self.generations):
            if not populasyon: break
            
            # Mevcut popülasyondaki en düşük maliyetli (en iyi) yolu bul.
            gen_best = min(populasyon, key=self.toplam_maliyet_hesapla)
            gen_cost = self.toplam_maliyet_hesapla(gen_best)
            
            # Genel en iyi yolu güncelle.
            if gen_cost < en_iyi_skor:
                en_iyi_skor = gen_cost
                en_iyi_yol = gen_best
            
            # ELİTİZM: Neslin en iyi bireyini bir sonraki nesle doğrudan aktar.
            yeni_pop = [en_iyi_yol] 
            
            # Yeni nesli ebeveynlerden üreterek doldur.
            while len(yeni_pop) < self.pop_size:
                p1 = random.choice(populasyon)
                p2 = random.choice(populasyon)
                
                child = self.caprazlama(p1, p2) # Üreme
                child = self.mutasyon(child)    # Değişim
                yeni_pop.append(child)
                
            populasyon = yeni_pop # Artık yeni nesil üzerinden devam edilir.

        sure = time.time() - start_time
        return en_iyi_yol, en_iyi_skor, sure

# --- GÖRSELLEŞTİRME ---
def rotayi_ciz(graf, yol, kaynak, hedef):
    """NetworkX ve Matplotlib kullanarak bulunan yolu graf üzerinde kırmızıyla çizer."""
    if not yol: return
    plt.figure(figsize=(10, 7))
    # Düğümlerin ekrandaki yerlerini belirle (seed sabitlendi ki grafik her seferinde aynı gözüksün).
    pos = nx.spring_layout(graf, seed=42)
    
    # Tüm düğümleri ve bağlantıları çiz.
    nx.draw(graf, pos, with_labels=True, node_size=300, node_color='lightgray', font_size=7)
    
    # Bulunan rotayı kırmızı kenarlarla belirginleştir.
    edges = [(yol[i], yol[i+1]) for i in range(len(yol)-1)]
    nx.draw_networkx_nodes(graf, pos, nodelist=yol, node_color='orange')
    nx.draw_networkx_edges(graf, pos, edgelist=edges, edge_color='red', width=2)
    
    plt.title(f"Genetik Algoritma Rota Analizi ({kaynak} -> {hedef})")
    plt.show()

# --- ANA PROGRAM (Uygulamanın Başladığı Nokta) ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   GENETİK ALGORİTMA ROTA BULUCU (DÜZELTİLMİŞ)")
    print("="*50)
    
    try:
        # 1. Adım: Kullanıcıdan kaynak ve hedef düğümleri al.
        k = int(input("👉 Başlangıç Düğümü (Kaynak): "))
        h = int(input("👉 Bitiş Düğümü (Hedef): "))

        # 2. Adım: Düğümlerin varlığını kontrol et.
        if k not in G.nodes or h not in G.nodes:
            print("\n❌ HATA: Düğüm numarası ağda mevcut değil!")
        else:
            # 3. Adım: Algoritmayı yapılandır ve çalıştır.
            # Popülasyon: 100, Nesil: 200, Ağırlıklar: Gecikme %40, Güvenilirlik %40, Kaynak %20.
            ga = GenetikAlgoritma(G, k, h, pop_size=100, nesil=200, agirliklar=[0.4, 0.4, 0.2])
            yol, maliyet, sure = ga.calistir()
            
            # 4. Adım: Sonuçları doğrula ve ekrana yazdır.
            if yol and yol[-1] == h:
                print("\n✅ ROTA BAŞARIYLA BULUNDU")
                print(f"⏱️ Süre: {sure:.4f} sn")
                print(f"🛣️ Rota: {yol}")
                print(f"💰 Maliyet Skoru: {maliyet:.4f}")
                
                # Her bir metriği ayrıca hesaplayıp göster.
                d = ga.calculate_path_delay(yol)
                r = ga.calculate_path_reliability_cost(yol)
                c = ga.calculate_resource_usage(yol)
                print(f"\n📊 Detaylar: Gecikme: {d:.2f}ms, Güv.Maliyeti: {r:.4f}, Kaynak: {c:.2f}")
                
                # 5. Adım: Grafiği göster.
                rotayi_ciz(G, yol, k, h)
            else:
                print("\n❌ HATA: Belirtilen hedefe ulaşılamadı!")

    except ValueError:
        print("\n❌ HATA: Lütfen sadece tam sayı giriniz.")
    except Exception as e:
        print(f"\n❌ Beklenmedik hata: {e}")