import random  # Rastgele sayı üretimi; mutasyon, çaprazlama ve popülasyon başlatma için.
import math    # Matematiksel işlemler; özellikle güvenilirlik maliyetindeki logaritma (ln) hesabı için.
import time    # Performans ölçümü; algoritmanın çözüm bulma süresini saniye cinsinden hesaplar.
import re      # Kullanıcıdan ağırlık alma fonksiyonunda string parçalama için.
import matplotlib.pyplot as plt # Görselleştirme; bulunan yolu grafik penceresinde çizmek için.
import networkx as nx           # Çizge teorisi; düğüm ve bağlantı (edge) yönetimi için ana kütüphane.

# 'ag.py' dosyasından oluşturulan Graf (G) nesnesini projenin içine dahil eder.
# Bu graf; tüm gecikme, güvenilirlik ve bant genişliği verilerini barındıran "ağ haritasıdır".
from ag import G

SEED = 42  # Tekrarlanabilirlik için varsayılan tohum (seed)

def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)


def prompt_weights(default=(0.4, 0.4, 0.2)):
    """
    Kullanıcıdan 3 ağırlık (Delay, Reliability, Resource) alır.
    - Boş geçilirse default kullanılır.
    - Toplam 1 değilse normalize eder.
    Kabul edilen formatlar:
      0.4 0.4 0.2
      0,4 0,4 0,2
      0.4,0.4,0.2  (virgülle ayrılmış)
    Not: "0,4,0,4,0,2" gibi boşluksuz ondalık-virgül formatı belirsiz olduğu için önerilmez.
    """
    print("\n⚙️  Ağırlıklar (Delay, Reliability, Resource)")
    print(f"    Varsayılan: {default[0]:.2f} {default[1]:.2f} {default[2]:.2f}")
    raw = input("    Üç değeri gir (Enter = varsayılan): ").strip()
    if not raw:
        w = [float(default[0]), float(default[1]), float(default[2])]
    else:
        low = raw.lower()
        if low in {"d", "default", "varsayilan", "varsayılan", "v"}:
            w = [float(default[0]), float(default[1]), float(default[2])]
        else:
            # boşluk/; varsa onları ayırıcı kabul et (ondalık virgül desteklenir)
            if any(sep in raw for sep in [" ", "\t", ";"]):
                parts = re.split(r"[;\s]+", raw.strip())
            else:
                # virgülle ayrılmış format (0.4,0.4,0.2) için
                parts = [p.strip() for p in raw.split(",") if p.strip()]

            if len(parts) != 3:
                raise ValueError("Ağırlıklar 3 adet olmalı. Örn: 0.4 0.4 0.2")

            w = []
            for p in parts:
                # ondalık virgül -> nokta
                p2 = p.replace(",", ".")
                w.append(float(p2))

    if any(x < 0 for x in w):
        raise ValueError("Ağırlıklar negatif olamaz.")

    s = sum(w)
    if s <= 0:
        raise ValueError("Ağırlıkların toplamı 0 olamaz.")

    # Normalize (toplam 1 değilse)
    if abs(s - 1.0) > 1e-9:
        w = [x / s for x in w]
        print(f"    (Bilgi) Toplam 1 olmadığı için normalize edildi. Yeni toplam=1.00")

    print(f"✅ Kullanılan ağırlıklar: Delay={w[0]:.3f}, Reliability={w[1]:.3f}, Resource={w[2]:.3f}")
    return w

# ==============================================================================
# 1. Genetik Algoritma Sınıfı (Meta-Sezgisel Çözücü)
# ==============================================================================
class GenetikAlgoritma:
    """QoS (Hizmet Kalitesi) parametrelerine göre en uygun yolu evrimsel süreçle bulan sınıf."""
    
    def __init__(self, graf, kaynak, hedef, pop_size=100, mutasyon_orani=0.1, nesil=100, agirliklar=None, min_bw=0, seed=None):
        """Sınıfın başlangıç ayarlarını (DNA'sını) yapan kurucu metod."""
        self.graph = graf           # Ağ haritasını sisteme tanıtır.
        self.kaynak = kaynak        # Rota nereden başlayacak (Örn: 8).
        self.hedef = hedef          # Rota nerede bitecek (Örn: 44).
        self.pop_size = pop_size    # Aynı anda kaç farklı yolun (çözümün) hayatta kalacağı.
        self.mutation_rate = mutasyon_orani # Bir yolun rastgele değişme ihtimali (%10).
        self.generations = nesil    # Evrimin kaç tur (kuşak) boyunca devam edeceği.
        self.min_bw = min_bw        # Kullanıcının "en az şu hız lazım" dediği alt limit kısıtı.
        set_seed(seed)              # Rastgelelik için sabit tohum değeri ayarlanır.
        
        # Kullanıcı tercihlerine göre ağırlıklar (PDF'e göre toplamı 1 olmalı).
        self.weights = agirliklar if agirliklar else [0.33, 0.33, 0.33]
        s = sum(self.weights)
        self.weights = [w/s for w in self.weights] if s > 0 else [1/3, 1/3, 1/3]

    # --- QOS METRİK HESAPLAMA FONKSİYONLARI ---
    
    def calculate_path_delay(self, path):
        """Yolun toplam gecikmesini (Link Gecikmesi + Düğüm İşlem Süresi) hesaplar."""
        total_delay = 0
        for i in range(len(path) - 1): # Yol üzerindeki her bir bağlantı (çizgi) için döner.
            u, v = path[i], path[i+1]
            total_delay += self.graph[u][v].get('delay', 0) # Bağlantı gecikmesini toplama ekler.
        for node in path[1:-1]: # Sadece ara durakların (kaynak ve hedef hariç) işlem süresini ekler.
            total_delay += self.graph.nodes[node].get('processing_delay', 0)
        return total_delay

    def calculate_path_reliability_cost(self, path):
        """Güvenilirliği (-log R) maliyetine çevirir. Değer ne kadar düşükse yol o kadar güvenilirdir."""
        total_cost = 0
        for i in range(len(path) - 1):
            r = self.graph[path[i]][path[i+1]].get('reliability', 0.99)
            if r <= 0: r = 0.0001 # Logaritma hatasını (sıfır olamaz) önlemek için alt sınır.
            total_cost += -math.log(r) # Çarpımsal güvenilirliği toplamsal maliyete dönüştürür.
        for node in path:
            r = self.graph.nodes[node].get('reliability', 0.99)
            if r <= 0: r = 0.0001
            total_cost += -math.log(r)
        return total_cost

    def calculate_resource_usage(self, path):
        """Düşük bant genişliğini cezalandıran maliyet formülü (1000 / Bant Genişliği)."""
        total_resource = 0
        for i in range(len(path) - 1):
            bw = self.graph[path[i]][path[i+1]].get('bandwidth', 100)
            if bw <= 0: bw = 1 # Paydanın 0 olup çökmesini engeller.
            total_resource += (1000.0 / bw) # Hız düştükçe maliyet skoru artar (Minimizasyon).
        return total_resource

    def toplam_maliyet_hesapla(self, path):
        """Tüm metrikleri ağırlıklarla toplar ve 'Bant Genişliği Kısıtı'nı denetler."""
        try:
            # 1. HEDEF KONTROLÜ: Yol senin girdiğin durakta (hedefte) bitmiyorsa elenir.
            if not path or path[-1] != self.hedef:
                return float('inf') # Sonsuz maliyet vererek algoritmadan dışlar.
            
            # 2. BANT GENİŞLİĞİ KONTROLÜ (SERT KISIT):
            # Yolun herhangi bir yerinde hız, kullanıcının istediği değerden (min_bw) düşükse yolu çöpe at.
            for i in range(len(path) - 1):
                link_bw = self.graph[path[i]][path[i+1]].get('bandwidth', 0)
                if link_bw < self.min_bw:
                    return float('inf') # Darboğaz olan yolu geçersiz sayar.

            # 3. NİHAİ SKOR HESABI (Weighted Sum Method):
            d = self.calculate_path_delay(path)
            r = self.calculate_path_reliability_cost(path)
            res = self.calculate_resource_usage(path)
            
            # Formül: (W1 * Gecikme) + (W2 * Güv.Maliyeti * 100) + (W3 * Kaynak)
            return (self.weights[0] * d) + (self.weights[1] * r * 100) + (self.weights[2] * res)
        except Exception:
            return float('inf') # Beklenmeyen hatalarda yolu eler.

    def uygunluk(self, path):
        """Uygunluk (Fitness): Maliyet ne kadar küçükse başarı puanı o kadar büyüktür (1 / Maliyet)."""
        cost = self.toplam_maliyet_hesapla(path)
        return 1.0 / (cost + 1e-9) # 1e-9: Sıfıra bölünme hatasını önleyen küçük sayı.

    def secim_turnuva(self, pop, k=3):
        """Turnuva seçilimi: maliyeti düşük olan birey seçilme şansını artırır."""
        if not pop:
            raise ValueError("Popülasyon boş")
        adaylar = random.sample(pop, k=min(k, len(pop)))
        return min(adaylar, key=self.toplam_maliyet_hesapla)

    # --- GENETİK ALGORİTMA OPERATÖRLERİ (Evrim Mekanizması) ---
    
    def rastgele_yol_bul(self):
        """Kaynaktan hedefe komşuluk ilişkilerini takip eden rastgele bir yol üretir."""
        try:
            path = [self.kaynak] # Yol başlangıç noktasından başlar.
            curr = self.kaynak
            visited = {self.kaynak} # Düğümlerin tekrar edilmemesi (loop olmaması) için tutulan liste.
            while curr != self.hedef:
                # Henüz uğranmamış komşu düğümleri listeler.
                neighbors = [n for n in self.graph.neighbors(curr)
                             if n not in visited and self.graph[curr][n].get('bandwidth', 0) >= self.min_bw]
                if not neighbors: return None # Çıkmaz sokağa girerse yolu iptal eder.
                curr = random.choice(neighbors) # Komşulardan rastgele birini seçer.
                path.append(curr)
                visited.add(curr)
                if len(path) > 100: return None # Yol çok uzarsa algoritmanın sonsuza girmesini önler.
            return path
        except Exception:
            return None

    def populasyon_olustur(self):
        """Belirlenen popülasyon boyutuna (Örn: 100) ulaşana kadar rastgele yollar üretir."""
        populasyon = []
        tries = 0
        while len(populasyon) < self.pop_size and tries < self.pop_size * 20:
            yol = self.rastgele_yol_bul()
            if yol: populasyon.append(yol) # Geçerli yolları havuzu ekler.
            tries += 1
        return populasyon

    def caprazlama(self, p1, p2):
        """Çaprazlama (Crossover): İki başarılı yolun (anne-baba) ortak bir düğümden takasını yapar."""
        # İki yolun ortak olan (başlangıç ve bitiş hariç) duraklarını bulur.
        common = [n for n in p1 if n in p2 and n != self.kaynak and n != self.hedef]
        if not common: return p1 # Ortak nokta yoksa üreme yapılamaz, p1'i korur.
        
        node = random.choice(common) # Rastgele bir ortak düğüm (gen) seçer.
        idx1, idx2 = p1.index(node), p2.index(node)
        new_path = p1[:idx1] + p2[idx2:] # P1'in başıyla P2'nin sonunu birleştirir.
        
        # Yolun geçerli (döngüsüz) ve hedefe ulaştığını kontrol eder.
        if len(new_path) == len(set(new_path)) and new_path[-1] == self.hedef:
            return new_path
        return p1

    def mutasyon(self, path):
        """Mutasyon: Yolun bir noktasını rastgele koparıp hedefe yeni bir parça örer (Çeşitlilik)."""
        if random.random() < self.mutation_rate and len(path) > 2:
            try:
                cut_idx = random.randint(1, len(path)-2) # Rastgele bir kırılma noktası.
                node = path[cut_idx]
                curr = node
                new_segment = []
                visited = set(path[:cut_idx+1])
                
                # Kırılan noktadan hedefe doğru yeniden yol bulmaya çalışır.
                for _ in range(50):
                    if curr == self.hedef: break
                    neighbors = [n for n in self.graph.neighbors(curr)
                             if n not in visited and self.graph[curr][n].get('bandwidth', 0) >= self.min_bw]
                    if not neighbors: return path # Bağlantı kurulamazsa mutasyon başarısız.
                    curr = random.choice(neighbors)
                    new_segment.append(curr)
                    visited.add(curr)
                
                # Yeni segment hedefe ulaştıysa eski yolla birleştirir.
                if new_segment and new_segment[-1] == self.hedef:
                    return path[:cut_idx+1] + new_segment
            except Exception:
                pass
        return path

    def calistir(self):
        """Genetik Algoritma ana döngüsünü (Seçim -> Çaprazlama -> Mutasyon) yönetir."""
        start_time = time.time()  # Zamanlayıcıyı başlatır.
        populasyon = self.populasyon_olustur()  # İlk nesil oluşturulur.

        if not populasyon:
            return None, float('inf'), 0  # Yol bulunamazsa erken çıkış.

        # Başlangıçta (None kalmasın diye) en iyiyi seç.
        en_iyi_yol = min(populasyon, key=self.toplam_maliyet_hesapla)
        en_iyi_skor = self.toplam_maliyet_hesapla(en_iyi_yol)

        # Sunum/Terminal için bilgilendirme mesajı.
        print(f"🧬 Genetik Algoritma Çalışıyor... (Hedef: {self.hedef}, Min BW: {self.min_bw} Mbps)")

        for _ in range(self.generations):  # Belirlenen nesil kadar evrim sürer.
            if not populasyon:
                break

            # Mevcut neslin en iyi bireyini bul
            gen_best = min(populasyon, key=self.toplam_maliyet_hesapla)
            gen_cost = self.toplam_maliyet_hesapla(gen_best)

            # Global en iyiyi güncelle
            if gen_cost < en_iyi_skor:
                en_iyi_skor = gen_cost
                en_iyi_yol = gen_best

            # Elitizm: en iyi bireyi yeni nesle aktar
            yeni_pop = [en_iyi_yol]

            # Yeni nesli doldur
            deneme = 0
            while len(yeni_pop) < self.pop_size and deneme < self.pop_size * 50:
                # Seçilim baskısı: ebeveynleri rastgele değil, turnuva seçilimi ile al
                p1 = self.secim_turnuva(populasyon, k=3)
                p2 = self.secim_turnuva(populasyon, k=3)

                child = self.caprazlama(p1, p2)
                child = self.mutasyon(child)

                # Geçersiz (inf maliyetli) çocukları ekleme
                if child and self.toplam_maliyet_hesapla(child) < float('inf'):
                    yeni_pop.append(child)

                deneme += 1

            # Eğer yeni nesil dolmadıysa, mevcut neslin en iyi bireyleriyle tamamla (çökmesin)
            if len(yeni_pop) < self.pop_size:
                for p in sorted(populasyon, key=self.toplam_maliyet_hesapla):
                    if len(yeni_pop) >= self.pop_size:
                        break
                    yeni_pop.append(p)

            populasyon = yeni_pop

        return en_iyi_yol, en_iyi_skor, time.time() - start_time  # En iyi sonuçları döndürür.

# --- GÖRSELLEŞTİRME ---
def rotayi_ciz(graf, yol, kaynak, hedef):
    """Bulunan yolu ağ haritası üzerinde görselleştirir."""
    if not yol: return
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(graf, seed=42) # Graf düğümlerinin ekrandaki dizilimini ayarlar.
    # Tüm düğümleri ve bağlantıları hafif gri renkte çizer.
    nx.draw(graf, pos, with_labels=True, node_size=300, node_color='lightgray', font_size=7)
    edges = [(yol[i], yol[i+1]) for i in range(len(yol)-1)] # Bulunan yolun kenarlarını listeler.
    # Bulunan yolu turuncu düğümler ve kalın kırmızı çizgilerle belirginleştirir.
    nx.draw_networkx_nodes(graf, pos, nodelist=yol, node_color='orange')
    nx.draw_networkx_edges(graf, pos, edgelist=edges, edge_color='red', width=2)
    plt.title(f"Genetik Algoritma Rota Analizi ({kaynak} -> {hedef})")
    plt.show() # Grafiği ekranda gösterir.

# --- ANA PROGRAM (Uygulamanın Giriş Kapısı) ---
if __name__ == "__main__":

    set_seed(SEED)

    print("\n" + "="*50)
    print("   GENETİK ALGORİTMA ROTA BULUCU (FULL SÜRÜM)")
    print("="*50)
    
    try:
        # Kullanıcıdan gerekli girdileri (Kaynak, Hedef, Hız) alır.
        k = int(input("👉 Başlangıç Düğümü (Kaynak): "))
        h = int(input("👉 Bitiş Düğümü (Hedef): "))
        istenen_bw = int(input("🚀 Minimum Bant Genişliği Talebi (Mbps): "))

        # Girilen düğümün haritada olup olmadığını denetler.
        if k not in G.nodes or h not in G.nodes:
            print("\n❌ HATA: Düğüm numarası ağda yok!")
        else:
            # Algoritma nesnesini başlatır (Ağırlıklar: Gecikme 0.4, Güven 0.4, Kaynak 0.2).
            agirliklar = prompt_weights(default=(0.4, 0.4, 0.2))
            ga = GenetikAlgoritma(G, k, h, pop_size=100, nesil=200, agirliklar=agirliklar, min_bw=istenen_bw)
            yol, maliyet, sure = ga.calistir() # Evrim sürecini başlatır.
            
            # Sonuçları terminale raporlar.
            if yol and yol[-1] == h:
                print("\n✅ ROTA BAŞARIYLA BULUNDU")
                print(f"⏱️  Hesaplama Süresi: {sure:.4f} saniye")
                print(f"🛣️  Rota: {yol}")
                print(f"💰 Toplam Maliyet Skoru: {maliyet:.4f}")
                
                # Rota üzerindeki her bir kriterin değerini ayrıca hesaplayıp gösterir.
                d = ga.calculate_path_delay(yol)
                r = ga.calculate_path_reliability_cost(yol)
                c = ga.calculate_resource_usage(yol)
                print(f"\n📊 QoS Detayları: Gecikme: {d:.2f}ms, Güv.Maliyeti: {r:.4f}, Kaynak: {c:.2f}")
                
                rotayi_ciz(G, yol, k, h) # Görsel sonucu açar.
            else:
                # Bant genişliği kısıtı yüzünden veya kopuk ağ yüzünden yol bulunamazsa mesaj verir.
                print("\n❌ HATA: Belirtilen kısıtlar altında hedefe ulaşılamadı!")

    except ValueError:
        # Sayı yerine harf girilmesi gibi hataları yakalar.
        print("\n❌ HATA: Lütfen geçerli bir tam sayı giriniz.")
    except Exception as e:
        # Beklenmedik sistem hatalarını yakalar.
        print(f"\n❌ Beklenmedik hata: {e}")
