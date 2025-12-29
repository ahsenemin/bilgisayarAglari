# =====================================================
# GEREKLİ KÜTÜPHANELER
# =====================================================
import pandas as pd              # CSV dosyalarını okumak için
import networkx as nx            # Ağ (Graph) yapısını oluşturmak için
import matplotlib.pyplot as plt  # Ağ ve grafik görselleştirme
import random                    # Rastgele seçimler (ACO algoritması)
import math                      # Matematiksel işlemler (log, exp)
import time                      # Algoritmanın çalışma süresini ölçmek
import statistics                # Ortalama, standart sapma vb.

def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)

# =====================================================
# 1. VERİ YÜKLEME VE NETWORK GRAPH OLUŞTURMA
# =====================================================
def create_network_graph():
    print("\nℹ️  Veri dosyaları kontrol ediliyor...", end=" ")

    # CSV dosyaları okunmaya çalışılır
    try:
        nodes_df = pd.read_csv(
            'BSM307_317_Guz2025_TermProject_NodeData.csv',
            sep=';', decimal=','
        )
        edges_df = pd.read_csv(
            'BSM307_317_Guz2025_TermProject_EdgeData.csv',
            sep=';', decimal=','
        )
    except FileNotFoundError:
        # Dosyalar bulunamazsa program durdurulur
        print("\n❌ HATA: CSV dosyaları bulunamadı!")
        return None

    # Yönsüz bir ağ (graph) oluşturulur
    G = nx.Graph()

    # -----------------------------
    # DÜĞÜMLERİ AĞA EKLEME
    # -----------------------------
    for _, row in nodes_df.iterrows():
        G.add_node(
            int(row['node_id']),                  # Düğüm ID
            processing_delay=float(row['s_ms']),  # Düğüm işlem gecikmesi (ms)
            reliability=float(row['r_node'])      # Düğüm güvenilirliği
        )

    # -----------------------------
    # BAĞLANTILARI AĞA EKLEME
    # -----------------------------
    for _, row in edges_df.iterrows():
        G.add_edge(
            int(row['src']),                      # Kaynak düğüm
            int(row['dst']),                      # Hedef düğüm
            bandwidth=float(row['capacity_mbps']),# Bant genişliği (Mbps)
            delay=float(row['delay_ms']),         # Bağlantı gecikmesi (ms)
            reliability=float(row['r_link'])      # Bağlantı güvenilirliği
        )

    print("Tamamlandı.")
    print(f"✅ Ağ Yüklendi: {G.number_of_nodes()} Düğüm, {G.number_of_edges()} Bağlantı.\n")
    return G

# =====================================================
# 2. YOL METRİKLERİNİ HESAPLAMA
# =====================================================
def calculate_metrics(G, path):
    total_delay = 0.0            # Toplam gecikme
    reliability_log_sum = 0.0    # Logaritmik güvenilirlik
    resource_cost = 0.0          # Kaynak kullanımı maliyeti

    # Geçersiz yol kontrolü
    if not path or len(path) < 2:
        return float('inf'), float('inf'), float('inf')

    # -----------------------------
    # DÜĞÜM METRİKLERİ
    # -----------------------------
    for i, node in enumerate(path):
        r_node = G.nodes[node]['reliability']

        # Güvenilirlikler çarpım olduğu için log kullanılır
        reliability_log_sum += -math.log(r_node)

        # Başlangıç ve bitiş hariç düğümlerde işlem gecikmesi eklenir
        if i != 0 and i != len(path) - 1:
            total_delay += G.nodes[node]['processing_delay']

    # -----------------------------
    # BAĞLANTI METRİKLERİ
    # -----------------------------
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = G[u][v]

        # Bağlantı gecikmesi
        total_delay += edge['delay']

        # Bağlantı güvenilirliği (log)
        reliability_log_sum += -math.log(edge['reliability'])

        # Bant genişliğine bağlı kaynak maliyeti
        resource_cost += (1000.0 / edge['bandwidth'])

    return total_delay, reliability_log_sum, resource_cost

# =====================================================
# 3. FITNESS (TOPLAM MALİYET) FONKSİYONU
# =====================================================
def calculate_fitness(metrics, weights):
    # Ağırlıklı toplam maliyet hesabı
    return (weights[0] * metrics[0]) + \
           (weights[1] * metrics[1]) + \
           (weights[2] * metrics[2])

# =====================================================
# 4. KARINCA KOLONİSİ OPTİMİZASYONU (ACO)
# =====================================================
class ACORouting:
    def __init__(self, graph, source, destination, required_bandwidth,
                 weights, n_ants=20, n_iterations=50,
                 alpha=1.0, beta=2.0, evaporation=0.5, Q=100, seed= None):

        self.G = graph                  # Ağ
        self.source = source            # Başlangıç düğümü
        self.dest = destination         # Hedef düğüm
        self.B = required_bandwidth     # Minimum bant genişliği
        self.weights = weights          # Maliyet ağırlıkları
        
        self.n_ants = n_ants            # Karınca sayısı
        self.n_iterations = n_iterations# Iterasyon sayısı
        self.alpha = alpha              # Feromon etkisi
        self.beta = beta                # Sezgisel bilginin etkisi
        self.evaporation = evaporation  # Feromon buharlaşma oranı
        self.Q = Q                      # Feromon bırakma sabiti

        # Tüm kenarlara başlangıç feromonu atanır
        self.pheromones = {edge: 1.0 for edge in self.G.edges()}
        self.history = []               # Yakınsama geçmişi
        set_seed(seed)                  # Rastgelelik için seed ayarlanır
    # -----------------------------
    # FEROMON DEĞERİNİ OKUMA
    # -----------------------------
    def get_pheromone(self, u, v):
        if self.G.has_edge(u, v):
            return self.pheromones.get((u, v),
                   self.pheromones.get((v, u), 1.0))
        return 0.0

    # -----------------------------
    # FEROMON GÜNCELLEME
    # -----------------------------
    def update_pheromone(self, u, v, amount):
        if (u, v) in self.pheromones:
            self.pheromones[(u, v)] += amount
        elif (v, u) in self.pheromones:
            self.pheromones[(v, u)] += amount

    # -----------------------------
    # SEZGİSEL (HEURISTIC) FONKSİYON
    # -----------------------------
    def get_heuristic(self, u, v):
        edge = self.G[u][v]
        node_v = self.G.nodes[v]

        # Bant genişliği yetersizse bu yol kullanılmaz
        if edge['bandwidth'] < self.B:
            return 0.0

        # Gecikme maliyeti
        d = edge['delay'] + \
            (node_v['processing_delay'] if v != self.dest else 0)

        # Güvenilirlik maliyeti (log)
        r = -math.log(edge['reliability']) - \
            math.log(node_v['reliability'])

        # Kaynak maliyeti
        bw_cost = 1000.0 / edge['bandwidth']

        # Toplam maliyet
        cost = (self.weights[0] * d) + \
               (self.weights[1] * r) + \
               (self.weights[2] * bw_cost)

        # Düşük maliyet = yüksek sezgisel değer
        return 1.0 / (cost + 0.0001)

    # -----------------------------
    # SONRAKİ DÜĞÜMÜ SEÇME
    # -----------------------------
    def select_next_node(self, current, visited):
        neighbors = [n for n in self.G.neighbors(current)
                     if n not in visited]

        if not neighbors:
            return None

        probs = []
        possible_neighbors = []
        denom = 0.0

        for n in neighbors:
            eta = self.get_heuristic(current, n)
            if eta == 0:
                continue

            tau = self.get_pheromone(current, n)
            score = (tau ** self.alpha) * (eta ** self.beta)

            probs.append(score)
            possible_neighbors.append(n)
            denom += score

        if denom == 0:
            return None

        probs = [p / denom for p in probs]

        # Olasılıksal seçim
        return random.choices(possible_neighbors, weights=probs, k=1)[0]

    # -----------------------------
    # ACO ANA ÇÖZÜM FONKSİYONU
    # -----------------------------
    def solve(self):
        best_path = None
        best_fitness = float('inf')

        for _ in range(self.n_iterations):
            paths = []

            for _ in range(self.n_ants):
                path = [self.source]
                visited = {self.source}
                curr = self.source

                while curr != self.dest:
                    nxt = self.select_next_node(curr, visited)
                    if not nxt:
                        break
                    path.append(nxt)
                    visited.add(nxt)
                    curr = nxt

                if curr == self.dest:
                    metrics = calculate_metrics(self.G, path)
                    fitness = calculate_fitness(metrics, self.weights)
                    paths.append((path, fitness))

                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_path = path

            # Yakınsama bilgisi
            self.history.append(best_fitness)

            # Feromon buharlaşması
            for k in self.pheromones:
                self.pheromones[k] *= (1 - self.evaporation)

            # Feromon bırakma
            for p, fit in paths:
                deposit = self.Q / fit
                for i in range(len(p) - 1):
                    self.update_pheromone(p[i], p[i + 1], deposit)

        return best_path, best_fitness, self.history

# =====================================================
# 5. GÖRSELLEŞTİRME
# =====================================================
def draw_results(G, path, s_node, d_node, score, history):
    print("🎨 Grafik çiziliyor...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Ağın genel görünümü
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=20, node_color='#dddddd')
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.1)

    # En iyi yol kırmızı çizilir
    if path:
        edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, ax=ax1,
                               nodelist=path, node_color='orange', node_size=60)
        nx.draw_networkx_edges(G, pos, ax=ax1,
                               edgelist=edges, edge_color='red', width=2.5)

    # Başlangıç ve hedef düğümler
    nx.draw_networkx_nodes(G, pos, ax=ax1,
                           nodelist=[s_node], node_color='green', node_size=150)
    nx.draw_networkx_nodes(G, pos, ax=ax1,
                           nodelist=[d_node], node_color='blue', node_size=150)

    ax1.set_title(f"ACO Rota: {s_node} → {d_node}\n(Maliyet: {score:.4f})")
    ax1.axis('off')

    # Yakınsama grafiği
    ax2.plot(history)
    ax2.set_title("Yakınsama Grafiği")
    ax2.set_xlabel("İterasyon")
    ax2.set_ylabel("En İyi Maliyet")

    plt.show()

# =====================================================
# 6. ANA UYGULAMA
# =====================================================
def run_application(G):
    print("\n📍 ROTA PLANLAMA")

    try:
        s_node = int(input("Başlangıç düğümü: "))
        d_node = int(input("Hedef düğüm: "))
        b_req = float(input("Bant genişliği: "))
        
    except:
        s_node, d_node, b_req = 8, 44, 4.0

    try:
        weights = [
            float(input("Delay ağırlığı: ")),
            float(input("Reliability ağırlığı: ")),
            float(input("Resource ağırlığı: "))
        ]
    except:
        weights = [0.4, 0.4, 0.2]

    aco = ACORouting(G, s_node, d_node, b_req, weights,
                     n_ants=20, n_iterations=200)

    path, fitness, history = aco.solve()

    if path:
        d, r, u = calculate_metrics(G, path)
        print("\n✅ EN İYİ YOL:", path)
        print("Toplam gecikme:", d)
        print("Toplam güvenilirlik:", math.exp(-r))
        print("Kaynak maliyeti:", u)

        draw_results(G, path, s_node, d_node, fitness, history)
    else:
        print("❌ Yol bulunamadı")

# =====================================================
# PROGRAM BAŞLANGICI
# =====================================================
if __name__ == "__main__":
    G = create_network_graph()
    if G:
        run_application(G)