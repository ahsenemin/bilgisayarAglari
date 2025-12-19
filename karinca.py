import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import time
import statistics

# ==========================================
# 1. VERİ YÜKLEME VE GRAFİK
# ==========================================
def create_network_graph():
    print("\nℹ️  Veri dosyaları kontrol ediliyor...", end=" ")
    try:
        nodes_df = pd.read_csv('BSM307_317_Guz2025_TermProject_NodeData.csv', sep=';', decimal=',')
        edges_df = pd.read_csv('BSM307_317_Guz2025_TermProject_EdgeData.csv', sep=';', decimal=',')
    except FileNotFoundError:
        print("\n❌ HATA: CSV dosyaları bulunamadı!")
        return None

    G = nx.Graph()

    for _, row in nodes_df.iterrows():
        G.add_node(int(row['node_id']), 
                   processing_delay=float(row['s_ms']), 
                   reliability=float(row['r_node']))

    for _, row in edges_df.iterrows():
        G.add_edge(int(row['src']), int(row['dst']), 
                   bandwidth=float(row['capacity_mbps']), 
                   delay=float(row['delay_ms']), 
                   reliability=float(row['r_link']))
    
    print("Tamamlandı.")
    print(f"✅ Ağ Yüklendi: {G.number_of_nodes()} Düğüm, {G.number_of_edges()} Bağlantı.\n")
    return G

# ==========================================
# 2. MALİYET HESAPLAMA
# ==========================================
def calculate_metrics(G, path):
    total_delay = 0.0
    reliability_log_sum = 0.0
    resource_cost = 0.0
    
    if not path or len(path) < 2:
        return float('inf'), float('inf'), float('inf')

    # Düğüm Metrikleri
    for i, node in enumerate(path):
        r_node = G.nodes[node]['reliability']
        reliability_log_sum += -math.log(r_node)
        
        if i != 0 and i != len(path) - 1:
            total_delay += G.nodes[node]['processing_delay']

    # Bağlantı Metrikleri
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge = G[u][v]
        total_delay += edge['delay']
        reliability_log_sum += -math.log(edge['reliability'])
        resource_cost += (1000.0 / edge['bandwidth'])

    return total_delay, reliability_log_sum, resource_cost

def calculate_fitness(metrics, weights):
    return (weights[0] * metrics[0]) + \
           (weights[1] * metrics[1]) + \
           (weights[2] * metrics[2])

# ==========================================
# 3. KARINCA KOLONİSİ (ACO)
# ==========================================
class ACORouting:
    def __init__(self, graph, source, destination, required_bandwidth, weights, 
                 n_ants=20, n_iterations=50, alpha=1.0, beta=2.0, evaporation=0.5, Q=100):
        self.G = graph
        self.source = source
        self.dest = destination
        self.B = required_bandwidth
        self.weights = weights
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        self.pheromones = {edge: 1.0 for edge in self.G.edges()}
        self.history = []

    def get_pheromone(self, u, v):
        if self.G.has_edge(u, v):
            return self.pheromones.get((u, v), self.pheromones.get((v, u), 1.0))
        return 0.0

    def update_pheromone(self, u, v, amount):
        if (u, v) in self.pheromones: self.pheromones[(u, v)] += amount
        elif (v, u) in self.pheromones: self.pheromones[(v, u)] += amount

    def get_heuristic(self, u, v):
        edge = self.G[u][v]
        node_v = self.G.nodes[v]
        
        # Bant Genişliği Kısıtı
        if edge['bandwidth'] < self.B:
            return 0.0
            
        d = edge['delay'] + (node_v['processing_delay'] if v != self.dest else 0)
        r = -math.log(edge['reliability']) - math.log(node_v['reliability'])
        bw_cost = 1000.0 / edge['bandwidth']
        
        cost = (self.weights[0]*d) + (self.weights[1]*r) + (self.weights[2]*bw_cost)
        return 1.0 / (cost + 0.0001)

    def select_next_node(self, current, visited):
        neighbors = [n for n in self.G.neighbors(current) if n not in visited]
        if not neighbors: return None
        
        probs = []
        denom = 0.0
        possible_neighbors = []

        for n in neighbors:
            eta = self.get_heuristic(current, n)
            if eta == 0: continue
            
            tau = self.get_pheromone(current, n)
            score = (tau ** self.alpha) * (eta ** self.beta)
            probs.append(score)
            denom += score
            possible_neighbors.append(n)
            
        if denom == 0 or not possible_neighbors: return None
            
        probs = [p/denom for p in probs]
        return random.choices(possible_neighbors, weights=probs, k=1)[0]

    def solve(self):
        best_path = None
        best_fitness = float('inf')

        for _ in range(self.n_iterations):
            path_list = []
            for _ in range(self.n_ants):
                path = [self.source]
                visited = {self.source}
                curr = self.source
                
                while curr != self.dest:
                    nxt = self.select_next_node(curr, visited)
                    if not nxt: break
                    path.append(nxt)
                    visited.add(nxt)
                    curr = nxt
                
                if curr == self.dest:
                    mets = calculate_metrics(self.G, path)
                    fitness = calculate_fitness(mets, self.weights)
                    path_list.append((path, fitness))
                    
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_path = path
            
            self.history.append(best_fitness)

            for k in self.pheromones:
                self.pheromones[k] *= (1.0 - self.evaporation)
            
            for p, fit in path_list:
                deposit = self.Q / fit
                for i in range(len(p)-1):
                    self.update_pheromone(p[i], p[i+1], deposit)
                    
        return best_path, best_fitness, self.history

# ==========================================
# 4. GÖRSELLEŞTİRME VE ÇIKTI YÖNETİMİ
# ==========================================
def draw_results(G, path, s_node, d_node, score, history):
    print("🎨 Grafik çiziliyor, lütfen bekleyin...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # SOL: Ağ Haritası
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=20, node_color='#dddddd')
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.1, edge_color='#999999')
    
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=path, node_size=60, node_color='orange')
        nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=path_edges, edge_color='red', width=2.5)
        
    nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=[s_node], node_size=150, node_color='green', label="Başlangıç")
    nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=[d_node], node_size=150, node_color='blue', label="Bitiş")
    
    ax1.set_title(f"ACO Rota: {s_node} -> {d_node}\n(Maliyet: {score:.4f})")
    ax1.legend()
    ax1.axis('off')

    # SAĞ: Yakınsama Grafiği
    ax2.plot(history, color='red', linewidth=2)
    ax2.set_title("Algoritma Yakınsama (Convergence)")
    ax2.set_xlabel("İterasyon Sayısı")
    ax2.set_ylabel("En İyi Maliyet (Cost)")
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def run_application(G):
    print("-" * 40)
    print("📍 ROTA PLANLAMA VE ANALİZ")
    print("-" * 40)
    
    # 1. ADIM: DÜĞÜM VE BANT GENİŞLİĞİ SEÇİMİ
    print("\n--- DÜĞÜM SEÇİMİ ---")
    try:
        s_node = int(input("👉 Başlangıç Düğümü (S) [0-249]: "))
        d_node = int(input("👉 Hedef Düğüm  (D) [0-249]: "))
        b_req  = float(input("👉 Talep Edilen Bant Genişliği [Mbps]: "))
    except ValueError:
        print("⚠️ Hatalı giriş! Varsayılanlar: S=8, D=44, B=4")
        s_node, d_node, b_req = 8, 44, 4.0

    # 2. ADIM: AĞIRLIK AYARLARI
    print("\n--- AĞIRLIK AYARLARI ---")
    print("Toplamları 1.0 olacak şekilde giriniz.")
    try:
        w_d   = float(input("1. Gecikme (Delay) : "))
        w_r   = float(input("2. Güvenilirlik    : "))
        w_res = float(input("3. Kaynak (Res)    : "))
        weights = [w_d, w_r, w_res]
    except ValueError:
        print("⚠️ Hatalı giriş! Varsayılan ağırlıklar (0.4, 0.4, 0.2) kullanılıyor.")
        weights = [0.4, 0.4, 0.2]

    # DENEY BAŞLANGICI
    repeats = 5
    results = []
    times = []
    
    best_overall_path = None
    best_overall_score = float('inf')
    best_overall_history = []
    
    # ACO Parametreleri
    ITERATION_COUNT = 50 

    print(f"\n🐜 Karıncalar {s_node} -> {d_node} rotasını arıyor ({ITERATION_COUNT} tur, {repeats} tekrar)...")
    print("-" * 65)
    print(f"{'Tekrar':<8} | {'Maliyet':<15} | {'Süre (sn)':<12} | {'Durum'}")
    print("-" * 65)

    for i in range(repeats):
        start_t = time.time()
        aco = ACORouting(G, s_node, d_node, b_req, weights, n_ants=20, n_iterations=ITERATION_COUNT)
        path, fitness, history = aco.solve()
        end_t = time.time()
        duration = end_t - start_t
        
        status = "✅ Bulundu" if path else "❌ Başarısız"
        fit_str = f"{fitness:.4f}" if path else "---"
        
        if path:
            results.append(fitness)
            times.append(duration)
            if fitness < best_overall_score:
                best_overall_score = fitness
                best_overall_path = path
                best_overall_history = history
        
        print(f"{i+1:<8} | {fit_str:<15} | {duration:<12.4f} | {status}")

    # --- FİNAL SUNUM EKRANI (TAM İSTENİLEN FORMAT) ---
    print("\n" + "="*60)
    if best_overall_path:
        avg_time = statistics.mean(times)
        
        print(f"✅ HEDEF BULUNDU! ({s_node} -> {d_node})")
        print("="*60)
        print(f"⏱️  Hesaplama Süresi : {avg_time:.4f} saniye ({ITERATION_COUNT} tur)")
        print(f"🛣️  Gidilen Yol ({len(best_overall_path)} Adım):")
        print(f"    {best_overall_path}")
        print("-" * 60)

        # METRİK DETAYLARI - GÖRSELDEKİ GİBİ
        d, r, u = calculate_metrics(G, best_overall_path)
        print("📊 METRİK DETAYLARI:")
        # Hizalama için :<27 kullanıyoruz
        print(f"   • {'Toplam Gecikme (Delay)':<27} : {d:.4f} ms")
        print(f"   • {'Toplam Güvenilirlik (Rel)':<27} : {math.exp(-r):.6f}")
        print(f"   • {'Kaynak Kullanımı (Res)':<27} : {u:.4f}")
        print("-" * 60)
        print(f"🏆 {'GENEL MALİYET SKORU':<27} : {best_overall_score:.4f}")
        print("-" * 60)

        # İSTATİSTİKSEL SONUÇLAR
        if results:
            avg_score = statistics.mean(results)
            std_dev = statistics.stdev(results) if len(results) > 1 else 0.0
            best_res = min(results)
            worst_res = max(results)

            print("\n📈 İSTATİSTİKSEL ANALİZ (5 Tekrar):")
            print(f"   • Ortalama Maliyet : {avg_score:.4f}")
            print(f"   • Standart Sapma   : {std_dev:.4f}")
            print(f"   • En İyi (Best)    : {best_res:.4f}")
            print(f"   • En Kötü (Worst)  : {worst_res:.4f}")
        
        draw_results(G, best_overall_path, s_node, d_node, best_overall_score, best_overall_history)
        
    else:
        print("❌ HATA: Uygun bir yol bulunamadı.")
        print("   (Bant genişliği kısıtı çok yüksek olabilir)")
        print("=" * 60)

if __name__ == "__main__":
    G = create_network_graph()
    if G:
        run_application(G)