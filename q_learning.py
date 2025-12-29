import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import random

import os

import ag as ag  # <-- Ağ (Graph) buradan geliyor (ag.py değişmedi!)

# ===============================
# RANDOM SEED (REPRODUCIBILITY)
# ===============================
def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)

# =========================================================
# 0) AĞIRLIKLAR (Weighted Sum Method) - DÜZENLENEBİLİR ALAN
# =========================================================
DEFAULT_WEIGHTS = {"w_delay": 1.0, "w_rel": 1.0, "w_bw": 1.0} # Ağırlıkları normalize et

def normalize_weights(w_delay, w_rel, w_bw):
    s = float(w_delay) + float(w_rel) + float(w_bw)

    if s <= 0:
        raise ValueError("Ağırlıkların toplamı 0 veya negatif olamaz.")
    return {
        "w_delay": float(w_delay) / s,
        "w_rel":   float(w_rel)   / s,
        "w_bw":    float(w_bw)    / s
    }

# ----- Yardımcı: güvenli dönüştürücüler (virgüllü ondalıklar için) -----
def safe_float(x):
    if isinstance(x, str):
        x = x.replace(',', '.').strip()
    return float(x)

def safe_int(x):
    if isinstance(x, str):
        x = x.strip()
    return int(x)

# =========================================================
# 1) AĞI ag.py İÇİNDEN AL
#    (ag.py SENİN 1. KODUN, DEĞİŞTİRMİYORUZ)
# =========================================================
G = ag.G  # NetworkX Graph

neighbors = {n: list(G.neighbors(n)) for n in G.nodes()} # node: [komşu düğümler] n düğümüne doğrudan bağlı olan komşu düğümleri verir

# =========================================================
# 2) DEMAND VERİSİNİ OKU (Opsiyonel)
# =========================================================
DEMAND_CSV = "BSM307_317_Guz2025_TermProject_DemandData.csv"  # varsa okunur, yoksa manuel
demand_df = None

if os.path.exists(DEMAND_CSV):
    demand_df = pd.read_csv(DEMAND_CSV, sep=';', decimal=',')  # src;dst;demand_mbps
    demand_df['src'] = demand_df['src'].apply(safe_int)
    demand_df['dst'] = demand_df['dst'].apply(safe_int)
    demand_df['demand_mbps'] = demand_df['demand_mbps'].apply(safe_float)

# =========================================================
# 3) ÖDÜL (Reward) - Demand'li Hard Constraint
#    NOT: Attribute isimleri ag.py'deki isimlere göre uyarlanmıştır:
#    Node: processing_delay, reliability
#    Edge: bandwidth, delay, reliability
# =========================================================
INVALID_ACTION_PENALTY = 1e6  # büyük ceza -> reward = -1e6

def reward_multi_with_demand(s, a, s_next, G, weights, demand_mbps=None):
    # s = mevcut düğüm
    # a = seçilen komşu düğüm
    # s_next = sonraki düğüm
    # G = ağ (networkx graph)
    # weights = ağırlıklar sözlüğü
    # demand_mbps = talep edilen bant genişliği (opsiyonel)
    """
    Multi-objective cost:
      cost = w_delay*delay + w_rel*(1-rel) + w_bw*(100/bandwidth)
    Hard constraint:
      eğer demand_mbps varsa ve edge bandwidth < demand -> aşırı ceza
    """

    edge = G.edges[s, s_next]
    node_next = G.nodes[s_next] # Edge ve sonraki düğümün attribute'ları

    # HARD CONSTRAINT (Demand)
    if demand_mbps is not None:
        if float(edge['bandwidth']) < float(demand_mbps):
            return -INVALID_ACTION_PENALTY
    # eğer talep edilen bant genişliği sağlanmıyorsa büyük ceza
    # (Bu, Q-learning'in bu tür yolları öğrenmemesini sağlar)

    delay = float(edge['delay']) + float(node_next['processing_delay'])  # ms, delay = delay_ms + s_ms
    rel = float(edge['reliability']) * float(node_next['reliability']) # reliability = r_link * r_node
    unreliab = 1.0 - rel # Güvenilirlik metriği maksimizasyon probleminden minimizasyona dönüştürülmüştür. Bu amaçla basit yaklaşım olarak 1 - rel kullanılmıştır.”

    inv_bw = 100.0 / max(1.0, float(edge['bandwidth'])) # geniş kapasite = düşük maliyet

    cost = (weights["w_delay"] * delay +
            weights["w_rel"]   * unreliab +
            weights["w_bw"]    * inv_bw)

    return -cost # Negatif maliyet = ödül, Amaç: ödülü maksimize etmek = maliyeti minimize etmek


# Kullanıcadan W_delay, W_rel, W_bw bir, demand_mbps bir kere çekilir
# Ağırlıklar ve demand değerini içine alarak, Q-learning’in beklediği imzaya sahip özelleştirilmiş bir reward fonksiyonu üretir; bu sayede öğrenme sırasında bu parametreler tekrar tekrar taşınmadan kullanılabilir.
def make_reward_fn(weights, demand_mbps=None):
    def _reward_fn(s, a, s_next, G):
        return reward_multi_with_demand(s, a, s_next, G, weights, demand_mbps=demand_mbps)
    return _reward_fn


# Hangi adımı seçeceğime nasıl karar veriyorum? keşfetme vs. sömürü dengesi
def choose_action(state, neighbors, Q, epsilon):
    if random.random() < epsilon: # 0 ile 1 arasında rastgele sayı < "epsilon" ise
        return random.choice(neighbors[state]) # rastgele komşuya git
    actions = neighbors[state] # mevcut düğümün komşuları
    q_vals = [Q[(state, a)] for a in actions] # her eylem için Q değeri
    max_q = max(q_vals) # en yüksek Q değeri
    best_actions = [a for a, q in zip(actions, q_vals) if q == max_q] # en iyi Q değerine sahip eylemler
    return random.choice(best_actions)  # en iyi eylemlerden rastgele seçim

# bir adım attığımda ortamda ne oluyor?
def step(state, action, goal_node, G, reward_fn, fail_with_reliability=False):
    s_next = action

    if fail_with_reliability:
        edge = G.edges[state, s_next]
        node_next = G.nodes[s_next]
        success_p = float(edge['reliability']) * float(node_next['reliability'])
        if random.random() > success_p:
            return state, -100.0, True

    r = reward_fn(state, action, s_next, G)
    terminated = False

    if s_next == goal_node:
        r += 50.0
        terminated = True

    return s_next, r, terminated

# Bu fonksiyon, her episode’da başlangıç düğümünden başlayarak epsilon-greedy politika ile adımlar atar, her adımda ödül ve sonraki durumun en iyi Q değerine göre Q(s,a) değerlerini güncelleyerek optimal politikayı öğrenir.
def q_learning(
    G, neighbors, start_node, goal_node,
    reward_fn, episodes=5000,
    alpha=0.1, gamma=0.95,
    epsilon_start=1.0, # yüzde yüz keşif ile başla
    epsilon_end=0.05,
    epsilon_decay_steps=3000,
    max_steps_per_episode=200,
    stochastic_fail=False
):
    """
    episodes: kaç kez deneme yapacak (ne kadar çok → o kadar öğrenir)
    alpha: öğrenme oranı (yeni bilgi eskiyi ne kadar değiştirsin)
    gamma: gelecek ödülün önemi (0.0 sadece anlık, 0.99 uzun vadeli)
    epsilon_start/end/decay: keşiften sömürüye geçiş planı
    max_steps_per_episode: episode sonsuza gitmesin diye güvenlik limiti
    stochastic_fail: reliability’ye bağlı rastgele “çökme” simülasyonu

    """
    Q = defaultdict(float) # Q-değerleri için sözlük, başlangıçta tüm değerler 0.0
    epsilon = epsilon_start
    eps_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
    # her episode sonunda epsilon’ı azaltmak için adım miktarı

    for _ in range(episodes):
        s = start_node
        for _t in range(max_steps_per_episode):
            if len(neighbors.get(s, [])) == 0:
                break

            a = choose_action(s, neighbors, Q, epsilon)
            s_next, r, done = step(s, a, goal_node, G, reward_fn, fail_with_reliability=stochastic_fail)

            best_next = 0.0
            if len(neighbors.get(s_next, [])) > 0:
                best_next = max(Q[(s_next, a2)] for a2 in neighbors[s_next])

            Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma * best_next - Q[(s, a)])
            s = s_next

            if done:
                break

        if epsilon > epsilon_end:
            epsilon = max(epsilon_end, epsilon - eps_decay)

    return Q

#Q-learning sonucu elde edilen Q tablosuna bakarak, başlangıç düğümünden hedef düğüme en yüksek Q-değerlerini izleyen yolu çıkarır. öğrenilmiş Q-değerlerini kullanarak her adımda en yüksek Q-değerine sahip eylemi seçip, döngüleri engelleyerek başlangıç düğümünden hedef düğümüne giden en iyi yolu üretir.
def greedy_path(Q, neighbors, start, goal, max_len=200):
    """

    Q → öğrenilmiş Q tablosu (Q[(state, action)])
    neighbors → her düğümden gidilebilecek komşular
    start → başlangıç düğümü
    goal → hedef düğümü
    max_len → güvenlik limiti (sonsuz döngü olmasın diye)

    """
    path = [start]
    s = start
    visited = set([start])

    for _ in range(max_len): # maksimum adım sayısı
        if s == goal: # hedefe ulaşıldıysa
            break
        if len(neighbors.get(s, [])) == 0: # gidilecek yer yoksa dur
            break

        actions = neighbors[s]
        q_vals = [Q[(s, a)] for a in actions] # Burada hesaplama yok, sadece okuma var. 
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(actions, q_vals) if q == max_q]

        a = random.choice(best_actions)
        s = a
        path.append(s)

        if s in visited and s != goal:
            break
        visited.add(s)

    return path

# verilen bir yol üzerindeki bağlantı ve düğüm özelliklerini kullanarak yolun gecikmesini, güvenilirliğini ve kapasite kısıtını hesaplar.
def path_metrics(path, G):
    total_delay = 0.0
    total_rel = 1.0
    bottleneck_bw = float('inf')

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        e = G.edges[u, v]
        n = G.nodes[v]

        total_delay += float(e['delay']) + float(n['processing_delay'])
        total_rel   *= float(e['reliability']) * float(n['reliability'])
        bottleneck_bw = min(bottleneck_bw, float(e['bandwidth']))

    return total_delay, total_rel, bottleneck_bw

def multi_cost(delay_ms, rel, bottleneck_bw, weights):
    unreliab = 1.0 - rel
    inv_bw = 100.0 / max(1.0, bottleneck_bw)
    return (weights["w_delay"] * delay_ms +
            weights["w_rel"]   * unreliab +
            weights["w_bw"]    * inv_bw)

def demand_feasible(path, G, demand_mbps):
    if demand_mbps is None:
        return True
    _, _, bottleneck = path_metrics(path, G)
    return bottleneck >= demand_mbps

# =========================================================
# 4) MAIN (ÇALIŞTIRILINCA AKAN KISIM)
# =========================================================
def main():
    print("\n--- Kaynak/ Hedef seçimi ---")
    if demand_df is not None and len(demand_df) > 0:
        print(f"Demand dosyası bulundu: {DEMAND_CSV} (satır sayısı: {len(demand_df)})")
        use_from_file = input("Demand içinden bir satır seçmek ister misin? (E/H) [E]: ").strip().lower()
        if use_from_file in ("", "e", "evet", "y", "yes"):
            idx_in = input(f"Satır index gir (0..{len(demand_df)-1}) [0]: ").strip()
            idx = int(idx_in) if idx_in else 0
            idx = max(0, min(idx, len(demand_df) - 1))
            row = demand_df.iloc[idx]
            start_node = int(row["src"])
            goal_node  = int(row["dst"])
            demand_mbps = float(row["demand_mbps"])
            print(f"Seçilen demand: src={start_node}, dst={goal_node}, demand={demand_mbps} Mbps")
        else:
            start_node = int(input("Başlangıç düğümü: "))
            goal_node  = int(input("Hedef düğümü: "))
            d_in = input("Demand (Mbps) [boş=kapasite kısıtı yok]: ").strip()
            seed_in = input("Seed (boş bırak = rastgele): ").strip()
            seed = int(seed_in) if seed_in else None
            set_seed(seed)
            demand_mbps = safe_float(d_in) if d_in else None
    else:
        start_node = int(input("Başlangıç düğümü: "))
        goal_node  = int(input("Hedef düğümü: "))
        d_in = input("Demand (Mbps) [boş=kapasite kısıtı yok]: ").strip()
        seed_in = input("Seed (boş bırak = rastgele): ").strip()
        seed = int(seed_in) if seed_in else None
        set_seed(seed)
        demand_mbps = safe_float(d_in) if d_in else None

    print("\n--- Ağırlıklar (W) ---")
    print("W_delay, W_rel, W_bw gir. (2-1-1 gibi girersen ben normalize ederim -> toplam=1)")
    w_delay_in = input("W_delay (gecikme) [varsayılan 1.0]: ").strip()
    w_rel_in   = input("W_rel (güvenirlik) [varsayılan 1.0]: ").strip()
    w_bw_in    = input("W_bw (kapasite) [varsayılan 1.0]: ").strip()

    w_delay = safe_float(w_delay_in) if w_delay_in else DEFAULT_WEIGHTS["w_delay"]
    w_rel   = safe_float(w_rel_in)   if w_rel_in   else DEFAULT_WEIGHTS["w_rel"]
    w_bw    = safe_float(w_bw_in)    if w_bw_in    else DEFAULT_WEIGHTS["w_bw"]

    weights = normalize_weights(w_delay, w_rel, w_bw)
    print(f"\nNormalize edilmiş ağırlıklar: {weights} (toplam=1)")

    if demand_mbps is None:
        print("Demand kısıtı: YOK (bandwidth sadece maliyette kullanılacak)")
    else:
        print(f"Demand kısıtı: VAR -> Her adımda bandwidth >= {demand_mbps} Mbps olmalı (hard constraint)")

    reward_fn = make_reward_fn(weights, demand_mbps=demand_mbps)

    Q = q_learning(
        G, neighbors,
        start_node=start_node, goal_node=goal_node,
        reward_fn=reward_fn,
        episodes=6000, alpha=0.10, gamma=0.97,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=4000,
        max_steps_per_episode=200,
        stochastic_fail=False
    )
    """
    
    episodes=6000: 6000 kez start→goal denemesi
    alpha=0.10: yeni bilgi %15 etkili
    gamma=0.97: uzun vadeyi çok önemsiyor
    epsilon_start=1.0 → başta %100 keşif
    epsilon_end=0.05 → sonda %5 keşif
    epsilon_decay_steps=4000 → 4000 episode boyunca epsilon azalır
    max_steps_per_episode=200: dolaşıp durmasın
    stochastic_fail=False: reliability’ye göre rastgele çökme kapalı

    """

    best_path = greedy_path(Q, neighbors, start_node, goal_node)
    print("\nQ-learning ile bulunan greedy yol:", best_path)

    d, r, bw = path_metrics(best_path, G)
    tcost = multi_cost(d, r, bw, weights)

    print(f"\nDelay(ms)={d:.3f} | Reliability={r:.6f} | BottleneckBW(Mbps)={bw:.1f}")
    print(f"Multi-objective cost = {tcost:.6f}")

    if demand_mbps is not None:
        ok = demand_feasible(best_path, G, demand_mbps)
        if ok:
            print(f"✅ Demand uygun: bottleneck {bw:.1f} >= demand {demand_mbps:.1f} Mbps")
        else:
            print(f"❌ Demand uygun değil: bottleneck {bw:.1f} < demand {demand_mbps:.1f} Mbps")
            print("   (Bu durumda ya graph'ta uygun yol yok, ya eğitim/parametreleri artırmak gerekir.)")

if __name__ == "__main__":
    main()
