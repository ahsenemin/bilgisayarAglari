import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1. DOSYALARI OKUMA (Senin dosya isimlerinle)
# Ayraç olarak ';' ve ondalık sayı için ',' kullanıldığını belirtiyoruz.
nodes_df = pd.read_csv('BSM307_317_Guz2025_TermProject_NodeData.csv', sep=';', decimal=',')
edges_df = pd.read_csv('BSM307_317_Guz2025_TermProject_EdgeData.csv', sep=';', decimal=',')

# 2. GRAF NESNESİNİ OLUŞTURMA
G = nx.Graph()

# 3. DÜĞÜMLERİ EKLEME (Özellikleriyle birlikte)
print("Düğümler ekleniyor...")
for index, row in nodes_df.iterrows():
    G.add_node(
        int(row['node_id']), 
        processing_delay=row['s_ms'],   # İşlem süresi
        reliability=row['r_node']       # Güvenilirlik
    )

# 4. BAĞLANTILARI EKLEME (Özellikleriyle birlikte)
print("Bağlantılar ekleniyor...")
for index, row in edges_df.iterrows():
    G.add_edge(
        int(row['src']), 
        int(row['dst']), 
        bandwidth=row['capacity_mbps'], # Bant genişliği
        delay=row['delay_ms'],          # Gecikme
        reliability=row['r_link']       # Güvenilirlik
    )


# 5. SONUÇ RAPORU
# Bu kısmı if bloğu içine alıyoruz ki import ettiğimizde çalışmasın
if __name__ == "__main__":
    print("-" * 30)
    print(f"BAŞARILI: Ağ Topolojisi Oluşturuldu.")
    print(f"Toplam Düğüm Sayısı: {G.number_of_nodes()}")
    print(f"Toplam Bağlantı Sayısı: {G.number_of_edges()}")
    print("-" * 30)

    # (İsteğe Bağlı) KÜÇÜK BİR GÖRSELLEŞTİRME
    plt.figure(figsize=(10, 8))
    nx.draw(G, node_size=10, node_color='blue', with_labels=False, alpha=0.6)
    plt.title("Oluşturulan Ağ Topolojisi (250 Düğüm)")
    plt.show() # İŞTE KODU KİLİTLEYEN SATIR BU, ARTIK SADECE BU DOSYA ÇALIŞTIRILINCA AÇILACAK