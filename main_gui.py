import sys
import random
import math
import time
import csv
import datetime
import networkx as nx
from collections import defaultdict

# --- Pastikan QSlider ada di import ---
from PySide6.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, 
                               QGraphicsView, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
                               QPushButton, QDoubleSpinBox, QSpinBox, QFrame, QComboBox, 
                               QTextEdit, QMessageBox, QProgressBar, QGridLayout, QGroupBox, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                               QSlider) # <--- TAMBAHKAN QSLIDER DISINI
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QFont

# ==============================================================================
# 1. DOSYA VE MODÜL KONTROLLERİ
# ==============================================================================
try: import ag; AG_AVAILABLE = True
except ImportError: AG_AVAILABLE = False; print("UYARI: ag.py (Ağ Oluşturucu) bulunamadı!")

try: from genetik_proje import GenetikAlgoritma; GA_AVAILABLE = True
except ImportError: GA_AVAILABLE = False; print("UYARI: genetik_proje.py bulunamadı!")

try: from karinca import ACORouting; ACO_AVAILABLE = True
except ImportError: ACO_AVAILABLE = False; print("UYARI: karinca.py bulunamadı!")

try: import q_learning as ql; RL_AVAILABLE = True
except ImportError: RL_AVAILABLE = False; print("UYARI: q_learning.py bulunamadı!")

DEMAND_FILE = "BSM307_317_Guz2025_TermProject_DemandData.csv"

# ==============================================================================
# 2. LOGIKA BACKEND (HESAPLAMA MOTORU)
# ==============================================================================
def calculate_path_metrics_detailed(graph, path):
    if not path or len(path) < 2: return 0, 0, 0
    total_delay = 0.0; rel_cost = 0.0; res_cost = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if not graph.has_edge(u, v): return 0,0,0
        edge = graph[u][v]
        total_delay += edge.get('delay', 0)
        r_link = edge.get('reliability', 0.99)
        rel_cost += -math.log(r_link if r_link > 0 else 1e-6)
        bw = edge.get('bandwidth', 100)
        res_cost += (1000.0 / (bw if bw > 0 else 1))

    for i, node in enumerate(path):
        n_data = graph.nodes[node]
        r_node = n_data.get('reliability', 0.99)
        rel_cost += -math.log(r_node if r_node > 0 else 1e-6)
        if i != 0 and i != len(path) - 1:
            total_delay += n_data.get('processing_delay', 0)

    return total_delay, rel_cost, res_cost

class NetworkManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.pos_cache = {} 
        self.load_from_ag()

    def load_from_ag(self):
        if AG_AVAILABLE and hasattr(ag, 'G'):
            orig = ag.G
            self.graph = orig.to_directed() if not orig.is_directed() else orig.copy()
            self.calculate_layout(seed=42)
        else:
            self.graph = nx.DiGraph()

    def calculate_layout(self, seed=42):
        if self.graph.number_of_nodes() > 0:
            nodes_sorted = sorted(list(self.graph.nodes()))
            temp_G = self.graph.subgraph(nodes_sorted)
            
            # --- PERBAIKAN: Jarak Antar Node Lebih Lebar ---
            # 'k' menentukan jarak optimal. Semakin besar k, semakin renggang.
            # Kita buat dinamis berdasarkan jumlah node.
            k_val = 2.0 / math.sqrt(self.graph.number_of_nodes()) 
            
            raw_pos = nx.spring_layout(temp_G, seed=seed, k=k_val, iterations=100)
            
            # --- PERBAIKAN: Skala Koordinat Sangat Besar ---
            # Sebelumnya 650, sekarang 2500 agar node benar-benar menyebar
            cx, cy, scale = 400, 400, 2500  
            
            self.pos_cache = {} 
            for n, p in raw_pos.items():
                self.pos_cache[n] = (cx + p[0]*scale, cy + p[1]*scale)

class RouteSolver:
    def __init__(self, manager): 
        self.net = manager

    def solve(self, algo_type, src, dst, weights, demand_bw=4.0):
        if self.net.graph.number_of_nodes() == 0: return None, {}, 0
        path, duration, metrics = None, 0, {}
        graph_copy = self.net.graph.copy()

        try:
            if algo_type == "Genetik Algoritma (GA)" and GA_AVAILABLE:
                ga = GenetikAlgoritma(graph_copy, src, dst, 50, 0.1, 50, weights)
                path, _, duration = ga.calistir()

            elif algo_type == "Karınca Kolonisi (ACO)" and ACO_AVAILABLE:
                start_t = time.time()
                aco = ACORouting(graph_copy, src, dst, demand_bw, weights, 20, 30)
                path = aco.solve()[0]
                duration = time.time() - start_t

            elif algo_type == "Q-Öğrenme (RL)" and RL_AVAILABLE:
                start_t = time.time()
                nbrs = {n: list(graph_copy.neighbors(n)) for n in graph_copy.nodes()}
                w_dict = {"w_delay": weights[0], "w_rel": weights[1], "w_bw": weights[2]}
                r_fn = ql.make_reward_fn(w_dict, demand_mbps=demand_bw)
                Q = ql.q_learning(graph_copy, nbrs, src, dst, r_fn, 800, 0.15, 0.97, 200, 0.9, 0.05, 600)
                path = ql.greedy_path(Q, nbrs, src, dst)
                if not path or path[-1] != dst: path = None
                duration = time.time() - start_t
            
            if path:
                d, r, c = calculate_path_metrics_detailed(self.net.graph, path)
                metrics = {'delay': d, 'rel_cost': r, 'res_cost': c}

        except Exception as e: 
            print(f"Hata: {e}")
            import traceback
            traceback.print_exc()

        return path, metrics, duration

# ==============================================================================
# 3. İŞ PARÇACIKLARI (WORKERS)
# ==============================================================================
class CalculationWorker(QThread):
    result_ready = Signal(str, object, object, float); finished_all = Signal()
    def __init__(self, solver, mode, params):
        super().__init__(); self.solver, self.mode, self.params = solver, mode, params
    def run(self):
        s, d, w1, w2, w3, algo = self.params; weights = [w1, w2, w3]
        if self.mode == "Single":
            p, m, t = self.solver.solve(algo, s, d, weights)
            self.result_ready.emit(algo, p, m, t)
        elif self.mode == "Compare":
            for name in ["Genetik Algoritma (GA)", "Karınca Kolonisi (ACO)", "Q-Öğrenme (RL)"]:
                p, m, t = self.solver.solve(name, s, d, weights)
                self.result_ready.emit(name, p, m, t)
        self.finished_all.emit()

class BatchExperimentWorker(QThread):
    log_signal = Signal(str); progress_signal = Signal(int); 
    finished_signal = Signal(str); finished_stats = Signal(dict) 

    def __init__(self, solver, weights):
        super().__init__(); self.solver, self.weights, self.is_running = solver, weights, True
    def run(self):
        self.log_signal.emit("🚀 Otomatik Deney Başlatılıyor..."); demands = []
        try:
            with open(DEMAND_FILE, 'r') as f:
                reader = csv.reader(f, delimiter=';'); next(reader)
                for row in reader:
                    if len(row) < 3: continue
                    demands.append((int(row[0]), int(row[1]), float(row[2].replace(',', '.'))))
        except FileNotFoundError:
            self.log_signal.emit(f"❌ HATA: {DEMAND_FILE} bulunamadı!"); self.finished_signal.emit("Error"); return

        TARGET_DEMANDS = demands[:5]; REPEATS = 3; 
        algos = ["Genetik Algoritma (GA)", "Karınca Kolonisi (ACO)", "Q-Öğrenme (RL)"]
        total_steps = len(TARGET_DEMANDS) * len(algos) * REPEATS; current_step = 0
        
        agg_stats = {a: {'cost': 0, 'time': 0, 'success': 0, 'count': 0} for a in algos}
        report_lines = [f"DENEY RAPORU - {datetime.datetime.now()}", f"Ağırlıklar (Gecikme/Güv/Kaynak): {self.weights}", "-" * 50]

        for idx, (s, d, bw) in enumerate(TARGET_DEMANDS):
            if not self.is_running: break
            header = f"\nTALEP #{idx+1}: S={s} -> D={d} (İstenen BW: {bw} Mbps)"
            self.log_signal.emit(header); report_lines.append(header)
            
            for algo in algos:
                sc, tc, tt = 0, 0, 0
                for r in range(REPEATS):
                    path, mets, dur = self.solver.solve(algo, s, d, self.weights, demand_bw=bw)
                    agg_stats[algo]['count'] += 1
                    agg_stats[algo]['time'] += dur
                    
                    if path:
                        cost = (self.weights[0]*mets['delay']) + (self.weights[1]*mets['rel_cost']*100) + (self.weights[2]*mets['res_cost'])
                        sc += 1; tc += cost; tt += dur
                        agg_stats[algo]['cost'] += cost
                        agg_stats[algo]['success'] += 1
                    
                    current_step += 1; self.progress_signal.emit(int((current_step / total_steps) * 100))
                
                avg_c = tc/sc if sc>0 else 0; avg_t = tt/REPEATS
                res_str = f"  [{algo}] Başarı: {sc}/{REPEATS} | Ort. Maliyet: {avg_c:.2f} | Ort. Süre: {avg_t:.4f}s"
                self.log_signal.emit(res_str); report_lines.append(res_str)

        filename = f"deney_sonuc_{int(time.time())}.txt"
        with open(filename, "w", encoding="utf-8") as f: f.write("\n".join(report_lines))
        
        self.log_signal.emit(f"\n✅ Deney Tamamlandı! Rapor: {filename}")
        self.finished_stats.emit(agg_stats)
        self.finished_signal.emit(filename)

# ==============================================================================
# 4. MODERN ARAYÜZ (TÜRKÇE UI)
# ==============================================================================
class NetworkVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSM307 - QoS Odaklı Akıllı Ağ Rotalama")
        self.resize(1400, 900)
        self.net = NetworkManager()
        self.solver = RouteSolver(self.net)
        
        # --- STYLESHEET ---
        self.setStyleSheet("""
            QMainWindow { background-color: #0f111a; font-family: 'Segoe UI', Roboto, sans-serif; }
            QFrame#Panel { background-color: #1a1c29; border-radius: 15px; border: 1px solid #2f334d; }
            QLabel { color: #a9b1d6; font-size: 13px; }
            QLabel#Header { color: #7aa2f7; font-size: 15px; font-weight: 900; letter-spacing: 1px; padding-bottom: 5px; border-bottom: 2px solid #7aa2f7; }
            QLabel#InputLabel { color: #bb9af7; font-weight: bold; font-size: 12px; margin-bottom: 2px; }
            QGroupBox { border: 1px solid #414868; border-radius: 10px; margin-top: 22px; font-weight: bold; color: #bb9af7; }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; background-color: #1a1c29; }
            QSpinBox, QComboBox { background-color: #24283b; color: white; padding: 8px; border: 1px solid #414868; border-radius: 6px; font-weight: bold; }
            QSlider::groove:horizontal { border: 1px solid #414868; height: 8px; background: #24283b; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #7aa2f7; border: 1px solid #7aa2f7; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; }
            QPushButton { background-color: #3b4261; color: white; border-radius: 8px; padding: 10px; font-weight: bold; border: 1px solid #414868; }
            QPushButton:hover { background-color: #414868; border: 1px solid #7aa2f7; }
            QPushButton#BtnRun { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7aa2f7, stop:1 #2ac3de); color: #0f111a; border: none; }
            QPushButton#BtnCmp { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #bb9af7, stop:1 #f7768e); color: #0f111a; border: none; }
            QPushButton#BtnBatch { background-color: #e0af68; color: #0f111a; border: none; }
            QTextEdit { background-color: #0f111a; color: #73daca; font-family: Consolas; border: 1px solid #2f334d; border-radius: 8px; }
            QProgressBar { border: none; background: #24283b; height: 6px; border-radius: 3px; }
            QProgressBar::chunk { background: #7aa2f7; border-radius: 3px; }
            QTableWidget { background-color: #0f111a; color: #c0caf5; border: 1px solid #414868; gridline-color: #2f334d; font-family: Consolas; font-size: 12px; }
            QHeaderView::section { background-color: #1a1c29; color: #7aa2f7; padding: 4px; border: 1px solid #2f334d; font-weight: bold; }
            QTableWidget::item { padding: 5px; }
            QTableWidget::item:selected { background-color: #3b4261; }
        """)

        central = QWidget(); self.setCentralWidget(central)
        main_lo = QHBoxLayout(central); main_lo.setSpacing(20); main_lo.setContentsMargins(20,20,20,20)

        # ==============================================================================
        # 1. PANEL KIRI (INPUT) - TAMPILAN BARU
        # ==============================================================================
        left = QFrame(); left.setObjectName("Panel"); left.setFixedWidth(340)
        left_lo = QVBoxLayout(left); left_lo.setSpacing(15); left_lo.setContentsMargins(20,20,20,20)

        title = QLabel("KONTROL MERKEZİ"); title.setObjectName("Header"); title.setAlignment(Qt.AlignCenter)
        left_lo.addWidget(title)

        # --- BAGIAN TOPOLOGI (EDITED: LABEL DI ATAS) ---
        grp_route = QGroupBox("TOPOLOJİ AYARLARI")
        # Gunakan HBox untuk menjejerkan Sumber dan Tujuan
        route_inner_lo = QHBoxLayout(grp_route); route_inner_lo.setSpacing(15)
        
        max_id = max(list(self.net.graph.nodes)) if self.net.graph.nodes else 0
        
        # Wadah Sumber
        src_cont = QWidget()
        src_vlo = QVBoxLayout(src_cont); src_vlo.setContentsMargins(0,0,0,0); src_vlo.setSpacing(5)
        lbl_s = QLabel("Kaynak (Source)"); lbl_s.setObjectName("InputLabel")
        self.spin_s = QSpinBox(); self.spin_s.setRange(0, max_id)
        src_vlo.addWidget(lbl_s); src_vlo.addWidget(self.spin_s)
        
        # Wadah Tujuan
        dst_cont = QWidget()
        dst_vlo = QVBoxLayout(dst_cont); dst_vlo.setContentsMargins(0,0,0,0); dst_vlo.setSpacing(5)
        lbl_d = QLabel("Hedef (Dest)"); lbl_d.setObjectName("InputLabel")
        self.spin_d = QSpinBox(); self.spin_d.setRange(0, max_id); self.spin_d.setValue(max_id)
        dst_vlo.addWidget(lbl_d); dst_vlo.addWidget(self.spin_d)

        route_inner_lo.addWidget(src_cont)
        route_inner_lo.addWidget(dst_cont)
        left_lo.addWidget(grp_route)

        # --- BAGIAN OPTIMASI & BOBOT (EDITED: SLIDER VERTICAL) ---
        grp_algo = QGroupBox("OPTİMİZASYON STRATEJİSİ")
        g_al_lo = QVBoxLayout(grp_algo); g_al_lo.setSpacing(15)
        
        # Pilihan Algoritma
        algo_cont = QWidget()
        algo_vlo = QVBoxLayout(algo_cont); algo_vlo.setContentsMargins(0,0,0,0); algo_vlo.setSpacing(5)
        algo_vlo.addWidget(QLabel("Algoritma Seçimi:", objectName="InputLabel"))
        self.combo = QComboBox(); self.combo.addItems(["Genetik Algoritma (GA)", "Karınca Kolonisi (ACO)", "Q-Öğrenme (RL)"])
        algo_vlo.addWidget(self.combo)
        g_al_lo.addWidget(algo_cont)
        
        # Judul Bobot
        g_al_lo.addWidget(QLabel("Ağırlıklar (Slider):", objectName="InputLabel"))

        # Fungsi Pembantu membuat Slider
        def create_slider_row(text, default_val):
            container = QWidget()
            v_layout = QVBoxLayout(container); v_layout.setContentsMargins(0,0,0,0); v_layout.setSpacing(2)
            
            # Label Judul Kecil
            lbl_title = QLabel(text); lbl_title.setStyleSheet("color: #a9b1d6; font-size: 11px;")
            v_layout.addWidget(lbl_title)
            
            # Baris Slider + Angka
            h_layout = QHBoxLayout(); h_layout.setContentsMargins(0,0,0,0)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(default_val * 100))
            
            lbl_val = QLabel(f"{default_val:.2f}")
            lbl_val.setStyleSheet("color: #7aa2f7; font-weight: bold; font-family: Consolas;")
            lbl_val.setFixedWidth(40); lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            # Update label saat digeser
            slider.valueChanged.connect(lambda v: lbl_val.setText(f"{v/100:.2f}"))
            
            h_layout.addWidget(slider)
            h_layout.addWidget(lbl_val)
            v_layout.addLayout(h_layout)
            
            return container, slider

        # Membuat 3 Slider secara Vertikal (Kebawah)
        w1_wid, self.slider_w1 = create_slider_row("Gecikme (Delay)", 0.4)
        w2_wid, self.slider_w2 = create_slider_row("Güvenilirlik (Reliability)", 0.3)
        w3_wid, self.slider_w3 = create_slider_row("Kaynak (Resource)", 0.3)

        g_al_lo.addWidget(w1_wid)
        g_al_lo.addWidget(w2_wid)
        g_al_lo.addWidget(w3_wid)
        
        left_lo.addWidget(grp_algo)

        left_lo.addSpacing(10)
        self.btn_run = QPushButton("▶  HESAPLA (BAŞLAT)"); self.btn_run.setObjectName("BtnRun") 
        self.btn_run.clicked.connect(self.start_single)
        left_lo.addWidget(self.btn_run)

        btn_row = QHBoxLayout()
        self.btn_cmp = QPushButton("▶  KARŞILAŞTIR"); self.btn_cmp.setObjectName("BtnCmp")
        self.btn_cmp.clicked.connect(self.start_compare)
        self.btn_batch = QPushButton("▶  TOPLU DENEY"); self.btn_batch.setObjectName("BtnBatch")
        self.btn_batch.clicked.connect(self.start_batch_experiment)
        btn_row.addWidget(self.btn_cmp); btn_row.addWidget(self.btn_batch)
        left_lo.addLayout(btn_row)

        self.pbar = QProgressBar(); self.pbar.setVisible(False)
        left_lo.addWidget(self.pbar)
        
        left_lo.addStretch()
        main_lo.addWidget(left)

        # ==============================================================================
        # 2. PANEL TENGAH (GRAFİK) - Tidak Berubah
        # ==============================================================================
        center = QFrame(); center.setObjectName("Panel")
        center_lo = QVBoxLayout(center); center_lo.setContentsMargins(15,15,15,15)
        
        tool_lo = QHBoxLayout()
        lbl_vis = QLabel("AĞ HARİTASI (250 Düğüm)"); lbl_vis.setObjectName("Header"); 
        tool_lo.addWidget(lbl_vis)
        tool_lo.addStretch()
        btn_redraw = QPushButton("🔄 Karıştır"); btn_redraw.setFixedWidth(80); btn_redraw.clicked.connect(self.redraw_network)
        tool_lo.addWidget(btn_redraw)
        center_lo.addLayout(tool_lo)

        self.scene = QGraphicsScene()
        
        # --- PERBAIKAN: SceneRect Sangat Luas ---
        # Karena skala koordinat diperbesar jadi 2500, canvas harus diperbesar juga
        # Rentang -2500 sampai 3500 (total 6000x6000 px)
        self.scene.setSceneRect(-2500, -2500, 6000, 6000) 

        self.view = QGraphicsView(self.scene); self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setBackgroundBrush(QBrush(QColor("#1a1c29"))); self.view.setStyleSheet("border: none; border-radius: 8px;")
        
        # Biarkan user bisa scroll/pan bebas
        self.view.setDragMode(QGraphicsView.ScrollHandDrag) 
        
        center_lo.addWidget(self.view)

        legend_lo = QHBoxLayout(); legend_lo.setContentsMargins(0, 5, 0, 0)
        def mk_leg(col, txt):
            l_col = QLabel("●"); l_col.setStyleSheet(f"color: {col}; font-size: 18px;")
            l_txt = QLabel(txt); l_txt.setStyleSheet("color: #a9b1d6; margin-right: 15px; font-weight: bold;")
            legend_lo.addWidget(l_col); legend_lo.addWidget(l_txt)
        mk_leg("#9ece6a", "Kaynak (S)"); mk_leg("#f7768e", "Hedef (D)"); mk_leg("#e0af68", "Seçilen Yol"); mk_leg("#565f89", "Düğüm")
        legend_lo.addStretch(); center_lo.addLayout(legend_lo)
        
        main_lo.addWidget(center, stretch=1)

        # ==============================================================================
        # 3. PANEL KANAN (HASIL & LOG) - Tidak Berubah
        # ==============================================================================
        right = QFrame(); right.setObjectName("Panel"); right.setFixedWidth(340)
        right_lo = QVBoxLayout(right); right_lo.setSpacing(15); right_lo.setContentsMargins(20,20,20,20)

        lbl_res = QLabel("ANALİZ RAPORU"); lbl_res.setObjectName("Header"); lbl_res.setAlignment(Qt.AlignCenter)
        right_lo.addWidget(lbl_res)

        self.lbl_status = QLabel("GİRİŞ BEKLENİYOR"); self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("background-color: #24283b; color: #565f89; font-weight: 900; font-size: 16px; border-radius: 10px; padding: 20px; border: 2px dashed #414868;")
        right_lo.addWidget(self.lbl_status)

        # --- GRUP 1: TEKLİ ANALİZ DETAYLARI ---
        self.grp_single_details = QWidget()
        single_lo = QVBoxLayout(self.grp_single_details); single_lo.setContentsMargins(0,0,0,0); single_lo.setSpacing(15)
        
        self.grp_overview = QGroupBox("Genel Bakış")
        ov_lo = QVBoxLayout(self.grp_overview); ov_lo.setSpacing(12)
        
        def mk_row(icon, label):
            w = QWidget(); hl = QHBoxLayout(w); hl.setContentsMargins(0,0,0,0)
            l1 = QLabel(f"{icon}  {label}"); l1.setStyleSheet("color: #c0caf5; font-weight: bold;")
            l2 = QLabel("-"); l2.setAlignment(Qt.AlignRight); l2.setStyleSheet("color: #7aa2f7; font-family: Consolas;")
            hl.addWidget(l1); hl.addWidget(l2)
            return w, l2

        w_time, self.lbl_time = mk_row("⏱️", "Çalışma Süresi")
        w_score, self.lbl_score = mk_row("🏆", "Toplam Uygunluk")
        ov_lo.addWidget(w_time); ov_lo.addWidget(w_score)
        
        lbl_r_head = QLabel("🛣️ İzlenen Rota:"); lbl_r_head.setStyleSheet("color: #bb9af7; font-weight: bold;")
        ov_lo.addWidget(lbl_r_head)
        self.lbl_route = QLabel("-"); self.lbl_route.setWordWrap(True); self.lbl_route.setAlignment(Qt.AlignCenter)
        self.lbl_route.setStyleSheet("background-color: #24283b; color: #7aa2f7; font-family: Consolas; font-size: 11px; padding: 8px; border-radius: 6px; border: 1px solid #414868;")
        ov_lo.addWidget(self.lbl_route)
        single_lo.addWidget(self.grp_overview)

        self.grp_qos = QGroupBox("QoS Detayları")
        qos_lo = QVBoxLayout(self.grp_qos); qos_lo.setSpacing(12)
        w_d, self.lbl_delay = mk_row("⚡", "Toplam Gecikme")
        w_r, self.lbl_rel = mk_row("🛡️", "Güvenilirlik Maliyeti")
        w_b, self.lbl_res = mk_row("💰", "Kaynak Maliyeti")
        qos_lo.addWidget(w_d); qos_lo.addWidget(w_r); qos_lo.addWidget(w_b)
        single_lo.addWidget(self.grp_qos)
        right_lo.addWidget(self.grp_single_details)

        # --- GRUP 2: KARŞILAŞTIRMA TABLOSU ---
        self.grp_compare_table = QGroupBox("Karşılaştırma Sonuçları")
        self.grp_compare_table.setVisible(False)
        cmp_lo = QVBoxLayout(self.grp_compare_table)
        
        self.table_res = QTableWidget()
        self.table_res.setColumnCount(3)
        self.table_res.setHorizontalHeaderLabels(["Algoritma", "Maliyet", "Süre"])
        self.table_res.verticalHeader().setVisible(False)
        self.table_res.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_res.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table_res.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        cmp_lo.addWidget(self.table_res)
        right_lo.addWidget(self.grp_compare_table)

        # --- GRUP 3: TOPLU DENEY TABLOSU ---
        self.grp_batch_table = QGroupBox("Toplu Deney Sonuçları (Ort.)")
        self.grp_batch_table.setVisible(False)
        batch_lo = QVBoxLayout(self.grp_batch_table)

        self.table_batch = QTableWidget()
        self.table_batch.setColumnCount(4)
        self.table_batch.setHorizontalHeaderLabels(["Alg.", "Maliyet", "Süre", "Başarı"])
        self.table_batch.verticalHeader().setVisible(False)
        self.table_batch.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_batch.setSelectionBehavior(QAbstractItemView.SelectRows)
        b_header = self.table_batch.horizontalHeader()
        b_header.setSectionResizeMode(0, QHeaderView.Stretch)
        b_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        b_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        b_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        batch_lo.addWidget(self.table_batch)
        right_lo.addWidget(self.grp_batch_table)

        # --- Log System ---
        right_lo.addSpacing(10)
        lbl_log = QLabel("Sistem Logları:"); lbl_log.setStyleSheet("color: #73daca; font-weight: bold; margin-top: 5px;")
        right_lo.addWidget(lbl_log)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        right_lo.addWidget(self.log)

        main_lo.addWidget(right)

        # Başlangıç Çizimi
        self.draw_graph_background(); self.path_items = []; self.comp_data = []

        QTimer.singleShot(100, lambda: self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio))

    # --- AUTO FIT RESIZE ---
    def resizeEvent(self, event):
        if hasattr(self, 'view') and hasattr(self, 'scene'):
             self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

    def showEvent(self, event):
        if hasattr(self, 'view') and hasattr(self, 'scene'):
             self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().showEvent(event)

    # --- GÖRSELLEŞTİRME MANTIĞI ---
    def redraw_network(self):
        self.log.append("🔄 Topoloji yeniden yerleştiriliyor...")
        self.net.calculate_layout(seed=random.randint(1, 10000))
        self.draw_graph_background()
        self.path_items = []
        # Fit view lagi setelah redraw
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def draw_graph_background(self):
        self.scene.clear(); self.path_items = []
        if self.net.graph.number_of_nodes() == 0: return
        
        # Garis tipis agar tidak semrawut
        pen = QPen(QColor(86, 95, 137, 30)); pen.setWidth(1) 
        
        for u, v in self.net.graph.edges():
            if u in self.net.pos_cache and v in self.net.pos_cache:
                p1 = self.net.pos_cache[u]; p2 = self.net.pos_cache[v]
                self.scene.addLine(p1[0], p1[1], p2[0], p2[1], pen)
        
        # Node agak kecil agar tidak saling menutupi
        brush = QBrush(QColor("#414868")); pen_n = QPen(Qt.NoPen)
        for n in self.net.graph.nodes():
            if n in self.net.pos_cache:
                x, y = self.net.pos_cache[n]
                # Ukuran node diperkecil sedikit (12px) agar renggang
                self.scene.addEllipse(x-6, y-6, 12, 12, pen_n, brush)
                
                # Opsional: Tampilkan ID node hanya jika di-zoom dekat (level of detail)
                # Tapi untuk performa, kita skip teks ID node di background

    def draw_path(self, path, color=QColor("#e0af68")):
        for item in self.path_items: 
            try: 
                if item.scene() == self.scene: self.scene.removeItem(item)
            except: pass
        self.path_items.clear()
        
        if not path: return

        pen_glow = QPen(color); pen_glow.setWidth(6); pen_glow.setColor(QColor(color.red(), color.green(), color.blue(), 80))
        pen_core = QPen(color); pen_core.setWidth(2)
        
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            if u in self.net.pos_cache and v in self.net.pos_cache:
                p1 = self.net.pos_cache[u]; p2 = self.net.pos_cache[v]
                self.path_items.append(self.scene.addLine(p1[0], p1[1], p2[0], p2[1], pen_glow))
                self.path_items.append(self.scene.addLine(p1[0], p1[1], p2[0], p2[1], pen_core))
        
        if path:
            s, e = path[0], path[-1]
            if s in self.net.pos_cache:
                sp = self.net.pos_cache[s]
                self.path_items.append(self.scene.addEllipse(sp[0]-15, sp[1]-15, 30, 30, QPen(Qt.NoPen), QBrush(QColor("#9ece6a"))))
            if e in self.net.pos_cache:
                ep = self.net.pos_cache[e]
                self.path_items.append(self.scene.addEllipse(ep[0]-15, ep[1]-15, 30, 30, QPen(Qt.NoPen), QBrush(QColor("#f7768e"))))

    # --- BUTON İŞLEVLERİ (MODIFIKASI: Ambil Nilai dari Slider) ---
    def set_ui_busy(self, busy):
        for b in [self.btn_run, self.btn_cmp, self.btn_batch]: b.setEnabled(not busy)
        self.pbar.setVisible(busy); 
        if not busy: self.pbar.setValue(0)
    
    # Helper untuk mengambil nilai float dari slider
    def get_weights(self):
        # Slider range 0-100, dibagi 100 jadi 0.0-1.0
        return self.slider_w1.value()/100.0, self.slider_w2.value()/100.0, self.slider_w3.value()/100.0

    def start_single(self):
        if not AG_AVAILABLE: QMessageBox.critical(self, "Hata", "ag.py eksik!"); return
        self.mode = "Single"; self.set_ui_busy(True)
        
        self.grp_single_details.setVisible(True)
        self.grp_compare_table.setVisible(False)
        self.grp_batch_table.setVisible(False)
        self.lbl_status.setText("HESAPLANIYOR...")
        
        self.log.append("--- Simülasyon Başlatılıyor ---")
        algo = self.combo.currentText()
        w1, w2, w3 = self.get_weights() # Ambil dari slider
        params = (self.spin_s.value(), self.spin_d.value(), w1, w2, w3, algo)
        self.worker = CalculationWorker(self.solver, "Single", params)
        self.worker.result_ready.connect(self.handle_result)
        self.worker.finished_all.connect(lambda: self.set_ui_busy(False))
        self.worker.start()

    def start_compare(self):
        if not AG_AVAILABLE: QMessageBox.critical(self, "Hata", "ag.py eksik!"); return
        self.mode = "Compare"; self.set_ui_busy(True); self.comp_data = []
        
        self.grp_single_details.setVisible(False)
        self.grp_compare_table.setVisible(True)
        self.grp_batch_table.setVisible(False)
        self.table_res.setRowCount(0) 
        self.lbl_status.setText("KIYASLANIYOR...")
        
        self.log.append("--- Karşılaştırma Başlatılıyor ---")
        w1, w2, w3 = self.get_weights() # Ambil dari slider
        params = (self.spin_s.value(), self.spin_d.value(), w1, w2, w3, "ALL")
        self.worker = CalculationWorker(self.solver, "Compare", params)
        self.worker.result_ready.connect(self.handle_result)
        self.worker.finished_all.connect(self.finish_compare)
        self.worker.start()

    def start_batch_experiment(self):
        if not AG_AVAILABLE: return
        self.set_ui_busy(True); self.pbar.setRange(0, 100); self.log.clear()
        
        self.grp_single_details.setVisible(False)
        self.grp_compare_table.setVisible(False)
        self.grp_batch_table.setVisible(True)
        self.table_batch.setRowCount(0)
        self.lbl_status.setText("TOPLU TEST...")

        w1, w2, w3 = self.get_weights() # Ambil dari slider
        weights = [w1, w2, w3]
        self.batch_worker = BatchExperimentWorker(self.solver, weights)
        self.batch_worker.log_signal.connect(self.log.append)
        self.batch_worker.progress_signal.connect(self.pbar.setValue)
        self.batch_worker.finished_signal.connect(self.finish_batch)
        self.batch_worker.finished_stats.connect(self.populate_batch_table) 
        self.batch_worker.start()

    def populate_batch_table(self, stats):
        self.table_batch.setRowCount(len(stats))
        row = 0
        for algo, data in stats.items():
            cnt = data['count'] if data['count'] > 0 else 1
            avg_cost = data['cost'] / data['success'] if data['success'] > 0 else 0
            avg_time = data['time'] / cnt
            success_rate = (data['success'] / cnt) * 100

            short_name = algo.split("(")[-1].strip(")") if "(" in algo else algo
            
            item_name = QTableWidgetItem(short_name)
            item_cost = QTableWidgetItem(f"{avg_cost:.2f}")
            item_time = QTableWidgetItem(f"{avg_time:.4f}s")
            item_succ = QTableWidgetItem(f"%{success_rate:.0f}")
            
            self.table_batch.setItem(row, 0, item_name)
            self.table_batch.setItem(row, 1, item_cost)
            self.table_batch.setItem(row, 2, item_time)
            self.table_batch.setItem(row, 3, item_succ)
            row += 1
        
        self.lbl_status.setText("TAMAMLANDI")
        self.lbl_status.setStyleSheet("background-color: #24283b; color: #9ece6a; font-weight: 900; font-size: 16px; border-radius: 10px; padding: 20px; border: 2px solid #9ece6a;")

    def finish_batch(self, filename):
        self.set_ui_busy(False)
        if filename != "Error": QMessageBox.information(self, "Başarılı", f"Deney Raporu Kaydedildi:\n{filename}")

    @Slot(str, object, object, float)
    def handle_result(self, algo, path, metrics, duration):
        w1, w2, w3 = self.get_weights() # Ambil dari slider
        cost = 0.0
        if path:
            cost = (w1*metrics.get('delay',0)) + (w2*metrics.get('rel_cost',0)*100) + (w3*metrics.get('res_cost',0))
        self.log.append(f"{algo}: Maliyet={cost:.2f} ({duration:.2f}s)")
        
        if self.mode == "Single":
            if path:
                self.lbl_status.setText("EN İYİ YOL BULUNDU")
                self.lbl_status.setStyleSheet("background-color: #24283b; color: #9ece6a; font-weight: 900; font-size: 16px; border-radius: 10px; padding: 20px; border: 2px solid #9ece6a;")
                self.lbl_time.setText(f"{duration:.4f} s")
                
                rota_str = " → ".join(map(str, path))
                self.lbl_route.setText(rota_str) 
                self.lbl_route.setToolTip(rota_str)
                self.log.append(f"📍 Rota: {rota_str}")

                self.lbl_score.setText(f"{cost:.4f}")
                self.lbl_delay.setText(f"{metrics.get('delay',0):.2f} ms")
                self.lbl_rel.setText(f"{metrics.get('rel_cost',0):.4f}")
                self.lbl_res.setText(f"{metrics.get('res_cost',0):.2f}")
                self.draw_path(path, QColor("#e0af68"))
            else:
                self.lbl_status.setText("YOL BULUNAMADI")
                self.lbl_status.setStyleSheet("background-color: #24283b; color: #f7768e; font-weight: 900; font-size: 16px; border-radius: 10px; padding: 20px; border: 2px solid #f7768e;")
                self.lbl_route.setText("-")
        elif self.mode == "Compare":
            self.comp_data.append({'name': algo, 'path': path, 'metrics': metrics, 'time': duration, 'cost': cost})

    def finish_compare(self):
        self.set_ui_busy(False)
        valid = [x for x in self.comp_data if x['path']]
        if not valid: 
            QMessageBox.warning(self, "Başarısız", "Hiçbir algoritma yol bulamadı."); 
            self.lbl_status.setText("BAŞARISIZ")
            return
        
        sorted_results = sorted(valid, key=lambda x: x['cost'])
        winner = sorted_results[0]
        
        self.lbl_status.setText(f"KAZANAN: {winner['name']}")
        self.lbl_status.setStyleSheet("background-color: #24283b; color: #bb9af7; font-weight: 900; font-size: 14px; border-radius: 10px; padding: 10px; border: 2px solid #bb9af7;")
        
        self.table_res.setRowCount(len(sorted_results))
        for i, res in enumerate(sorted_results):
            short_name = res['name'].split("(")[-1].strip(")") if "(" in res['name'] else res['name']
            
            item_name = QTableWidgetItem(short_name)
            item_cost = QTableWidgetItem(f"{res['cost']:.2f}")
            item_time = QTableWidgetItem(f"{res['time']:.4f}s")
            
            if i == 0:
                for it in [item_name, item_cost, item_time]:
                    it.setForeground(QBrush(QColor("#9ece6a")))
                    it.setFont(QFont("Consolas", 9, QFont.Bold))
            
            self.table_res.setItem(i, 0, item_name)
            self.table_res.setItem(i, 1, item_cost)
            self.table_res.setItem(i, 2, item_time)

        self.draw_path(winner['path'], QColor("#bb9af7"))
        self.log.append(f"Karşılaştırma tamamlandı. Kazanan: {winner['name']}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NetworkVisualizer()
    window.show()
    sys.exit(app.exec())