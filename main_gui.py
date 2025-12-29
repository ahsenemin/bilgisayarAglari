# =====================================================
# GEREKLİ KÜTÜPHANELER
# =====================================================
import glob                      # Dosya listeleme işlemleri
import os                        # İşletim sistemi etkileşimi
import sys                       # Sistem argümanları ve çıkış
import random                    # Rastgelelik işlemleri
import math                      # Matematiksel fonksiyonlar
import time                      # Zaman ölçümü
import networkx as nx            # Ağ (Graph) yapısı ve algoritmaları
from collections import defaultdict
import io

# PySide6 (Grafik Arayüz) Bileşenleri
from PySide6.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, 
                               QGraphicsView, QVBoxLayout, QHBoxLayout, QWidget, QLabel, 
                               QPushButton, QDoubleSpinBox, QSpinBox, QFrame, QComboBox, 
                               QTextEdit, QMessageBox, QProgressBar, QGridLayout, QGroupBox, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                               QSlider, QTabWidget, QTextBrowser)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QObject
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QFont, QPainterPath
from PySide6.QtWidgets import QCheckBox


# =====================================================
# 1. HARİCİ MODÜL KONTROLLERİ
# =====================================================
# Proje içerisindeki algoritma modüllerinin varlığı kontrol edilir.

try: 
    import ag
    AG_AVAILABLE = True          # Ağ topolojisi (ag.py)
except ImportError: 
    AG_AVAILABLE = False
    print("UYARI: ag.py bulunamadı!")

try: 
    from genetik_proje import GenetikAlgoritma
    GA_AVAILABLE = True          # Genetik Algoritma
except ImportError: 
    GA_AVAILABLE = False
    print("UYARI: genetik_proje.py bulunamadı!")

try: 
    from karinca import ACORouting
    ACO_AVAILABLE = True         # Karınca Kolonisi
except ImportError: 
    ACO_AVAILABLE = False
    print("UYARI: karinca.py bulunamadı!")

try: 
    import q_learning as ql
    RL_AVAILABLE = True          # Q-Learning
except ImportError: 
    RL_AVAILABLE = False
    print("UYARI: q_learning.py bulunamadı!")

try: 
    import deney_duzenegi
    DENEY_AVAILABLE = True       # Toplu deney modülü
except ImportError: 
    DENEY_AVAILABLE = False
    print("UYARI: deney_duzenegi.py bulunamadı!")

# =====================================================
# 2. HESAPLAMA MOTORU (BACKEND)
# =====================================================

def calculate_path_metrics_detailed(graph, path):
    """
    Verilen yol (path) için Gecikme, Güvenilirlik ve Kaynak maliyetlerini hesaplar.
    """
    if not path or len(path) < 2: 
        return 0, 0, 0
    
    total_delay = 0.0
    rel_cost = 0.0
    res_cost = 0.0

    # Kenar (Edge) Maliyetleri
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if not graph.has_edge(u, v): 
            return 0, 0, 0
        
        edge = graph[u][v]
        total_delay += edge.get('delay', 0)
        
        # Güvenilirlik (Logaritmik toplam)
        r_link = edge.get('reliability', 0.99)
        rel_cost += -math.log(r_link if r_link > 0 else 1e-6)
        
        # Kaynak Maliyeti (Bant genişliği ile ters orantılı)
        bw = edge.get('bandwidth', 100)
        res_cost += (1000.0 / (bw if bw > 0 else 1))

    # Düğüm (Node) Maliyetleri
    for i, node in enumerate(path):
        n_data = graph.nodes[node]
        r_node = n_data.get('reliability', 0.99)
        rel_cost += -math.log(r_node if r_node > 0 else 1e-6)
        
        # Başlangıç ve bitiş hariç işlem gecikmesi
        if i != 0 and i != len(path) - 1:
            total_delay += n_data.get('processing_delay', 0)

    return total_delay, rel_cost, res_cost

class NetworkManager:
    """
    Ağ yapısını tutar ve görselleştirme koordinatlarını yönetir.
    """
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
        # Düğümlerin ekrandaki yerleşimini hesaplar (Spring Layout)
        if self.graph.number_of_nodes() > 0:
            nodes_sorted = sorted(list(self.graph.nodes()))
            temp_G = self.graph.subgraph(nodes_sorted)
            
            k_val = 4.0 / math.sqrt(self.graph.number_of_nodes()) 
            raw_pos = nx.spring_layout(temp_G, seed=seed, k=k_val, iterations=30)
            
            cx, cy, scale = 400, 400, 2500  
            self.pos_cache = {} 
            for n, p in raw_pos.items():
                self.pos_cache[n] = (cx + p[0]*scale, cy + p[1]*scale)

class RouteSolver:
    def _set_seed(self, seed):
        if seed is None:
            return
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except Exception:
            pass


    """
    GUI ile algoritmalar arasındaki köprü sınıfı.
    """
    def __init__(self, manager): 
        self.net = manager

    def solve(self, algo_type, src, dst, weights, demand_bw=4.0, seed=None):
        if self.net.graph.number_of_nodes() == 0: return None, {}, 0
        path, duration, metrics = None, 0, {}
        graph_copy = self.net.graph.copy()

        try:
            # 1. Genetik Algoritma
            if algo_type == "Genetik Algoritma (GA)" and GA_AVAILABLE:
                self._set_seed(seed)   # <-- BURAYA (GA çalışmadan hemen önce)
                ga = GenetikAlgoritma(graph_copy, src, dst, 50, 0.1, 50, weights, min_bw=demand_bw , seed=seed) 
                path, _, duration = ga.calistir()

            # 2. Karınca Kolonisi (ACO)
            elif algo_type == "Karınca Kolonisi (ACO)" and ACO_AVAILABLE:
                self._set_seed(seed)   # <-- BURAYA (GA çalışmadan hemen önce)
                start_t = time.time()
                aco = ACORouting(graph_copy, src, dst, demand_bw, weights, 20, 30 , seed=seed)
                path = aco.solve()[0]
                duration = time.time() - start_t
            # 3. Q-Learning (RL)
            elif algo_type == "Q-Öğrenme (RL)" and RL_AVAILABLE:
                self._set_seed(seed)   # <-- BURAYA (GA çalışmadan hemen önce)
                start_t = time.time()
                nbrs = {n: list(graph_copy.neighbors(n)) for n in graph_copy.nodes()}
                w_dict = {"w_delay": weights[0], "w_rel": weights[1], "w_bw": weights[2]}
                r_fn = ql.make_reward_fn(w_dict, demand_mbps=demand_bw)
                Q = ql.q_learning(graph_copy, nbrs, src, dst, r_fn, 800, 0.15, 0.97, 200, 0.9, 0.05, 600)
                path = ql.greedy_path(Q, nbrs, src, dst)
                if not path or path[-1] != dst: path = None
                duration = time.time() - start_t


            
            # Sonuç Metrikleri
            if path:
                d, r, c = calculate_path_metrics_detailed(self.net.graph, path)
                metrics = {'delay': d, 'rel_cost': r, 'res_cost': c}

        except Exception as e: 
            print(f"Hata: {e}")
            import traceback
            traceback.print_exc()

        return path, metrics, duration

# =====================================================
# 3. İŞ PARÇACIKLARI (THREADS)
# =====================================================

class EmittingStream(QObject):
    # stdout çıktılarını yakalayıp sinyal olarak yayar
    textWritten = Signal(str)
    def write(self, text):
        self.textWritten.emit(str(text))
    def flush(self):
        pass

class CalculationWorker(QThread):
    """
    Algoritmaları arayüzü dondurmadan arka planda çalıştırır.
    """
    result_ready = Signal(str, object, object, float)
    finished_all = Signal()
    
    def __init__(self, solver, mode, params):
        super().__init__()
        self.solver, self.mode, self.params = solver, mode, params
    
    def run(self):
        s, d, bw, w1, w2, w3, algo, seed = self.params
        weights = [w1, w2, w3]
        
        if self.mode == "Single":
            p, m, t = self.solver.solve(algo, s, d, weights, demand_bw=bw, seed=seed)
            self.result_ready.emit(algo, p, m, t)
        
        elif self.mode == "Compare":
            for name in ["Genetik Algoritma (GA)", "Karınca Kolonisi (ACO)", "Q-Öğrenme (RL)"]:
                p, m, t = self.solver.solve(name, s, d, weights, demand_bw=bw, seed=seed)
                self.result_ready.emit(name, p, m, t)
        
        self.finished_all.emit()

class MassExperimentWorker(QThread):
    """
    deney_duzenegi.py dosyasını çalıştırarak toplu deney yapar.
    """
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self):
        super().__init__()

    def run(self):
        if not DENEY_AVAILABLE:
            self.log_signal.emit("HATA: deney_duzenegi.py bulunamadı.")
            self.finished_signal.emit()
            return

        # Konsol yönlendirme
        original_stdout = sys.stdout
        stream = EmittingStream()
        stream.textWritten.connect(self.on_text_written)
        sys.stdout = stream

        original_argv = sys.argv
        sys.argv = ["deney_duzenegi.py", "--repeats", "5", "--demands", "20"] 

        try:
            self.log_signal.emit("--- Toplu Deney Başlatılıyor (20 Demand, 5 Tekrar) ---")
            deney_duzenegi.main()
            self.log_signal.emit("--- Toplu Deney Tamamlandı ---")
        except Exception as e:
            self.log_signal.emit(f"HATA: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = original_stdout
            sys.argv = original_argv
            self.finished_signal.emit()

    def on_text_written(self, text):
        text = text.rstrip()
        if text:
            self.log_signal.emit(text)

# =====================================================
# 4. GRAFİKSEL KULLANICI ARAYÜZÜ (GUI)
# =====================================================

class NetworkVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSM307 - QoS Odaklı Akıllı Ağ Rotalama")
        self.resize(1450, 900)
        self.net = NetworkManager()
        self.solver = RouteSolver(self.net)
        
        # Arayüz Stili (Modern Dark Theme)
        self.setStyleSheet("""
            QMainWindow { background-color: #0f111a; font-family: 'Segoe UI', Roboto, sans-serif; }
            QFrame#Panel { background-color: #1a1c29; border-radius: 15px; border: 1px solid #2f334d; }
            QLabel { color: #a9b1d6; font-size: 13px; }
            QLabel#Header { color: #7aa2f7; font-size: 15px; font-weight: 900; letter-spacing: 1px; padding-bottom: 5px; border-bottom: 2px solid #7aa2f7; }
            QLabel#InputLabel { color: #bb9af7; font-weight: bold; font-size: 12px; margin-bottom: 2px; }
            QGroupBox { border: 1px solid #414868; border-radius: 10px; margin-top: 22px; font-weight: bold; color: #bb9af7; }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; background-color: #1a1c29; }
            QSpinBox, QComboBox, QDoubleSpinBox { background-color: #24283b; color: white; padding: 8px; border: 1px solid #414868; border-radius: 6px; font-weight: bold; }
            QSlider::groove:horizontal { border: 1px solid #414868; height: 8px; background: #24283b; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #7aa2f7; border: 1px solid #7aa2f7; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; }
            QPushButton { background-color: #3b4261; color: white; border-radius: 8px; padding: 10px; font-weight: bold; border: 1px solid #414868; }
            QPushButton:hover { background-color: #414868; border: 1px solid #7aa2f7; }
            QPushButton#BtnRun { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7aa2f7, stop:1 #2ac3de); color: #0f111a; border: none; }
            QPushButton#BtnCmp { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #bb9af7, stop:1 #f7768e); color: #0f111a; border: none; }
            QPushButton#BtnMass { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e0af68, stop:1 #ff9e64); color: #0f111a; border: none; }
            QTextEdit, QTextBrowser { background-color: #0f111a; color: #c0caf5; font-family: Consolas; border: 1px solid #2f334d; border-radius: 8px; }
            QProgressBar { border: none; background: #24283b; height: 6px; border-radius: 3px; }
            QProgressBar::chunk { background: #7aa2f7; border-radius: 3px; }
            QTableWidget { background-color: #0f111a; color: #c0caf5; border: 1px solid #414868; gridline-color: #2f334d; font-family: Consolas; font-size: 12px; }
            QHeaderView::section { background-color: #1a1c29; color: #7aa2f7; padding: 4px; border: 1px solid #2f334d; font-weight: bold; }
            QTableWidget::item { padding: 5px; }
            QTableWidget::item:selected { background-color: #3b4261; }
            QTabWidget::pane { border: 1px solid #414868; border-radius: 8px; background-color: #1a1c29; }
            QTabBar::tab { background: #24283b; color: #a9b1d6; padding: 10px 15px; border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: bold; margin-right: 2px; }
            QTabBar::tab:selected { background: #3b4261; color: #7aa2f7; border-bottom: 2px solid #7aa2f7; }
            QTabBar::tab:hover { background: #2f334d; color: white; }
            QCheckBox { color: #a9b1d6; font-weight: bold; spacing: 8px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QCheckBox::indicator:unchecked { border: 1px solid #414868; background: #24283b; border-radius: 3px; }
            QCheckBox::indicator:checked { border: 1px solid #7aa2f7; background: #7aa2f7; border-radius: 3px; }
            QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
                color: #565f89;
                border: 1px solid #2f334d;
                background-color: #1f2335;
            }
        """)

        central = QWidget(); self.setCentralWidget(central)
        main_lo = QHBoxLayout(central); main_lo.setSpacing(20); main_lo.setContentsMargins(20,20,20,20)

        # -----------------------------
        # 1. SOL PANEL (AYARLAR)
        # -----------------------------
        left = QFrame(); left.setObjectName("Panel"); left.setFixedWidth(340)
        left_lo = QVBoxLayout(left); left_lo.setSpacing(15); left_lo.setContentsMargins(20,20,20,20)

        title = QLabel("KONTROL MERKEZİ"); title.setObjectName("Header"); title.setAlignment(Qt.AlignCenter)
        left_lo.addWidget(title)

        # Topoloji Ayarları
        grp_route = QGroupBox("TOPOLOJİ & TALEP AYARLARI")
        route_main_lo = QVBoxLayout(grp_route); route_main_lo.setSpacing(10)
        
        row1_lo = QHBoxLayout(); row1_lo.setSpacing(10)
        max_id = max(list(self.net.graph.nodes)) if self.net.graph.nodes else 0
        
        # Kaynak ve Hedef
        src_cont = QWidget()
        src_vlo = QVBoxLayout(src_cont); src_vlo.setContentsMargins(0,0,0,0); src_vlo.setSpacing(5)
        lbl_s = QLabel("Kaynak (Source)"); lbl_s.setObjectName("InputLabel")
        self.spin_s = QSpinBox(); self.spin_s.setRange(0, max_id)
        src_vlo.addWidget(lbl_s); src_vlo.addWidget(self.spin_s)
        
        dst_cont = QWidget()
        dst_vlo = QVBoxLayout(dst_cont); dst_vlo.setContentsMargins(0,0,0,0); dst_vlo.setSpacing(5)
        lbl_d = QLabel("Hedef (Dest)"); lbl_d.setObjectName("InputLabel")
        self.spin_d = QSpinBox(); self.spin_d.setRange(0, max_id); self.spin_d.setValue(max_id)
        dst_vlo.addWidget(lbl_d); dst_vlo.addWidget(self.spin_d)

        row1_lo.addWidget(src_cont); row1_lo.addWidget(dst_cont)
        route_main_lo.addLayout(row1_lo)

        # Bant Genişliği
        row2_lo = QHBoxLayout(); row2_lo.setSpacing(10)
        bw_cont = QWidget()
        bw_vlo = QVBoxLayout(bw_cont); bw_vlo.setContentsMargins(0,0,0,0); bw_vlo.setSpacing(5)
        lbl_bw = QLabel("Bant Genişliği (Bandwidth)"); lbl_bw.setObjectName("InputLabel")
        self.spin_bw = QDoubleSpinBox()
        self.spin_bw.setRange(0.1, 1000.0); self.spin_bw.setValue(10.0); self.spin_bw.setSuffix(" Mbps")
        bw_vlo.addWidget(lbl_bw); bw_vlo.addWidget(self.spin_bw)
        row2_lo.addWidget(bw_cont)
        route_main_lo.addLayout(row2_lo)
        left_lo.addWidget(grp_route)
        
        # Strateji Ayarları
        grp_algo = QGroupBox("OPTİMİZASYON STRATEJİSİ")
        g_al_lo = QVBoxLayout(grp_algo); g_al_lo.setSpacing(15)

        # Seed girişi
        seed_cont = QFrame()
        seed_cont.setStyleSheet("""
            QFrame {
                background-color: #16161e;
                border: 1px solid #2f334d;
                border-radius: 10px;
                padding: 8px;
            }
        """)

        seed_lo = QVBoxLayout(seed_cont)
        seed_lo.setContentsMargins(10, 10, 10, 10)
        seed_lo.setSpacing(8)

        # Üst satır: checkbox + küçük açıklama
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        self.chk_seed = QCheckBox("Seed kullan")
        self.chk_seed.setChecked(False)



        top_row.addWidget(self.chk_seed)
        top_row.addStretch()
        seed_lo.addLayout(top_row)

        # Alt satır: label + spinbox aynı hizada
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        lbl_seed = QLabel("Seed")
        lbl_seed.setObjectName("InputLabel")

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000)
        self.spin_seed.setValue(42)
        self.spin_seed.setEnabled(False)
        self.spin_seed.setFixedWidth(140)  # görünüm daha dengeli
        self.spin_seed.setAlignment(Qt.AlignCenter)

        bottom_row.addWidget(lbl_seed)
        bottom_row.addStretch()
        bottom_row.addWidget(self.spin_seed)

        seed_lo.addLayout(bottom_row)

        # Connect
        self.chk_seed.toggled.connect(self.spin_seed.setEnabled)

        # gruba ekle
        g_al_lo.addWidget(seed_cont)


    
        algo_cont = QWidget()
        algo_vlo = QVBoxLayout(algo_cont); algo_vlo.setContentsMargins(0,0,0,0); algo_vlo.setSpacing(5)
        algo_vlo.addWidget(QLabel("Algoritma Seçimi:", objectName="InputLabel"))
        self.combo = QComboBox(); 
        self.combo.addItems(["Genetik Algoritma (GA)", "Karınca Kolonisi (ACO)", "Q-Öğrenme (RL)"])
        
        

        algo_vlo.addWidget(self.combo)
        g_al_lo.addWidget(algo_cont)
        
        # Ağırlıklar
        g_al_lo.addWidget(QLabel("Ağırlıklar:", objectName="InputLabel"))
        def create_slider_row(text, default_val):
            container = QWidget()
            v_layout = QVBoxLayout(container); v_layout.setContentsMargins(0,0,0,0); v_layout.setSpacing(2)
            lbl_title = QLabel(text); lbl_title.setStyleSheet("color: #a9b1d6; font-size: 11px;")
            v_layout.addWidget(lbl_title)
            h_layout = QHBoxLayout(); h_layout.setContentsMargins(0,0,0,0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100); slider.setValue(int(default_val * 100))
            lbl_val = QLabel(f"{default_val:.2f}")
            lbl_val.setStyleSheet("color: #7aa2f7; font-weight: bold; font-family: Consolas;")
            lbl_val.setFixedWidth(40); lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            slider.valueChanged.connect(lambda v: lbl_val.setText(f"{v/100:.2f}"))
            h_layout.addWidget(slider); h_layout.addWidget(lbl_val)
            v_layout.addLayout(h_layout)
            return container, slider
        w1_wid, self.slider_w1 = create_slider_row("Gecikme (Delay)", 0.4)
        w2_wid, self.slider_w2 = create_slider_row("Güvenilirlik (Reliability)", 0.3)
        w3_wid, self.slider_w3 = create_slider_row("Kaynak (Resource)", 0.3)
        g_al_lo.addWidget(w1_wid); g_al_lo.addWidget(w2_wid); g_al_lo.addWidget(w3_wid)
        left_lo.addWidget(grp_algo)

        # Butonlar
        left_lo.addSpacing(10)
        self.btn_run = QPushButton("▶  HESAPLA (BAŞLAT)"); self.btn_run.setObjectName("BtnRun") 
        self.btn_run.clicked.connect(self.start_single)
        left_lo.addWidget(self.btn_run)

        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        self.btn_cmp = QPushButton("▶  KARŞILAŞTIR"); self.btn_cmp.setObjectName("BtnCmp")
        self.btn_cmp.clicked.connect(self.start_compare)
        btn_row.addWidget(self.btn_cmp)
        self.btn_mass = QPushButton("▶  TOPLU DENEY"); self.btn_mass.setObjectName("BtnMass")
        self.btn_mass.setToolTip("deney_duzenegi.py dosyasını çalıştırır")
        self.btn_mass.clicked.connect(self.start_mass_experiment)
        btn_row.addWidget(self.btn_mass)
        left_lo.addLayout(btn_row)

        self.pbar = QProgressBar(); self.pbar.setVisible(False)
        left_lo.addWidget(self.pbar)
        left_lo.addStretch()
        main_lo.addWidget(left)

        # -----------------------------
        # 2. ORTA PANEL (GRAFİK)
        # -----------------------------
        center = QFrame(); center.setObjectName("Panel")
        center_lo = QVBoxLayout(center); center_lo.setContentsMargins(15,15,15,15)
        tool_lo = QHBoxLayout()
        lbl_vis = QLabel("AĞ HARİTASI (250 Düğüm)"); lbl_vis.setObjectName("Header"); 
        tool_lo.addWidget(lbl_vis); tool_lo.addStretch()
        btn_redraw = QPushButton("🔄 Karıştır"); btn_redraw.setFixedWidth(80); btn_redraw.clicked.connect(self.redraw_network)
        tool_lo.addWidget(btn_redraw)
        center_lo.addLayout(tool_lo)

        self.scene = QGraphicsScene(); self.scene.setSceneRect(-2500, -2500, 6000, 6000) 
        self.view = QGraphicsView(self.scene); self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setBackgroundBrush(QBrush(QColor("#1a1c29"))); self.view.setStyleSheet("border: none; border-radius: 8px;")
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

        # -----------------------------
        # 3. SAĞ PANEL (SONUÇLAR)
        # -----------------------------
        right = QFrame(); right.setObjectName("Panel"); right.setFixedWidth(380) 
        right_lo = QVBoxLayout(right); right_lo.setSpacing(10); right_lo.setContentsMargins(10,10,10,10)

        self.lbl_status = QLabel("GİRİŞ BEKLENİYOR"); self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("background-color: #24283b; color: #565f89; font-weight: 900; font-size: 16px; border-radius: 10px; padding: 15px; border: 2px dashed #414868;")
        right_lo.addWidget(self.lbl_status)

        self.tabs = QTabWidget()
        right_lo.addWidget(self.tabs)

        # Sekme 1: Analiz
        self.tab_analysis = QWidget()
        self.tabs.addTab(self.tab_analysis, "📊 Analiz")
        analysis_lo = QVBoxLayout(self.tab_analysis); analysis_lo.setContentsMargins(10,10,10,10); analysis_lo.setSpacing(10)
        
        self.grp_single_details = QWidget()
        single_lo = QVBoxLayout(self.grp_single_details); single_lo.setContentsMargins(0,0,0,0); single_lo.setSpacing(10)
        
        self.grp_overview = QGroupBox("Genel Bakış")
        ov_lo = QVBoxLayout(self.grp_overview); ov_lo.setSpacing(8)
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
        qos_lo = QVBoxLayout(self.grp_qos); qos_lo.setSpacing(8)
        w_d, self.lbl_delay = mk_row("⚡", "Toplam Gecikme")
        w_r, self.lbl_rel = mk_row("🛡️", "Güvenilirlik Maliyeti")
        w_b, self.lbl_res = mk_row("💰", "Kaynak Maliyeti")
        qos_lo.addWidget(w_d); qos_lo.addWidget(w_r); qos_lo.addWidget(w_b)
        single_lo.addWidget(self.grp_qos)
        analysis_lo.addWidget(self.grp_single_details)

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
        analysis_lo.addWidget(self.grp_compare_table)
        analysis_lo.addStretch()

        # Sekme 2: Rapor
        self.tab_report = QWidget()
        self.tabs.addTab(self.tab_report, "📄 Rapor")
        rep_lo = QVBoxLayout(self.tab_report); rep_lo.setContentsMargins(0,10,0,0)
        self.report_viewer = QTextBrowser()
        self.report_viewer.setHtml("<div style='color:#565f89; text-align:center; margin-top:50px;'>Henüz rapor oluşturulmadı.<br>Toplu deney başlatın.</div>")
        rep_lo.addWidget(self.report_viewer)
        
        # Sekme 3: Log
        self.tab_log = QWidget()
        self.tabs.addTab(self.tab_log, "🖥️ Log")
        log_lo = QVBoxLayout(self.tab_log); log_lo.setContentsMargins(0,10,0,0)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        log_lo.addWidget(self.log)

        main_lo.addWidget(right)

        self.draw_graph_background(); self.path_items = []; self.comp_data = []
        QTimer.singleShot(100, lambda: self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio))

    # -----------------------------
    # YARDIMCI METOTLAR
    # -----------------------------
    def resizeEvent(self, event):
        if hasattr(self, 'view') and hasattr(self, 'scene'): self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

    def showEvent(self, event):
        if hasattr(self, 'view') and hasattr(self, 'scene'): self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        super().showEvent(event)

    def redraw_network(self):
        self.log.append("Topoloji yeniden yerleştiriliyor..."); self.net.calculate_layout(seed=random.randint(1, 10000))
        self.draw_graph_background(); self.path_items = []; self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def draw_graph_background(self):
        self.scene.clear(); self.path_items = []
        if self.net.graph.number_of_nodes() == 0: return
        self.view.setUpdatesEnabled(False)
        edge_path = QPainterPath()
        for u, v in self.net.graph.edges():
            if u in self.net.pos_cache and v in self.net.pos_cache:
                p1 = self.net.pos_cache[u]; p2 = self.net.pos_cache[v]
                edge_path.moveTo(p1[0], p1[1]); edge_path.lineTo(p2[0], p2[1])
        pen = QPen(QColor(86, 95, 137, 40)); pen.setWidth(1); self.scene.addPath(edge_path, pen)
        brush = QBrush(QColor("#414868")); pen_n = QPen(Qt.NoPen)
        for n in self.net.graph.nodes():
            if n in self.net.pos_cache:
                x, y = self.net.pos_cache[n]; self.scene.addEllipse(x-6, y-6, 12, 12, pen_n, brush)
        self.view.setUpdatesEnabled(True)

    def draw_path(self, path, color=QColor("#e0af68")):
        for item in self.path_items: 
            try: 
                if item.scene() == self.scene: self.scene.removeItem(item)
            except: pass
        self.path_items.clear()
        if not path: return
        pen_glow = QPen(color); pen_glow.setWidth(100); pen_glow.setColor(QColor(color.red(), color.green(), color.blue(), 100)) 
        pen_core = QPen(color); pen_core.setWidth(50); 
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            if u in self.net.pos_cache and v in self.net.pos_cache:
                p1 = self.net.pos_cache[u]; p2 = self.net.pos_cache[v]
                self.path_items.append(self.scene.addLine(p1[0], p1[1], p2[0], p2[1], pen_glow))
                self.path_items.append(self.scene.addLine(p1[0], p1[1], p2[0], p2[1], pen_core))
        if path:
            s, e = path[0], path[-1]; node_radius = 100 
            if s in self.net.pos_cache:
                sp = self.net.pos_cache[s]; self.path_items.append(self.scene.addEllipse(sp[0]-node_radius, sp[1]-node_radius, node_radius*2, node_radius*2, QPen(Qt.NoPen), QBrush(QColor("#9ece6a"))))
            if e in self.net.pos_cache:
                ep = self.net.pos_cache[e]; self.path_items.append(self.scene.addEllipse(ep[0]-node_radius, ep[1]-node_radius, node_radius*2, node_radius*2, QPen(Qt.NoPen), QBrush(QColor("#f7768e"))))

    def set_ui_busy(self, busy):
        for b in [self.btn_run, self.btn_cmp, self.btn_mass]: b.setEnabled(not busy)
        self.pbar.setVisible(busy); 
        if not busy: self.pbar.setValue(0)

    def get_weights(self): 
        return self.slider_w1.value()/100.0, self.slider_w2.value()/100.0, self.slider_w3.value()/100.0

    # -----------------------------
    # İŞLEM BAŞLATMA
    # -----------------------------
    def start_single(self):
        if not AG_AVAILABLE: QMessageBox.critical(self, "Hata", "ag.py eksik!"); return
        self.mode = "Single"; self.set_ui_busy(True); self.tabs.setCurrentIndex(0) 
        self.grp_single_details.setVisible(True); self.grp_compare_table.setVisible(False)
        self.lbl_status.setText("HESAPLANIYOR...")
        self.log.append("--- Simülasyon Başlatılıyor ---")
        algo = self.combo.currentText(); w1, w2, w3 = self.get_weights(); bw_val = self.spin_bw.value() 
        seed_val = self.spin_seed.value() if self.chk_seed.isChecked() else None
        params = (self.spin_s.value(), self.spin_d.value(), bw_val, w1, w2, w3, algo, seed_val)


        self.worker = CalculationWorker(self.solver, "Single", params)
        self.worker.result_ready.connect(self.handle_result); self.worker.finished_all.connect(lambda: self.set_ui_busy(False))
        self.worker.start()

    def start_compare(self):
        if not AG_AVAILABLE: QMessageBox.critical(self, "Hata", "ag.py eksik!"); return
        self.mode = "Compare"; self.set_ui_busy(True); self.comp_data = []; self.tabs.setCurrentIndex(0) 
        self.grp_single_details.setVisible(False); self.grp_compare_table.setVisible(True)
        self.table_res.setRowCount(0); self.lbl_status.setText("KIYASLANIYOR...")
        self.log.append("--- Karşılaştırma Başlatılıyor ---")
        w1, w2, w3 = self.get_weights(); bw_val = self.spin_bw.value() 
        seed_val = self.spin_seed.value() if self.chk_seed.isChecked() else None
        params = (self.spin_s.value(), self.spin_d.value(), bw_val, w1, w2, w3, "ALL", seed_val)
        self.worker = CalculationWorker(self.solver, "Compare", params)
        self.worker.result_ready.connect(self.handle_result); self.worker.finished_all.connect(self.finish_compare)
        self.worker.start()

    def start_mass_experiment(self):
        if not DENEY_AVAILABLE: QMessageBox.critical(self, "Hata", "deney_duzenegi.py bulunamadı."); return
        self.set_ui_busy(True); self.log.clear(); self.tabs.setCurrentIndex(2) 
        self.lbl_status.setText("DENEY ÇALIŞIYOR...")
        self.lbl_status.setStyleSheet("background-color: #24283b; color: #e0af68; font-weight: 900; font-size: 16px; border-radius: 10px; padding: 15px; border: 2px solid #e0af68;")
        self.mass_worker = MassExperimentWorker()
        self.mass_worker.log_signal.connect(self.append_log_no_newline)
        self.mass_worker.finished_signal.connect(self.finish_mass_experiment)
        self.mass_worker.start()

    def append_log_no_newline(self, text):
        text = text.rstrip()
        if text:
            self.log.append(text)
            sb = self.log.verticalScrollBar(); sb.setValue(sb.maximum())

    # -----------------------------
    # RAPORLAMA VE SONUÇ İŞLEME
    # -----------------------------
    def finish_mass_experiment(self):
        self.set_ui_busy(False)
        self.lbl_status.setText("DENEY TAMAMLANDI")
        self.lbl_status.setStyleSheet("background-color: #24283b; color: #9ece6a; font-weight: 900; font-size: 16px; border-radius: 10px; padding: 15px; border: 2px solid #9ece6a;")
        
        try:
            list_of_files = glob.glob('deney_detay_*.txt') 
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                with open(latest_file, "r", encoding="utf-8") as f:
                    raw_content = f.read()
                
                pretty_html = self.parse_and_format_report(raw_content, latest_file)
                self.report_viewer.setHtml(pretty_html)
                self.tabs.setCurrentIndex(1) 
            else:
                QMessageBox.warning(self, "Uyarı", "Rapor dosyası bulunamadı.")
        except Exception as e:
            self.log.append(f"Rapor hatası: {e}")
            import traceback
            traceback.print_exc()

        QMessageBox.information(self, "Bilgi", "Toplu deney tamamlandı. Rapor sekmesine bakınız.")

    def parse_and_format_report(self, text, filename):
        """
        Ham metin raporunu okur ve özet bir HTML formatına dönüştürür.
        Detaylı loglar arayüzde gösterilmez, sadece istatistiksel özet sunulur.
        """
        import re

        # --- CSS STİL TANIMLAMALARI ---
        style = """
        <style>
            body { font-family: 'Segoe UI', Consolas, sans-serif; color: #c0caf5; background-color: #1a1c29; }
            
            /* Rapor Başlığı */
            .report-header { border-bottom: 2px solid #7aa2f7; padding-bottom: 10px; margin-bottom: 20px; }
            h2 { color: #7aa2f7; margin: 0; font-size: 20px; }
            .meta { color: #565f89; font-size: 12px; font-style: italic; margin-top: 5px; }

            /* Deney Kartı */
            .exp-card { 
                background-color: #24283b; border: 1px solid #414868; border-radius: 8px; 
                padding: 10px 15px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .exp-title { font-size: 13px; font-weight: bold; color: #e0af68; margin-bottom: 8px; border-bottom: 1px dashed #414868; padding-bottom: 4px; }

            /* İstatistik Tablosu */
            table.sum-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 5px; }
            table.sum-table th { 
                text-align: left; background-color: #1f2335; color: #7aa2f7; 
                padding: 6px 8px; border: 1px solid #414868; font-weight: 600;
            }
            table.sum-table td { 
                padding: 6px 8px; border: 1px solid #414868; color: #c0caf5; vertical-align: middle; 
            }
            
            .algo-name { font-weight: bold; color: #bb9af7; }
            .val-cost { color: #7dcfff; font-family: Consolas; font-weight: bold; }
            .val-stats { color: #a9b1d6; font-family: Consolas; font-size: 11px; }
            .val-success { color: #9ece6a; font-weight: bold; }
            .val-fail { color: #f7768e; font-weight: bold; }
            
            /* Genel Özet Kutusu */
            .summary-box { background-color: #16161e; border: 1px dashed #7aa2f7; padding: 15px; margin-top: 30px; border-radius: 8px; }
        </style>
        """

        html_out = [style, f"<div class='report-header'><h2>📄 ÖZET İSTATİSTİK RAPORU</h2><div class='meta'>Tam detaylı veriler dosyada saklıdır: {filename}</div></div>"]

        parts = text.split("=== Deney")
        
        # Üst Bilgi (Tarih, Ağırlıklar vb.)
        header_info = parts[0].strip().replace("\n", "<br>")
        html_out.append(f"<div style='margin-bottom:20px; color:#a9b1d6; font-size:12px; border:1px solid #2f334d; padding:10px; border-radius:6px;'>{header_info}</div>")

        # Her deney bloğu için döngü
        for part in parts[1:]:
            lines = part.strip().split("\n")
            title_line = lines[0].strip() # Örn: 01: S=0, D=5...
            
            # Genel özet bölümü kontrolü
            if "Genel Başarı Özeti" in part:
                html_out.append("<div class='summary-box'><h3>GENEL DEĞERLENDİRME</h3>")
                summary_content = part.split("Genel Başarı Özeti ===")[1].strip().replace("\n", "<br>")
                html_out.append(f"<div style='font-size:13px; line-height:1.6;'>{summary_content}</div></div>")
                continue

            # Deney kartını başlat
            html_out.append(f"<div class='exp-card'><div class='exp-title'>📌 Deney {title_line}</div>")

            # Tabloyu başlat
            html_out.append("""
            <table class='sum-table'>
                <tr>
                    <th width="15%">Algoritma</th>
                    <th width="10%">Başarı</th>
                    <th width="15%">Ort. Süre</th>
                    <th width="25%">Maliyet (Ort ± Std)</th>
                    <th width="20%">En İyi - En Kötü</th>
                    <th width="15%">Durum</th>
                </tr>
            """)

            # Algoritma satırlarını işleyen yardımcı fonksiyon
            def render_algo_row(block_lines):
                if not block_lines: return ""
                
                header_line = block_lines[0] # [GA] ...
                if not header_line.startswith("["): return ""
                
                name = header_line.split("]")[0].replace("[", "").upper()
                
                # Başarı Oranını Ayrıştır
                success_rate = "N/A"
                if "Başarı:" in header_line:
                    success_rate = header_line.split("Başarı:")[1].split("|")[0].strip()

                # İstatistikleri Ayrıştır
                avg_time = "-"
                cost_avg_std = "-"
                cost_best_worst = "-"
                status_note = "<span style='color:#9ece6a'>Uygun</span>"

                for l in block_lines:
                    # Süre
                    if "Süre (sn)" in l and "Ortalama:" in l:
                        t_val = l.split("Ortalama:")[1].split(",")[0].strip()
                        avg_time = f"{float(t_val):.4f}s"
                    
                    # Maliyet İstatistikleri
                    if "Maliyet ->" in l and "Ortalama:" in l:
                        parts_c = l.split(",")
                        avg_c = parts_c[0].split(":")[1].strip()
                        std_c = parts_c[1].split(":")[1].strip()
                        best_c = parts_c[2].split(":")[1].strip()
                        worst_c = parts_c[3].split(":")[1].strip()
                        
                        cost_avg_std = f"{avg_c} ± {std_c}"
                        cost_best_worst = f"{best_c} - {worst_c}"

                    # Başarısızlık Kontrolü
                    if "Başarısız denemeler" in l:
                        status_note = "<span class='val-fail'>Hatalı</span>"

                # Başarı durumuna göre renklendirme
                s_class = "val-success" if "0/" not in success_rate else "val-fail"

                return f"""
                <tr>
                    <td class='algo-name'>{name}</td>
                    <td class='{s_class}'>{success_rate}</td>
                    <td>{avg_time}</td>
                    <td class='val-cost'>{cost_avg_std}</td>
                    <td class='val-stats'>{cost_best_worst}</td>
                    <td>{status_note}</td>
                </tr>
                """

            # Satır satır okuyarak algoritmaları ayıkla
            buffer = []
            for line in lines[1:]:
                sline = line.strip()
                if sline.startswith("["):
                    if buffer: html_out.append(render_algo_row(buffer))
                    buffer = [sline]
                else:
                    buffer.append(sline)
            
            # Son tamponu işle
            if buffer: html_out.append(render_algo_row(buffer))

            html_out.append("</table></div>") # Tablo ve kartı kapat

        return "\n".join(html_out)

    @Slot(str, object, object, float)
    def handle_result(self, algo, path, metrics, duration):
        w1, w2, w3 = self.get_weights(); cost = 0.0
        if path: cost = (w1*metrics.get('delay',0)) + (w2*metrics.get('rel_cost',0)*100) + (w3*metrics.get('res_cost',0))
        self.log.append(f"{algo}: Maliyet={cost:.2f} ({duration:.2f}s)")
        
        if self.mode == "Single":
            if path:
                self.lbl_status.setText("EN İYİ YOL BULUNDU")
                self.lbl_status.setStyleSheet("background-color: #24283b; color: #9ece6a; font-weight: 900; font-size: 16px; border-radius: 10px; padding: 15px; border: 2px solid #9ece6a;")
                self.lbl_time.setText(f"{duration:.4f} s")
                rota_str = " → ".join(map(str, path))
                self.lbl_route.setText(rota_str); self.lbl_route.setToolTip(rota_str)
                self.log.append(f"Rota: {rota_str}")
                self.lbl_score.setText(f"{cost:.4f}")
                self.lbl_delay.setText(f"{metrics.get('delay',0):.2f} ms")
                self.lbl_rel.setText(f"{metrics.get('rel_cost',0):.4f}")
                self.lbl_res.setText(f"{metrics.get('res_cost',0):.2f}")
                self.draw_path(path, QColor("#e0af68"))
            else:
                self.lbl_status.setText("YOL BULUNAMADI")
                self.lbl_status.setStyleSheet("background-color: #24283b; color: #f7768e; font-weight: 900; font-size: 16px; border-radius: 10px; padding: 15px; border: 2px solid #f7768e;")
                self.lbl_route.setText("-")
        elif self.mode == "Compare":
            self.comp_data.append({'name': algo, 'path': path, 'metrics': metrics, 'time': duration, 'cost': cost})

    def finish_compare(self):
        self.set_ui_busy(False)
        valid = [x for x in self.comp_data if x['path']]
        if not valid: 
            QMessageBox.warning(self, "Başarısız", "Hiçbir algoritma yol bulamadı."); self.lbl_status.setText("BAŞARISIZ"); return
        
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
                    it.setForeground(QBrush(QColor("#9ece6a"))); it.setFont(QFont("Consolas", 9, QFont.Bold))
            
            self.table_res.setItem(i, 0, item_name); self.table_res.setItem(i, 1, item_cost); self.table_res.setItem(i, 2, item_time)
        
        self.draw_path(winner['path'], QColor("#bb9af7"))
        self.log.append(f"Karşılaştırma tamamlandı. Kazanan: {winner['name']}")

# =====================================================
# PROGRAM BAŞLANGICI
# =====================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NetworkVisualizer()
    window.show()
    sys.exit(app.exec())

"""

elif 
    algo_type == "Q-Öğrenme (RL)" and RL_AVAILABLE:
    start_t = time.time()
    nbrs = {n: list(graph_copy.neighbors(n)) for n in graph_copy.nodes()}
    w_dict = {"w_delay": weights[0], "w_rel": weights[1], "w_bw": weights[2]}
    r_fn = ql.make_reward_fn(w_dict, demand_mbps=demand_bw)
    Q = ql.q_learning(graph_copy, nbrs, src, dst, r_fn, 800, 0.15, 0.97, 200, 0.9, 0.05, 600)
    path = ql.greedy_path(Q, nbrs, src, dst)
    if not path or path[-1] != dst: path = None
    duration = time.time() - start_t

"""