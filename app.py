import os
# Mencegah error libGL di Streamlit Cloud
os.environ["QT_QPA_PLATFORM"] = "offscreen" 

import streamlit as st
import streamlit.components.v1 as components
import cv2
import time
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import json
import folium
import requests
from urllib.parse import urljoin
from ultralytics import YOLO
from sklearn.ensemble import RandomForestRegressor

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="OptiFlow AI Engine", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# ADVANCED CSS INJECTION 
# ==========================================
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #151924 0%, #090b10 100%);
        color: #e6edf3;
    }

    .hero-title {
        background: linear-gradient(135deg, #a8c0ff 0%, #3f2b96 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle { color: #8b949e; font-weight: 400; font-size: 1.1rem; margin-bottom: 2rem; }

    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .metric-title { font-size: 0.9rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #ffffff; }
    .metric-unit { font-size: 1rem; font-weight: 400; color: #8b949e; }

    .alert-box { padding: 16px 20px; border-radius: 12px; margin-bottom: 15px; font-size: 0.95rem; line-height: 1.5; }
    .alert-ai { background: rgba(66, 133, 244, 0.08); border-left: 4px solid #4285f4; }
    .alert-ai-warn { background: rgba(217, 101, 112, 0.08); border-left: 4px solid #d96570; }
    .alert-danger { background: rgba(255, 75, 75, 0.1); border: 1px solid rgba(255, 75, 75, 0.3); }
    .alert-safe { background: rgba(46, 160, 67, 0.05); border: 1px solid rgba(46, 160, 67, 0.2); }

    [data-testid="stSidebar"] { background-color: #0d1117 !important; border-right: 1px solid #30363d; }
    h3 { font-weight: 600 !important; color: #e6edf3 !important; font-size: 1.3rem !important; margin-bottom: 1rem !important; }
    
    .pulse-dot {
        height: 10px; width: 10px; background-color: #ff4b4b; border-radius: 50%; display: inline-block; margin-right: 8px;
        box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(255, 75, 75, 0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); } }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# UI Header
st.markdown('<h1 class="hero-title">OptiFlow AI Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Advanced Spatial Telemetry & Predictive Traffic Control</p>', unsafe_allow_html=True)
st.markdown("<hr style='border-color: #30363d; margin-top: 0; margin-bottom: 2rem;'>", unsafe_allow_html=True)

os.makedirs("traffic_data", exist_ok=True)

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

model = load_yolo_model()

BOBOT_SMP = {2: 1.0, 3: 0.5, 5: 2.5, 7: 2.5}

# --- DATABASE CCTV JOGJAKARTA (M3U8 HLS DIRECT STREAM) ---
DATABASE_CCTV = {
    "SIMPANG KM NOL (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_kmnol.stream/playlist.m3u8streamlit",
    "SIMPANG WIROBRAJAN (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_wirobrajan.stream/playlist.m3u8",
    "SIMPANG SERANGAN (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_serangan.stream/playlist.m3u8",
    "SIMPANG GONDOMANAN (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_gondomanan.stream/playlist.m3u8",
    "SIMPANG PASAR GADING 2 (BARAT)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_PasarGading2.stream/playlist.m3u8"
}

# --- KOORDINAT LOKASI JOGJAKARTA ---
KOORDINAT_PETA = {
    "SIMPANG KM NOL (PTZ)": [-7.8012, 110.3647],
    "SIMPANG WIROBRAJAN (PTZ)": [-7.8011, 110.3524],
    "SIMPANG SERANGAN (PTZ)": [-7.8000, 110.3540],
    "SIMPANG GONDOMANAN (PTZ)": [-7.8023, 110.3694],
    "SIMPANG PASAR GADING 2 (BARAT)": [-7.8146, 110.3653]
}

if 'is_playing' not in st.session_state:
    st.session_state.is_playing = True

# --- SIDEBAR ---
st.sidebar.markdown('<h3 style="margin-top: 0;">⚙️ Konfigurasi Sistem</h3>', unsafe_allow_html=True)
pilihan_lokasi = st.sidebar.selectbox("📍 Titik Observasi", list(DATABASE_CCTV.keys()))

if DATABASE_CCTV[pilihan_lokasi] == "manual":
    youtube_url = st.sidebar.text_input("🔗 URL M3U8 Live Stream")
else:
    youtube_url = DATABASE_CCTV[pilihan_lokasi]

ROI_POLYGONS = {}
status_kalibrasi = False
if os.path.exists('roi_config.json'):
    with open('roi_config.json', 'r') as f:
        database_roi = json.load(f)
    if pilihan_lokasi in database_roi:
        for nama_area, titik in database_roi[pilihan_lokasi].items():
            ROI_POLYGONS[nama_area] = np.array(titik)
        if len(ROI_POLYGONS) > 0: status_kalibrasi = True

st.sidebar.markdown("<hr style='border-color: #30363d;'>", unsafe_allow_html=True)
st.sidebar.markdown('<h3>💾 Manajemen Database</h3>', unsafe_allow_html=True)
semua_file = os.listdir("traffic_data")
prefix_lokasi = f"log_{pilihan_lokasi.replace(' ', '_')}_"
file_tersedia = [f for f in semua_file if f.startswith(prefix_lokasi) and f.endswith(".csv")]

pilihan_tanggal_download = None
if file_tersedia:
    daftar_tanggal = [f.replace(prefix_lokasi, "").replace(".csv", "") for f in file_tersedia]
    daftar_tanggal.sort(reverse=True)
    pilihan_tanggal_download = st.sidebar.selectbox("📅 Ekstrak Data", daftar_tanggal)
    nama_file_terpilih = f"{prefix_lokasi}{pilihan_tanggal_download}.csv"
    path_file_terpilih = os.path.join("traffic_data", nama_file_terpilih)
    with open(path_file_terpilih, "rb") as f:
        st.sidebar.download_button("📥 Unduh CSV", data=f, file_name=nama_file_terpilih, mime="text/csv", use_container_width=True)

if 'history_data' not in st.session_state or st.sidebar.button("🔄 Segarkan Antarmuka", use_container_width=True):
    st.session_state.history_data = pd.DataFrame(columns=['Waktu (Detik)', 'Total Unit', 'Beban SMP'])
    st.session_state.start_time = time.time()
    st.session_state.last_log_time = time.time()

def train_traffic_forecaster(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Menit_Hari'] = df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute
        X = df[['Menit_Hari']].values
        y = df['Beban SMP'].values if 'Beban SMP' in df.columns else df['Total Unit'].values
        model_ml = RandomForestRegressor(n_estimators=50, random_state=42)
        model_ml.fit(X, y)
        return model_ml
    except:
        return None

model_prediksi_lokasi = None
if file_tersedia:
    path_arsip_terbaru = os.path.join("traffic_data", f"{prefix_lokasi}{daftar_tanggal[0]}.csv")
    model_prediksi_lokasi = train_traffic_forecaster(path_arsip_terbaru)

# --- NAVIGASI MULTI-TAB ---
tab1, tab2 = st.tabs(["👁️ Live Telemetry", "📊 Analytics & Insights"])

with tab2:
    st.markdown("### 📊 Historis Beban Jalan")
    if not file_tersedia:
        st.warning("Data historis belum tersedia.")
    else:
        st.markdown(f"<span style='color:#8b949e;'>Arsip Aktif: {pilihan_lokasi} ({pilihan_tanggal_download})</span>", unsafe_allow_html=True)
        try:
            df_historis = pd.read_csv(path_file_terpilih)
            df_historis['Timestamp'] = pd.to_datetime(df_historis['Timestamp'])
            kolom_y = 'Beban SMP' if 'Beban SMP' in df_historis.columns else 'Total Unit'
            
            puncak = df_historis[kolom_y].max()
            waktu_puncak = df_historis.loc[df_historis[kolom_y].idxmax(), 'Timestamp'].strftime("%H:%M:%S")
            rata_rata = round(df_historis[kolom_y].mean(), 1)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="glass-card"><div class="metric-title">📈 RATA-RATA BEBAN</div><div class="metric-value">{rata_rata} <span class="metric-unit">SMP</span></div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="glass-card"><div class="metric-title">🔥 TITIK TERTINGGI</div><div class="metric-value">{puncak} <span class="metric-unit">SMP</span></div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="glass-card"><div class="metric-title">⏰ JAM PUNCAK</div><div class="metric-value">{waktu_puncak}</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            fig_historis = px.area(df_historis, x='Timestamp', y=kolom_y, color_discrete_sequence=['#a8c0ff'])
            fig_historis.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_historis, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membaca file data. Error: {e}")

with tab1:
    col_status, col_btn = st.columns([4, 1])
    with col_status:
        status_text = "<span class='pulse-dot'></span> <b>Memonitor secara Real-Time</b>" if st.session_state.is_playing else "⏸️ <b>Pemantauan Dijeda</b>"
        st.markdown(f"<div style='padding: 10px 0;'>{status_text}</div>", unsafe_allow_html=True)
    with col_btn:
        if st.session_state.is_playing:
            if st.button("Jeda Sistem", use_container_width=True): st.session_state.is_playing = False; st.rerun()
        else:
            if st.button("Lanjutkan Sistem", use_container_width=True): st.session_state.is_playing = True; st.rerun()
            
    panel_peringatan_proaktif = st.empty()
    panel_alarm_darurat = st.empty()       
            
    col_video, col_data = st.columns([1.7, 1.0])
    
    with col_video:
        frame_window = st.empty() 
        st.markdown("<br><h3>📈 Kurva Arus Aktual</h3>", unsafe_allow_html=True)
        grafik_window = st.empty()

    with col_data:
        st.markdown("<h3>⚡ Telemetri Sensor</h3>", unsafe_allow_html=True)
        panel_metrik_1 = st.empty()
        st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
        panel_metrik_2 = st.empty()
        
        st.markdown("<br><h3>🗺️ Peta Radar</h3>", unsafe_allow_html=True)
        lat, lon = KOORDINAT_PETA.get(pilihan_lokasi, [-7.7956, 110.3695])
        
        m = folium.Map(location=[lat, lon], zoom_start=16, control_scale=True)
        folium.Marker(location=[lat, lon], tooltip=pilihan_lokasi, icon=folium.Icon(color="red", icon="camera")).add_to(m)
        components.html(m._repr_html_(), height=260)

# --- FUNGSI PENGAMBIL TOKEN DINAMIS (ANTI KEDALUWARSA) ---
def dapatkan_stream_aktif(master_url):
    if master_url == "manual": return master_url
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        resp = requests.get(master_url, headers=headers, timeout=5)
        for line in resp.text.splitlines():
            if line.endswith('.m3u8') and not line.startswith('#'):
                return urljoin(master_url, line)
        return master_url
    except:
        return master_url

if st.session_state.is_playing:
    with tab1: 
        if not status_kalibrasi:
            frame_window.error(f"❌ '{pilihan_lokasi}' belum dikalibrasi.")
        elif not youtube_url:
            frame_window.error("❌ Harap masukkan URL Streaming!")
        else:
            try:
                # Mengambil URL chunklist aktif untuk pertama kali
                url_aktif = dapatkan_stream_aktif(youtube_url)
                cap = cv2.VideoCapture(url_aktif)
                
                frame_count = 0 
                kendaraan_diam = [] 
                
                # --- PENGATURAN PARAMETER AMBANG BATAS BALANCED ---
                BATAS_JARAK_PIKSEL = 45  
                BATAS_WAKTU_MOGOK = 150.0  # Ubah dari 30.0 menjadi 150.0 detik (2.5 menit)
                
                while True:
                    ret, frame = cap.read()
                    if not ret: 
                        st.warning("⚠️ Token segmen habis. Meminta token baru ke server Dishub...")
                        time.sleep(1)
                        # Reconnect dengan mengambil token baru secara otomatis
                        url_aktif = dapatkan_stream_aktif(youtube_url)
                        cap = cv2.VideoCapture(url_aktif)
                        continue
                        
                    frame = cv2.resize(frame, (640, 360))
                    results = model.predict(frame, classes=[2, 3, 5, 7], conf=0.15, verbose=False)
                    
                    total_unit_kendaraan = 0
                    total_beban_smp = 0.0
                    deteksi_valid_sekarang = [] 
                    
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        cls_id = int(box.cls[0])
                        
                        for nama_area, poligon in ROI_POLYGONS.items():
                            if cv2.pointPolygonTest(poligon, (cx, cy), False) >= 0:
                                total_unit_kendaraan += 1
                                total_beban_smp += BOBOT_SMP.get(cls_id, 1.0)
                                deteksi_valid_sekarang.append((cx, cy, x1, y1, x2, y2, cls_id))
                                break
                    
                    waktu_sekarang = time.time()
                    waktu_lokal = datetime.datetime.now()
                    
                    with panel_peringatan_proaktif.container():
                        if model_prediksi_lokasi is not None:
                            waktu_prediksi = waktu_lokal + datetime.timedelta(minutes=30)
                            menit_prediksi = waktu_prediksi.hour * 60 + waktu_prediksi.minute
                            beban_teramal = model_prediksi_lokasi.predict([[menit_prediksi]])[0]
                            beban_teramal = round(float(beban_teramal), 1)
                            
                            if beban_teramal >= 12.0:
                                st.markdown(f"<div class='alert-box alert-ai-warn'><span style='color: #d96570; font-weight: 700;'>✨ AI PRECOGNITION:</span> Pukul {waktu_prediksi.strftime('%H:%M')}, diprediksi terjadi lonjakan arus hingga <b>{beban_teramal} SMP</b>.</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='alert-box alert-ai'><span style='color: #4285f4; font-weight: 700;'>✨ AI PRECOGNITION:</span> Kondisi arus 30 menit ke depan diprediksi stabil (~{beban_teramal} SMP).</div>", unsafe_allow_html=True)
                    
                    for det in deteksi_valid_sekarang:
                        cx, cy, x1, y1, x2, y2, cid = det
                        cocok = False
                        for kend in kendaraan_diam:
                            jarak = np.sqrt((cx - kend['pos'][0])**2 + (cy - kend['pos'][1])**2)
                            if jarak < BATAS_JARAK_PIKSEL:
                                kend['pos'] = (cx, cy); kend['box'] = (x1, y1, x2, y2); kend['last_seen'] = waktu_sekarang
                                cocok = True; break
                        if not cocok:
                            kendaraan_diam.append({'pos': (cx, cy), 'box': (x1, y1, x2, y2), 'first_seen': waktu_sekarang, 'last_seen': waktu_sekarang})
                    
                    kendaraan_diam = [k for k in kendaraan_diam if waktu_sekarang - k['last_seen'] < 2.0]
                    annotated_frame = results[0].plot()
                    
                    for nama_area, poligon in ROI_POLYGONS.items():
                        cv2.polylines(annotated_frame, [poligon], isClosed=True, color=(168, 192, 255), thickness=1)
                    
                    jumlah_anomali = 0
                    for kend in kendaraan_diam:
                        waktu_berhenti = waktu_sekarang - kend['first_seen']
                        if waktu_berhenti > BATAS_WAKTU_MOGOK:
                            jumlah_anomali += 1
                            bx1, by1, bx2, by2 = kend['box']
                            cv2.rectangle(annotated_frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 3)
                            cv2.putText(annotated_frame, f"BOTTLENECK", (int(bx1), int(by1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    with panel_alarm_darurat.container():
                        if jumlah_anomali > 0:
                            st.markdown(f"<div class='alert-box alert-danger'><h4 style='color:#ff4b4b; margin:0;'>🚨 BOTTLENECK TERDETEKSI</h4><p style='margin:5px 0 0 0; color:#e6edf3;'>Terdapat {jumlah_anomali} kendaraan statis melebihi {int(BATAS_WAKTU_MOGOK)} detik.</p></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='alert-box alert-safe'><h4 style='color:#2ea043; margin:0;'>✅ STATUS NORMAL</h4><p style='margin:5px 0 0 0; color:#8b949e;'>Arus lalu lintas terpantau lancar tanpa anomali stagnasi.</p></div>", unsafe_allow_html=True)

                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(rgb_frame, channels="RGB")
                    
                    with panel_metrik_1.container(): 
                        st.markdown(f'<div class="glass-card"><div class="metric-title">📊 VOLUME KASAR</div><div class="metric-value">{total_unit_kendaraan} <span class="metric-unit">Unit</span></div></div>', unsafe_allow_html=True)
                    with panel_metrik_2.container(): 
                        st.markdown(f'<div class="glass-card"><div class="metric-title">⚖️ BEBAN JALAN</div><div class="metric-value">{total_beban_smp} <span class="metric-unit">SMP</span></div></div>', unsafe_allow_html=True)
                    
                    if waktu_sekarang - st.session_state.last_log_time >= 5:
                        csv_dinamis = f"traffic_data/log_{pilihan_lokasi.replace(' ', '_')}_{waktu_lokal.strftime('%Y-%m-%d')}.csv"
                        df_log = pd.DataFrame([{'Timestamp': waktu_lokal.strftime("%Y-%m-%d %H:%M:%S"), 'Total Unit': total_unit_kendaraan, 'Beban SMP': total_beban_smp}])
                        file_exists = os.path.exists(csv_dinamis)
                        df_log.to_csv(csv_dinamis, mode='a', header=not file_exists, index=False)
                        st.session_state.last_log_time = waktu_sekarang
                    
                    frame_count += 1
                    if frame_count % 10 == 0:
                        waktu_relatif = round(waktu_sekarang - st.session_state.start_time, 1)
                        data_baru = pd.DataFrame([{'Waktu (Detik)': waktu_relatif, 'Total Unit': total_unit_kendaraan, 'Beban SMP': total_beban_smp}])
                        st.session_state.history_data = pd.concat([st.session_state.history_data, data_baru], ignore_index=True)
                        if len(st.session_state.history_data) > 50: st.session_state.history_data = st.session_state.history_data.iloc[-50:]
                        
                        fig = px.line(st.session_state.history_data, x='Waktu (Detik)', y=['Total Unit', 'Beban SMP'], color_discrete_sequence=['#a8c0ff', '#3f2b96'])
                        fig.update_layout(
                            legend_title_text='', 
                            margin=dict(l=0, r=0, t=10, b=0), 
                            xaxis_title="", 
                            yaxis_title="Nilai Metrik",
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        grafik_window.plotly_chart(fig, use_container_width=True)
                         
            except Exception as e:
                frame_window.error(f"🚨 **Gagal memuat kamera.** Error: {e}")
            finally:
                if 'cap' in locals(): cap.release()