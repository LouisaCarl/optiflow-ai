import os
os.environ["QT_QPA_PLATFORM"] = "offscreen" 

import streamlit as st
import streamlit.components.v1 as components
import cv2
import time
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import folium
import requests
import math 
from urllib.parse import urljoin
from ultralytics import YOLO
from sklearn.ensemble import RandomForestRegressor

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="OptiFlow TMC Dashboard", layout="wide", initial_sidebar_state="expanded")

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0b0f19; color: #e2e8f0; }
    .stApp { background-color: #0b0f19; }
    .tmc-header { display: flex; justify-content: space-between; align-items: center; background: linear-gradient(90deg, #111827 0%, #0f172a 100%); padding: 15px 25px; border-radius: 8px; border: 1px solid #1e293b; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); }
    .tmc-title { font-family: 'Rajdhani', sans-serif; font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin: 0; letter-spacing: 1px;}
    .tmc-subtitle { font-size: 0.9rem; color: #94a3b8; margin: 0; text-transform: uppercase; letter-spacing: 2px;}
    .live-indicator { display: flex; align-items: center; font-family: 'Rajdhani', sans-serif; font-weight: 600; color: #ef4444; font-size: 1.2rem; letter-spacing: 1px;}
    .blink-dot { height: 12px; width: 12px; background-color: #ef4444; border-radius: 50%; margin-right: 8px; animation: blink 1s infinite; box-shadow: 0 0 8px #ef4444;}
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    .kpi-grid { display: flex; gap: 15px; margin-bottom: 20px; }
    .kpi-card { flex: 1; background-color: #111827; border-radius: 8px; padding: 15px 20px; border: 1px solid #1e293b; border-top: 4px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.3); position: relative; overflow: hidden; }
    .kpi-card.alert { border-top-color: #ef4444; }
    .kpi-card.warning { border-top-color: #f59e0b; }
    .kpi-card.success { border-top-color: #10b981; }
    .kpi-title { color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .kpi-value { font-family: 'Rajdhani', sans-serif; color: #f8fafc; font-size: 2.5rem; font-weight: 700; line-height: 1; }
    .kpi-unit { font-size: 1rem; color: #64748b; font-weight: 500; font-family: 'Inter', sans-serif;}
    .panel-container { background-color: #111827; border-radius: 8px; padding: 15px; border: 1px solid #1e293b; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .panel-title { font-family: 'Rajdhani', sans-serif; font-size: 1.2rem; font-weight: 600; color: #cbd5e1; border-bottom: 1px solid #334155; padding-bottom: 8px; margin-bottom: 15px; margin-top:0;}
    
    /* MODIFIKASI: Layout Insight Card Horizontal */
    .insights-grid { display: flex; gap: 15px; width: 100%; }
    .insight-card { flex: 1; background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .insight-badge { color: #ffffff; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.75rem; text-transform: uppercase; display: inline-block; margin-bottom: 8px;}
    .insight-text { font-size: 0.9rem; color: #e2e8f0; margin: 0; line-height: 1.5;}

    .incident-alert { background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 12px 15px; border-radius: 4px; margin-bottom: 10px; }
    .incident-title { color: #ef4444; font-weight: 700; margin:0 0 4px 0; font-size: 0.95rem; text-transform: uppercase;}
    .incident-desc { color: #e2e8f0; margin:0; font-size: 0.85rem;}
    .ai-alert { background: rgba(139, 92, 246, 0.1); border-left: 4px solid #8b5cf6; padding: 12px 15px; border-radius: 4px; margin-bottom: 10px; }
    .ai-title { color: #a78bfa; font-weight: 700; margin:0 0 4px 0; font-size: 0.95rem; text-transform: uppercase;}
    [data-testid="stSidebar"] { background-color: #080c14 !important; border-right: 1px solid #1e293b; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# STATE MANAGEMENT & PASSWORD STORAGE
# ==========================================
os.makedirs("traffic_data", exist_ok=True)
os.makedirs("temp_upload", exist_ok=True)

if 'admin_password' not in st.session_state:
    st.session_state.admin_password = "123456"

@st.cache_resource
def load_yolo_models():
    return YOLO('yolov8n.pt'), YOLO('yolo_ambulans.pt')

model_base, model_amb = load_yolo_models()
BOBOT_SMP = {2: 1.0, 3: 0.5, 5: 2.5, 7: 2.5, 99: 1.0}

DATABASE_CCTV = {
    "SIMPANG KM NOL (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_kmnol.stream/playlist.m3u8",
    "SIMPANG WIROBRAJAN (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_wirobrajan.stream/playlist.m3u8",
    "SIMPANG SERANGAN (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_serangan.stream/playlist.m3u8",
    "SIMPANG GONDOMANAN (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_gondomanan.stream/playlist.m3u8",
    "SIMPANG PASAR GADING 2 (BARAT)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_PasarGading2.stream/playlist.m3u8",
    "SIMPANG SGM (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_simpangsgm.stream/playlist.m3u8",
    "SIMPANG MANTRIGAWEN (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Mantrigawen_PTZ.stream/playlist.m3u8",
    "SIMPANG BAUSASRAN (PTZ)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_S4_Bausasran.stream/playlist.m3u8",
    "UTARA-TIMUR GARDENA (JL. URIP SUMOHARJO) V. TIMUR": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Utara-Timur_Gardena_Jl_Urip%20Sumoharjo_V_Timur.stream/playlist.m3u8",
    "SIMPANG KARANG TUNGGAL (S3 JL. KOL.SUGIYONO)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SopMerah2.stream/playlist.m3u8",
    "UNGGAH VIDEO LOKAL (MP4)": "upload"
}

KOORDINAT_PETA = {
    "SIMPANG KM NOL (PTZ)": [-7.8012, 110.3647], "SIMPANG WIROBRAJAN (PTZ)": [-7.8011, 110.3524],
    "SIMPANG SERANGAN (PTZ)": [-7.8000, 110.3540], "SIMPANG GONDOMANAN (PTZ)": [-7.8023, 110.3694],
    "SIMPANG PASAR GADING 2 (BARAT)": [-7.8146, 110.3653], "SIMPANG SGM (PTZ)": [-7.8003, 110.3920], 
    "SIMPANG MANTRIGAWEN (PTZ)": [-7.8105, 110.3735], "SIMPANG BAUSASRAN (PTZ)": [-7.7965, 110.3785], 
    "UTARA-TIMUR GARDENA (JL. URIP SUMOHARJO) V. TIMUR": [-7.7828, 110.3824], "SIMPANG KARANG TUNGGAL (S3 JL. KOL.SUGIYONO)": [-7.8165, 110.3730]
}

if 'is_playing' not in st.session_state: st.session_state.is_playing = True
if 'last_toast_time' not in st.session_state: st.session_state.last_toast_time = 0

def dapatkan_stream_aktif(master_url):
    try:
        resp = requests.get(master_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        for line in resp.text.splitlines():
            if line.endswith('.m3u8') and not line.startswith('#'): return urljoin(master_url, line)
        return master_url
    except: return master_url

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
    except: return None

# ==========================================
# SIDEBAR KONFIGURASI CONTROL & AUTHENTICATION
# ==========================================
st.sidebar.markdown('<h2 style="color:#38bdf8; font-family:\'Rajdhani\'; margin-top:0;">TMC CONTROL</h2>', unsafe_allow_html=True)

akses_pengguna = st.sidebar.radio("🔑 User Access", ["👤 User", "👨‍💻 Admin Developer"])
is_admin = False

if akses_pengguna == "👨‍💻 Admin Developer":
    input_pwd = st.sidebar.text_input("🛡️ Sandi Pengembang", type="password")
    if input_pwd == st.session_state.admin_password:
        is_admin = True
        st.sidebar.success("Otorisasi Pengembang Diterima!")
        
        with st.sidebar.expander("⚙️ Manajemen Kredensial"):
            pwd_baru = st.text_input("Sandi Baru", type="password")
            konfirmasi_pwd = st.text_input("Konfirmasi Sandi", type="password")
            if st.button("Perbarui Sandi"):
                if pwd_baru == konfirmasi_pwd and pwd_baru != "":
                    st.session_state.admin_password = pwd_baru
                    st.success("Sandi pengembang diperbarui!")
                    st.rerun()
                else: st.error("Sandi tidak cocok/kosong!")
    else:
        if input_pwd != "": st.sidebar.error("Sandi Pengembang Salah!")

st.sidebar.markdown("<hr style='border-color: #1e293b; margin: 10px 0;'>", unsafe_allow_html=True)
pilihan_lokasi = st.sidebar.selectbox("📍 PILIH UNIT CCTV", list(DATABASE_CCTV.keys()))
youtube_url = DATABASE_CCTV[pilihan_lokasi]

if youtube_url == "upload":
    uploaded_file = st.sidebar.file_uploader("📂 UNGGAH VIDEO (.MP4)", type=["mp4"])
    if uploaded_file is not None:
        temp_video_path = os.path.join("temp_upload", "video_sementara.mp4")
        with open(temp_video_path, "wb") as f: f.write(uploaded_file.read())
        youtube_url = temp_video_path
    else: youtube_url = None

ROI_POLYGONS = {}
status_kalibrasi = False

if pilihan_lokasi == "UNGGAH VIDEO LOKAL (MP4)":
    ROI_POLYGONS["Area_Layar_Penuh"] = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]])
    status_kalibrasi = True
elif os.path.exists('roi_config.json'):
    with open('roi_config.json', 'r') as f: database_roi = json.load(f)
    if pilihan_lokasi in database_roi:
        for nama_area, titik in database_roi[pilihan_lokasi].items(): ROI_POLYGONS[nama_area] = np.array(titik)
        if len(ROI_POLYGONS) > 0: status_kalibrasi = True

# ==========================================
# TRAINING MODEL ML (DI LUAR LOOP VIDEO)
# ==========================================
model_prediksi_lokasi = None
semua_file = os.listdir("traffic_data")
prefix_lokasi = f"log_{pilihan_lokasi.replace(' ', '_')}_"
file_tersedia = [f for f in semua_file if f.startswith(prefix_lokasi) and f.endswith(".csv")]

if file_tersedia:
    daftar_tanggal = [f.replace(prefix_lokasi, "").replace(".csv", "") for f in file_tersedia]
    daftar_tanggal.sort(reverse=True)
    path_arsip_terbaru = os.path.join("traffic_data", f"{prefix_lokasi}{daftar_tanggal[0]}.csv")
    model_prediksi_lokasi = train_traffic_forecaster(path_arsip_terbaru)

if is_admin:
    st.sidebar.markdown('<h3 style="color:#94a3b8; font-size:1rem; font-family:\'Rajdhani\';">💾 ROOT DATABASE</h3>', unsafe_allow_html=True)
    if file_tersedia:
        pilihan_tanggal_download = st.sidebar.selectbox("ARSIP TANGGAL", daftar_tanggal)
        nama_file_terpilih = f"{prefix_lokasi}{pilihan_tanggal_download}.csv"
        path_file_terpilih = os.path.join("traffic_data", nama_file_terpilih)
        with open(path_file_terpilih, "rb") as f:
            st.sidebar.download_button("⬇️ DOWNLOAD RAW CSV", data=f, file_name=nama_file_terpilih, mime="text/csv", use_container_width=True)

if 'history_data' not in st.session_state or st.sidebar.button("🔄 BOOTSTRAP ENGINE", use_container_width=True):
    st.session_state.history_data = pd.DataFrame(columns=['Waktu (Detik)', 'Total Unit', 'Beban SMP'])
    st.session_state.start_time = time.time()
    st.session_state.last_log_time = time.time()

if st.sidebar.button("⏯️ SWITCH STREAM STATE", use_container_width=True): 
    st.session_state.is_playing = not st.session_state.is_playing
    st.rerun()


# ==========================================
# LAYOUT UTAMA (COMMAND CENTER MASTER GRID)
# ==========================================
label_header = "USER" if akses_pengguna == "👤 User" else "ADMIN"

st.markdown(f"""
<div class="tmc-header">
    <div>
        <h1 class="tmc-title">OPTIFLOW TRAFFIC MANAGEMENT</h1>
        <p class="tmc-subtitle">NODE ACCESSIBILITY: {label_header} | CURRENT NODE: {pilihan_lokasi}</p>
    </div>
    <div class="live-indicator">
        <span class="blink-dot"></span> {'MONITORING AKTIF' if st.session_state.is_playing else 'STREAM DIJEDA'}
    </div>
</div>
""", unsafe_allow_html=True)

kpi_container = st.empty()

# --- RESTRUKTURISASI TATA LETAK KOLOM ---
col1, col2 = st.columns([7, 3])

with col1:
    st.markdown('<h3 class="panel-title">📹 LIVE COMPUTER VISION FEED</h3>', unsafe_allow_html=True)
    video_frame = st.empty()
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    
    st.markdown('<h3 class="panel-title">📈 LIVE TRAFFIC TRENDS</h3>', unsafe_allow_html=True)
    chart_frame = st.empty()
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    
    # Insights dipindahkan ke kolom 1 di bawah grafik
    st.markdown('<h3 class="panel-title">📑 INTERACTIVE EXECUTIVE INSIGHTS</h3>', unsafe_allow_html=True)
    insights_frame = st.empty()

with col2:
    st.markdown('<h3 class="panel-title">⚠️ INCIDENT & AI ALERTS</h3>', unsafe_allow_html=True)
    status_frame = st.empty()
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    
    # Map container naik ke atas menggantikan insights
    map_container = st.empty()

def update_kpi_cards(vol, smp, bottleneck_count, pred_smp):
    status_class = "alert" if bottleneck_count > 0 else "success"
    status_text = "CONGESTED" if bottleneck_count > 0 else "OPTIMIZED"
    html = f"""
    <div class="kpi-grid">
        <div class="kpi-card {status_class}">
            <div class="kpi-title">Intersection Status</div>
            <div class="kpi-value">{status_text}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Traffic Volume</div>
            <div class="kpi-value">{vol} <span class="kpi-unit">Vehicles</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Road Load Index</div>
            <div class="kpi-value">{smp} <span class="kpi-unit">SMP</span></div>
        </div>
        <div class="kpi-card warning">
            <div class="kpi-title">30 Min Forecast</div>
            <div class="kpi-value">{pred_smp} <span class="kpi-unit">SMP</span></div>
        </div>
    </div>
    """
    kpi_container.markdown(html, unsafe_allow_html=True)

if pilihan_lokasi in KOORDINAT_PETA:
    with map_container.container():
        st.markdown('<div class="panel-container"><h3 class="panel-title">📍 SPATIAL MAP MAPBOX</h3>', unsafe_allow_html=True)
        lat, lon = KOORDINAT_PETA[pilihan_lokasi]
        m = folium.Map(location=[lat, lon], zoom_start=16, tiles="CartoDB positron", control_scale=True)
        folium.Marker(location=[lat, lon], tooltip=pilihan_lokasi, icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
        components.html(m._repr_html_(), height=250) # Disesuaikan height-nya agar pas
        st.markdown('</div>', unsafe_allow_html=True)
else:
    with map_container.container():
        st.markdown('<div class="panel-container"><h3 class="panel-title">📍 SPATIAL MAP</h3><p style="color:#64748b; font-size:0.9rem;">Map telemetry offline for local video analysis.</p></div>', unsafe_allow_html=True)

if st.session_state.is_playing:
    if not status_kalibrasi:
        video_frame.error("SYSTEM HALTED: Calibration missing for this node.")
    elif not youtube_url:
        video_frame.warning("WAITING FOR INPUT: Please upload MP4 or select active feed.")
    else:
        try:
            url_aktif = youtube_url if pilihan_lokasi == "UNGGAH VIDEO LOKAL (MP4)" else dapatkan_stream_aktif(youtube_url)
            cap = cv2.VideoCapture(url_aktif)
            
            frame_count = 0 
            kendaraan_diam = [] 
            BATAS_JARAK_PIKSEL = 45  
            BATAS_WAKTU_MOGOK = 150.0  
            
            while True:
                ret, frame = cap.read()
                if not ret: 
                    if pilihan_lokasi == "UNGGAH VIDEO LOKAL (MP4)":
                        st.session_state.is_playing = False; st.rerun(); break
                    else:
                        time.sleep(1)
                        cap = cv2.VideoCapture(dapatkan_stream_aktif(youtube_url))
                        continue
                    
                frame = cv2.resize(frame, (640, 360))
                
                results_base = model_base.predict(frame, classes=[2, 3, 5, 7], conf=0.15, verbose=False)
                results_amb = model_amb.predict(frame, conf=0.25, verbose=False)
                
                total_unit_kendaraan = 0
                total_beban_smp = 0.0
                deteksi_valid_sekarang = [] 
                
                for box in results_base[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0]; cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cls_id = int(box.cls[0])
                    
                    for nama_area, poligon in ROI_POLYGONS.items():
                        if cv2.pointPolygonTest(poligon, (cx, cy), False) >= 0:
                            total_unit_kendaraan += 1
                            total_beban_smp += BOBOT_SMP.get(cls_id, 1.0)
                            deteksi_valid_sekarang.append((cx, cy, x1, y1, x2, y2))
                            break

                for box in results_amb[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]); cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    for nama_area, poligon in ROI_POLYGONS.items():
                        if cv2.pointPolygonTest(poligon, (cx, cy), False) >= 0:
                            total_unit_kendaraan += 1
                            total_beban_smp += BOBOT_SMP.get(99, 1.0)
                            break
                
                annotated_frame = results_base[0].plot(line_width=1, font_size=0.4) 
                for box in results_amb[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1) 
                    cv2.putText(annotated_frame, f"AMBULANCE", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                
                waktu_sekarang = time.time()
                
                for det in deteksi_valid_sekarang:
                    cx, cy, x1, y1, x2, y2 = det
                    cocok = False
                    for kend in kendaraan_diam:
                        if np.sqrt((cx - kend['pos'][0])**2 + (cy - kend['pos'][1])**2) < BATAS_JARAK_PIKSEL:
                            kend['pos'] = (cx, cy); kend['box'] = (x1, y1, x2, y2); kend['last_seen'] = waktu_sekarang
                            cocok = True; break
                    if not cocok: kendaraan_diam.append({'pos': (cx, cy), 'box': (x1, y1, x2, y2), 'first_seen': waktu_sekarang, 'last_seen': waktu_sekarang})
                
                kendaraan_diam = [k for k in kendaraan_diam if waktu_sekarang - k['last_seen'] < 2.0]
                for poligon in ROI_POLYGONS.values(): cv2.polylines(annotated_frame, [poligon], isClosed=True, color=(14, 165, 233), thickness=1)
                
                jumlah_anomali = 0
                for kend in kendaraan_diam:
                    if (waktu_sekarang - kend['first_seen']) > BATAS_WAKTU_MOGOK:
                        jumlah_anomali += 1
                        bx1, by1, bx2, by2 = kend['box']
                        cv2.rectangle(annotated_frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 1) 
                        cv2.putText(annotated_frame, f"BOTTLENECK", (int(bx1), int(by1)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                if jumlah_anomali > 0 and (waktu_sekarang - st.session_state.last_toast_time) > 45:
                    st.toast(f"🚨 ALERT: Terdeteksi {jumlah_anomali} titik kemacetan di area sensor!", icon="⚠️")
                    st.session_state.last_toast_time = waktu_sekarang

                video_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                beban_teramal = 0.0
                if model_prediksi_lokasi is not None:
                    wp = datetime.datetime.now() + datetime.timedelta(minutes=30)
                    beban_teramal = round(float(model_prediksi_lokasi.predict([[wp.hour * 60 + wp.minute]])[0]), 1)
                
                update_kpi_cards(total_unit_kendaraan, total_beban_smp, jumlah_anomali, beban_teramal)
                
                alert_html = '<div class="panel-container">'
                if jumlah_anomali > 0:
                    alert_html += f'<div class="incident-alert"><h4 class="incident-title">CRITICAL: BOTTLENECK</h4><p class="incident-desc">{jumlah_anomali} static vehicle(s) detected blocking traffic flow.</p></div>'
                else:
                    alert_html += '<div style="color:#10b981; margin-bottom:15px; font-size:0.9rem;">✔️ No active physical bottlenecks.</div>'
                
                if beban_teramal >= 12.0:
                    alert_html += f'<div class="ai-alert"><h4 class="ai-title">AI FORECAST WARNING</h4><p class="incident-desc" style="color:#c4b5fd;">High congestion predicted in 30 mins ({beban_teramal} SMP).</p></div>'
                else:
                    alert_html += f'<div style="color:#8b5cf6; font-size:0.9rem;">✨ AI Forecast: Flow remains optimal (~{beban_teramal} SMP).</div>'
                alert_html += '</div>'
                status_frame.markdown(alert_html, unsafe_allow_html=True)

                # --- INTERACTIVE INSIGHT REPORT ON APP (DIUBAH KE FORMAT HORIZONTAL GRID) ---
                rec_signal = "SIKLUS NORMAL"
                rec_color = "#10b981"
                rec_desc = "Kapasitas lajur jalan raya masih mencukupi untuk menampung volume arus kendaraan saat ini."
                
                if total_beban_smp > 15.0 or jumlah_anomali > 0:
                    rec_signal = "PERPANJANG SIKLUS HIJAU (+15 DETIK)"
                    rec_color = "#f59e0b"
                    rec_desc = "AI mendeteksi lonjakan volume kendaraan. Rekomendasi penambahan waktu hijau untuk menguras antrean jalan."
                if jumlah_anomali > 3:
                    rec_signal = "DISPATCH UNIT PATROLI KEPOLISIAN"
                    rec_color = "#ef4444"
                    rec_desc = "Stagnasi kritis terdeteksi melebihi batas toleransi. Diperlukan intervensi fisik manual oleh petugas di lapangan."

                # HTML Baru menggunakan Flexbox Grid (.insights-grid)
                insights_html = f"""
                <div class="insights-grid">
                    <div class="insight-card" style="border-left: 4px solid {rec_color};">
                        <span class="insight-badge" style="background-color: {rec_color};">Traffic Intelligence Control</span>
                        <h4 style="margin: 8px 0 4px 0; font-family:'Rajdhani'; font-size:1.2rem; color:{rec_color}; font-weight:700;">{rec_signal}</h4>
                        <p class="insight-text">{rec_desc}</p>
                    </div>
                    <div class="insight-card" style="border-left: 4px solid #3b82f6;">
                        <span class="insight-badge" style="background-color: #3b82f6;">Analisis Efisiensi Ekonomi</span>
                        <p class="insight-text" style="margin-top: 8px;">Beban saat ini senilai <b>{total_beban_smp} SMP</b> berpotensi meningkatkan delay simpang sebesar <b>{round(total_beban_smp * 1.2, 1)} detik/kendaraan</b> jika siklus lampu tidak diadaptasikan.</p>
                    </div>
                </div>
                """
                insights_frame.markdown(insights_html, unsafe_allow_html=True)

                waktu_lokal = datetime.datetime.now()
                if waktu_sekarang - st.session_state.last_log_time >= 30:
                    csv_dinamis = f"traffic_data/log_{pilihan_lokasi.replace(' ', '_')}_{waktu_lokal.strftime('%Y-%m-%d')}.csv"
                    df_log = pd.DataFrame([{'Timestamp': waktu_lokal.strftime("%Y-%m-%d %H:%M:%S"), 'Total Unit': total_unit_kendaraan, 'Beban SMP': total_beban_smp}])
                    df_log.to_csv(csv_dinamis, mode='a', header=not os.path.exists(csv_dinamis), index=False)
                    st.session_state.last_log_time = waktu_sekarang
                
                frame_count += 1
                if frame_count % 10 == 0:
                    wr = round(waktu_sekarang - st.session_state.start_time, 1)
                    st.session_state.history_data = pd.concat([st.session_state.history_data, pd.DataFrame([{'Waktu (Detik)': wr, 'Total Unit': total_unit_kendaraan, 'Beban SMP': total_beban_smp}])], ignore_index=True)
                    if len(st.session_state.history_data) > 40: st.session_state.history_data = st.session_state.history_data.iloc[-40:]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=st.session_state.history_data['Waktu (Detik)'], y=st.session_state.history_data['Beban SMP'], mode='lines', fill='tozeroy', name='Beban SMP', line=dict(color='#0ea5e9', width=3), fillcolor='rgba(14, 165, 233, 0.2)'))
                    fig.add_trace(go.Scatter(x=st.session_state.history_data['Waktu (Detik)'], y=st.session_state.history_data['Total Unit'], mode='lines', name='Total Unit', line=dict(color='#8b5cf6', width=2, dash='dot')))
                    
                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor='rgba(17, 24, 39, 1)', plot_bgcolor='rgba(17, 24, 39, 1)',
                        margin=dict(l=20, r=20, t=10, b=20), height=250,
                        xaxis=dict(showgrid=True, gridcolor='#1e293b', title=""), 
                        yaxis=dict(showgrid=True, gridcolor='#1e293b', title=""),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    chart_frame.plotly_chart(fig, use_container_width=True) 
                     
        except Exception as e:
            video_frame.error(f"ENGINE FAILURE: {e}")
        finally:
            if 'cap' in locals(): cap.release()