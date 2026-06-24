import cv2
import numpy as np
import json
import os
import requests
from urllib.parse import urljoin

# --- DATABASE 10 CCTV AKTIF TERFILTER ---
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
    "SIMPANG KARANG TUNGGAL (S3 JL. KOL.SUGIYONO)": "https://cctvjss.jogjakota.go.id/atcs/ATCS_Lampu_Merah_SopMerah2.stream/playlist.m3u8"
}

titik_roi = []
sedang_menggambar = False
frame_copy = None

def mouse_callback(event, x, y, flags, param):
    global titik_roi, sedang_menggambar
    if event == cv2.EVENT_LBUTTONDOWN:
        titik_roi.append((x, y))
        sedang_menggambar = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(titik_roi) > 0: titik_roi.pop()

def dapatkan_stream_aktif(master_url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(master_url, headers=headers, timeout=5)
        for line in resp.text.splitlines():
            if line.endswith('.m3u8') and not line.startswith('#'):
                return urljoin(master_url, line)
        return master_url
    except:
        return master_url

def kalibrasi_cctv():
    global titik_roi, frame_copy
    print("=== KALIBRASI ROI JOGJAKARTA (DYNAMIC TOKEN) ===")
    daftar_lokasi = list(DATABASE_CCTV.keys())
    for i, lokasi in enumerate(daftar_lokasi): print(f"{i + 1}. {lokasi}")
        
    pilihan = int(input("\nPilih nomor CCTV: ")) - 1
    nama_lokasi = daftar_lokasi[pilihan]
    url_master = DATABASE_CCTV[nama_lokasi]
    
    print(f"\nMeminta token aktif untuk {nama_lokasi}...")
    url_aktif = dapatkan_stream_aktif(url_master)
    print(f"Token didapat: {url_aktif}")
    
    cap = cv2.VideoCapture(url_aktif)
    
    berhasil = False
    for _ in range(15): 
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 360))
            frame_asli = frame.copy()
            frame_copy = frame.copy()
            berhasil = True
            break
            
    cap.release()
    
    if not berhasil:
        print("❌ Gagal. Server mungkin sedang offline.")
        return

    cv2.namedWindow(f"Kalibrasi: {nama_lokasi}")
    cv2.setMouseCallback(f"Kalibrasi: {nama_lokasi}", mouse_callback)

    while True:
        frame_tampil = frame_asli.copy()
        if len(titik_roi) > 0:
            for pt in titik_roi: cv2.circle(frame_tampil, pt, 4, (0, 0, 255), -1)
            if len(titik_roi) > 1: cv2.polylines(frame_tampil, [np.array(titik_roi)], False, (0, 255, 255), 2)
            if len(titik_roi) > 2: cv2.polylines(frame_tampil, [np.array(titik_roi)], True, (0, 255, 0), 2)

        cv2.imshow(f"Kalibrasi: {nama_lokasi}", frame_tampil)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and len(titik_roi) >= 3:
            if os.path.exists('roi_config.json'):
                with open('roi_config.json', 'r') as f: data_roi = json.load(f)
            else:
                data_roi = {}
                
            if nama_lokasi not in data_roi: 
                data_roi[nama_lokasi] = {}
            data_roi[nama_lokasi]["Area_Jalan"] = titik_roi
            with open('roi_config.json', 'w') as f: json.dump(data_roi, f, indent=4)
            print(f"\n✅ BERHASIL DISIMPAN!")
            break
        elif key == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    kalibrasi_cctv()