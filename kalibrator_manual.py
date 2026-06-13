import cv2
import numpy as np
import json
import os
from vidgear.gears import CamGear

# --- DATABASE CCTV ---
DATABASE_CCTV = {
    "1": ("NOL KM (PTZ)", "https://www.youtube.com/watch?v=V9a2vVNBx2g"),
    "2": ("SIMPANG TUGU", "https://www.youtube.com/watch?v=1v52cQ1qJBA"),
    "3": ("MARGO UTOMO - UTARA TETEG", "https://www.youtube.com/watch?v=JUI1Wx4E25Q"),
    "4": ("SIMPANG MANDALA KRIDA (PTZ)", "https://www.youtube.com/watch?v=-c3KPR-mRcg"),
    "5": ("SIMPANG BALAIKOTA V. UTARA-BARAT", "http://youtube.com/watch?v=yicTxt9-pdE"),
    "6": ("SIMPANG AMONGROGO V. TIMUR", "https://www.youtube.com/watch?v=8LYEvmmfuZM"),
    "7": ("SIMPANG TERBAN V. TIMUR", "https://www.youtube.com/watch?v=EX47R3JdZe4"),
    "8": ("MARGO UTOMO - OPTIC TUGU", "https://www.youtube.com/watch?v=ghJg364QRkQ"),
    "9": ("SIMPANG TERBAN V. BARAT", "https://www.youtube.com/watch?v=heXWmGGHC9U")
}

# --- MENU TERMINAL ---
print("🚥 OPTIFLOW AI - KALIBRATOR POLIGON FLEKSIBEL 🚥")
print("="*50)
for key, (nama, url) in DATABASE_CCTV.items():
    print(f"[{key}] {nama}")
print("="*50)

pilihan = input("Pilih nomor lokasi CCTV yang ingin dikalibrasi: ")

if pilihan not in DATABASE_CCTV:
    print("❌ Pilihan tidak valid. Skrip dihentikan.")
    exit()

nama_lokasi, youtube_url = DATABASE_CCTV[pilihan]
print(f"\n✅ Membuka stream untuk: {nama_lokasi}...")
print("Mohon tunggu beberapa saat...\n")

# --- INISIALISASI STREAM ---
options = {"STREAM_RESOLUTION": "480p", "STREAM_PARAMS": {"nocheckcertificate": True}}
stream = CamGear(source=youtube_url, stream_mode=True, logging=False, **options).start()

# Mengambil satu frame statis yang jernih untuk digambar
frame_statis = None
for _ in range(30): # Lewati beberapa frame awal agar kamera fokus
    frame_statis = stream.read()
    
stream.stop() # Hentikan stream agar hemat internet, kita hanya butuh 1 gambar

if frame_statis is None:
    print("❌ Gagal mengambil gambar dari CCTV.")
    exit()

frame_statis = cv2.resize(frame_statis, (640, 360))
frame_bersih = frame_statis.copy() # Cadangan gambar tanpa coretan

# --- VARIABEL STATE MOUSE ---
titik_area_sekarang = []
semua_area_tersimpan = {}
indeks_area = 1

def gambar_poligon(event, x, y, flags, param):
    global titik_area_sekarang
    
    # Jika klik kiri, tambahkan titik ke memori
    if event == cv2.EVENT_LBUTTONDOWN:
        titik_area_sekarang.append([x, y])

cv2.namedWindow("OptiFlow Kalibrator")
cv2.setMouseCallback("OptiFlow Kalibrator", gambar_poligon)

print("🛠️ CARA PENGGUNAAN:")
print("- KLIK KIRI   : Tambah titik sudut jalan (bebas berapapun).")
print("- SPASI / 'C' : Tutup garis yang sedang digambar menjadi 1 Area Sensor.")
print("- HURUF 'R'   : Batalkan/Reset garis yang sedang digambar.")
print("- ENTER / 'S' : Simpan semua area ke database dan KELUAR.")

while True:
    frame_tampil = frame_bersih.copy()
    
    # 1. Gambar semua area yang SUDAH TERSIMPAN (Warna Hijau)
    for nama_area, poligon in semua_area_tersimpan.items():
        pts = np.array(poligon, np.int32)
        cv2.polylines(frame_tampil, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Efek warna transparan di dalam poligon hijau
        overlay = frame_tampil.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, frame_tampil, 0.7, 0, frame_tampil)
        cv2.putText(frame_tampil, nama_area, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 2. Gambar area yang SEDANG DIBUAT saat ini (Warna Merah)
    if len(titik_area_sekarang) > 0:
        pts = np.array(titik_area_sekarang, np.int32)
        # Gambar titiknya
        for pt in titik_area_sekarang:
            cv2.circle(frame_tampil, tuple(pt), 4, (0, 0, 255), -1)
        # Gambar garis yang menghubungkan (belum tertutup)
        if len(titik_area_sekarang) > 1:
            cv2.polylines(frame_tampil, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
            
    # Teks Instruksi di Layar
    cv2.putText(frame_tampil, "Klik kiri: Buat Titik | SPASI: Simpan Area | ENTER: Selesai", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame_tampil, f"Area ke-{indeks_area} (Titik: {len(titik_area_sekarang)})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("OptiFlow Kalibrator", frame_tampil)
    key = cv2.waitKey(1) & 0xFF

    # Tekan 'C' atau SPASI untuk menutup poligon dan menyimpannya
    if key == ord('c') or key == 32: 
        if len(titik_area_sekarang) >= 3: # Minimal 3 titik untuk membuat bentuk
            nama = f"Area Sensor {indeks_area}"
            semua_area_tersimpan[nama] = titik_area_sekarang.copy()
            print(f"✅ {nama} tersimpan dengan {len(titik_area_sekarang)} titik.")
            titik_area_sekarang = [] # Kosongkan untuk area berikutnya
            indeks_area += 1
        else:
            print("⚠️ Minimal butuh 3 titik untuk membuat area!")

    # Tekan 'R' untuk mereset titik yang sedang digambar
    elif key == ord('r'):
        titik_area_sekarang = []
        print("🔄 Titik area saat ini di-reset.")

    # Tekan 'S' atau ENTER untuk menyimpan semua data ke JSON dan keluar
    elif key == ord('s') or key == 13: 
        if len(semua_area_tersimpan) > 0:
            file_json = 'roi_config.json'
            
            # Baca data lama jika ada
            if os.path.exists(file_json):
                with open(file_json, 'r') as f:
                    database_roi = json.load(f)
            else:
                database_roi = {}
                
            # Update data untuk lokasi ini
            database_roi[nama_lokasi] = semua_area_tersimpan
            
            with open(file_json, 'w') as f:
                json.dump(database_roi, f, indent=4)
                
            print(f"\n💾 BERHASIL! Konfigurasi {nama_lokasi} disimpan ke '{file_json}'.")
            print("Silakan jalankan ulang aplikasi Streamlit Anda.")
        else:
            print("\n❌ Tidak ada area yang disimpan. Keluar tanpa menyimpan.")
        break

cv2.destroyAllWindows()