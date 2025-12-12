# Laporan Teknis: Sistem Avatar VTuber Berbasis Sprite 2D

## 1. Pendahuluan
Sistem ini adalah aplikasi **VTuber (Virtual YouTuber) 2D sederhana** yang menggunakan teknologi *computer vision* untuk menggerakkan avatar sprite (gambar statis) berdasarkan gerakan pengguna secara *real-time* melalui webcam. Sistem mendeteksi orientasi kepala, arah tubuh, dan pose tangan untuk mengganti gambar avatar secara dinamis tanpa memerlukan model 3D yang kompleks (Rigging).

## 2. Arsitektur Sistem
Sistem dibangun menggunakan bahasa pemrograman **Python** dengan ketergantungan utama pada pustaka berikut:

* **OpenCV (`cv2`):** Bertanggung jawab atas pemrosesan citra, manipulasi frame, dan rendering sprite (overlay) ke layar.
* **MediaPipe (`mp.solutions.holistic`):** Digunakan untuk mendeteksi *landmarks* (titik kunci) pada wajah, tubuh, dan tangan dengan presisi tinggi.
* **NumPy:** Digunakan untuk operasi matriks dan perhitungan trigonometri (aljabar linear) dalam menghitung rotasi dan sudut vektor.

### Alur Kerja Utama
1.  **Input:** Mengambil frame video dari webcam ($1280 \times 720$).
2.  **Deteksi:** MediaPipe Holistic memproses frame untuk mendapatkan koordinat 3D tubuh dan wajah.
3.  **Kalkulasi:** Menghitung rotasi kepala, kemiringan tubuh, dan sudut lengan berdasarkan koordinat landmarks.
4.  **Logika State:** Menentukan status pose (kiri, kanan, depan, tangan lurus/bengkok) menggunakan *threshold*.
5.  **Seleksi Sprite:** Memilih file gambar `.png` yang sesuai dengan status pose saat ini.
6.  **Rendering:** Menggabungkan (*alpha blending*) sprite terpilih ke latar belakang (canvas).

## 3. Logika Deteksi dan Perhitungan

### A. Rotasi Kepala (Head Rotation)
Sistem menggunakan algoritma **PnP (Perspective-n-Point)** untuk memperkirakan rotasi kepala (Yaw).
* **Metode:** `cv2.solvePnP`
* **Input:** Titik 2D dari wajah pengguna (MediaPipe Face Mesh: Hidung, Dagu, Mata, Mulut) dipetakan ke model wajah standar 3D.
* **Output:** Estimasi sudut rotasi (Yaw).
* **Smoothing:** Menggunakan rumus *Exponential Moving Average* untuk mengurangi getaran (*jitter*):
    ```python
    current_rot = prev_rot * (1 - smooth_factor) + new_yaw * smooth_factor
    ```

### B. Arah Tubuh (Body Direction)
Arah tubuh dideteksi berdasarkan perbedaan koordinat vertikal ($y$) antara bahu kiri dan bahu kanan (Landmark 11 dan 12).
* **Rumus:** $\Delta y = y_{bahu\_kiri} - y_{bahu\_kanan}$
* **Logika:**
    * Jika $\Delta y > 0.015$ (positif signifikan), tubuh dianggap menghadap **KIRI**.
    * Jika $\Delta y < -0.015$ (negatif signifikan), tubuh dianggap menghadap **KANAN**.
    * Selain itu, tubuh dianggap **DEPAN**.

### C. Pose Lengan (Arm Pose)
Sistem mendeteksi apakah lengan ditekuk atau lurus menggunakan perhitungan sudut vektor antara tiga titik: Bahu ($A$), Siku ($B$), dan Pergelangan Tangan ($C$).

Rumus Kosinus Vektor:
$$
\text{Sudut} = \arccos \left( \frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}| \times |\vec{BC}|} \right) \times \frac{180}{\pi}
$$

* **Threshold:** Jika sudut < **110 derajat**, lengan dianggap `BENT` (menekuk/melambai).

## 4. Sistem Sprite (Aset & Seleksi)

Sistem bergantung pada file gambar eksternal yang harus berada di folder `position/`. Logika pemilihan sprite bersifat hierarkis (prioritas).

### Hierarki Pemilihan Sprite
1.  **Prioritas 1 (Tangan):** Jika tangan terdeteksi `BENT`, sistem memprioritaskan sprite tangan (`lf_hand` atau `rg_hand`).
2.  **Prioritas 2 (Kombinasi):** Jika kepala dan tubuh menoleh ke arah yang sama, gunakan sprite kombinasi (`lf_face_body`, `rg_face_body`).
3.  **Prioritas 3 (Kepala):** Jika hanya kepala yang menoleh, gunakan sprite wajah (`lf_face`, `rg_face`).
4.  **Default:** Jika tidak ada kondisi terpenuhi, tampilkan `front.png`.

### Tabel Aset Gambar
| Nama File | Fungsi Visual |
| :--- | :--- |
| `front.png` | Pose standar menghadap depan. |
| `lf_face.png` | Wajah menoleh ke kiri. |
| `rg_face.png` | Wajah menoleh ke kanan. |
| `lf_face_body.png` | Wajah dan tubuh ke kiri. |
| `rg_face_body.png` | Wajah dan tubuh ke kanan. |
| `lf_hand.png` | Pose tangan kiri naik/menyapa. |
| `rg_hand.png` | Pose tangan kanan naik/menyapa. |
| `bg.png` | Latar belakang aplikasi. |

## 5. Fitur Tambahan & Utilitas

* **Position Smoothing:** Posisi avatar ($x$) diperhalus dengan faktor `0.7` agar pergerakan avatar mengikuti pengguna tidak terlihat patah-patah.
* **Debug Mode:** Menekan tombol `s` menampilkan statistik *real-time* di layar, termasuk nilai rotasi kepala, selisih Y bahu, dan sudut siku.
* **Flip Frame:** Kamera di-*flip* secara horizontal (`cv2.flip(frame, 1)`) agar gerakan terasa seperti cermin yang natural.

## 6. Cara Penggunaan

1.  **Persiapan Folder:**
    Pastikan struktur direktori:
    ```text
    /project_root
      ├── main.py
      └── /position
          ├── front.png
          ├── ... (aset lainnya)
    ```
2.  **Instalasi Library:**
    ```bash
    pip install opencv-python mediapipe numpy
    ```
3.  **Eksekusi:**
    Jalankan perintah `python main.py`. Tekan `q` untuk keluar, dan `s` untuk melihat data sensor debug.

## 7. Kesimpulan
Kode ini menyediakan kerangka kerja dasar yang fungsional untuk VTuber 2D. Penggunaan `Enum` untuk manajemen *state* membuat kode mudah dibaca dan dimodifikasi. Keunggulan utamanya adalah performa yang ringan karena menggunakan manipulasi gambar 2D sederhana alih-alih rendering 3D yang berat, sehingga dapat berjalan lancar di komputer dengan spesifikasi menengah.
