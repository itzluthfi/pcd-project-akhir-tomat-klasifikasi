# 🍅 Sistem Klasifikasi Kematangan Buah Tomat

## 📋 Deskripsi Proyek

Aplikasi berbasis **Computer Vision** dan **Machine Learning** untuk mengklasifikasikan tingkat kematangan buah tomat secara otomatis menggunakan analisis citra digital. Sistem ini menggunakan kombinasi **GLCM (Gray Level Co-occurrence Matrix)** untuk ekstraksi fitur tekstur dan **Color Moment** untuk ekstraksi fitur warna, kemudian mengklasifikasikan menggunakan algoritma **Support Vector Machine (SVM)**.

### 🎯 Tujuan

- Mengotomatisasi proses klasifikasi kematangan tomat
- Meningkatkan akurasi dan konsistensi dalam sortir buah
- Membantu petani dan distributor dalam quality control

### 🏆 Kategori Klasifikasi

1. **🟠 Mentah** - Tomat yang belum matang (warna hijau kekuningan)
2. **🟢 Muda** - Tomat setengah matang (warna hijau)
3. **🔴 Matang** - Tomat matang sempurna (warna merah)

---

## 🧠 Algoritma dan Metodologi

### 1. **Ekstraksi Fitur GLCM (Gray Level Co-occurrence Matrix)**

GLCM adalah metode analisis tekstur yang menghitung frekuensi kemunculan pasangan pixel dengan intensitas tertentu.

**Langkah-langkah GLCM:**

```
1. Konversi gambar ke Grayscale
   ↓
2. Buat matriks co-occurrence (256×256)
   - Hitung pasangan pixel horizontal (0°)
   - Pixel[i,j] dan Pixel[i,j+1]
   ↓
3. Normalisasi matriks
   - Ubah frekuensi → probabilitas
   ↓
4. Hitung 4 fitur statistik:
   • Contrast: Perbedaan intensitas lokal
   • Dissimilarity: Variasi intensitas
   • Homogeneity: Keseragaman tekstur
   • Energy: Uniformitas distribusi
```

**Rumus Fitur GLCM:**

- **Contrast**: `Σ Σ P(i,j) × (i-j)²`
- **Dissimilarity**: `Σ Σ P(i,j) × |i-j|`
- **Homogeneity**: `Σ Σ P(i,j) / (1 + (i-j)²)`
- **Energy**: `Σ Σ P(i,j)²`

### 2. **Ekstraksi Fitur Color Moment**

Color Moment merepresentasikan distribusi warna menggunakan 3 momen statistik untuk setiap channel warna.

**Mengapa HSV lebih baik dari RGB?**

| Aspek              | HSV                                         | RGB                         |
| ------------------ | ------------------------------------------- | --------------------------- |
| **Hue (H)**        | Merepresentasikan warna murni (hijau→merah) | Tercampur dengan brightness |
| **Saturation (S)** | Intensitas/kejenuhan warna                  | Tidak ada pemisahan         |
| **Value (V)**      | Kecerahan terpisah dari warna               | Tercampur dengan warna      |
| **Untuk Tomat**    | ✅ Ideal untuk deteksi kematangan           | ❌ Terpengaruh pencahayaan  |

**3 Momen Statistik per Channel:**

1. **Mean (μ)**: Rata-rata nilai warna

   ```
   μ = (1/N) Σ pixel_value
   ```

   - Hue rendah = Merah (matang)
   - Hue tinggi = Hijau (muda)

2. **Standard Deviation (σ)**: Variasi warna

   ```
   σ = √[(1/N) Σ (pixel_value - μ)²]
   ```

   - Tinggi = Warna bervariasi
   - Rendah = Warna seragam

3. **Skewness**: Kemencengan distribusi
   ```
   Skewness = (1/N) Σ [(pixel_value - μ) / σ]³
   ```
   - Positif = Condong kanan
   - Negatif = Condong kiri

**Total Fitur**: 9 fitur (3 momen × 3 channel HSV)

### 3. **Klasifikasi dengan SVM (Support Vector Machine)**

SVM mencari **hyperplane** (bidang pemisah) optimal yang memisahkan kelas-kelas data dengan margin maksimal.

**Parameter SVM yang Digunakan:**

- **Kernel**: RBF (Radial Basis Function)

  ```
  K(x, x') = exp(-γ ||x - x'||²)
  ```

  - Cocok untuk data non-linear
  - Dapat menangani pola kompleks

- **C = 1.0**: Parameter regularisasi

  - Trade-off antara margin dan error
  - Mengontrol overfitting

- **Gamma = 'scale'**: Parameter kernel
  ```
  gamma = 1 / (n_features × X.var())
  ```
  - Menentukan jangkauan pengaruh satu data

**Proses Training:**

```
Dataset (100%)
    ↓
Split Data
    ├─→ Training (75%) → Fit SVM Model
    └─→ Testing (25%) → Evaluasi Akurasi
                ↓
        Confusion Matrix
        Classification Report
```

---

## 📊 Arsitektur Sistem

```mermaid
graph TD
    A[Input: Gambar Tomat] --> B[Preprocessing]
    B --> C[Resize 128x128]
    C --> D[Ekstraksi Fitur]

    D --> E[GLCM Features]
    D --> F[Color Moment HSV/RGB]

    E --> G[4 Fitur Tekstur]
    F --> H[9 Fitur Warna]

    G --> I[Gabung: 13 Fitur]
    H --> I

    I --> J[SVM Classifier]
    J --> K{Prediksi}

    K --> L[🟠 Mentah]
    K --> M[🟢 Muda]
    K --> N[🔴 Matang]

    style A fill:#e3f2fd
    style K fill:#fff3e0
    style L fill:#fff9c4
    style M fill:#c8e6c9
    style N fill:#ffcdd2
```

---

## 🚀 Instalasi dan Penggunaan

### Persyaratan Sistem

- Python 3.8 atau lebih tinggi
- Windows/Linux/MacOS

### Langkah Instalasi

1. **Clone atau Download Project**

   ```bash
   cd project-akhir
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Library yang Dibutuhkan:**

   - `opencv-python` - Pengolahan citra
   - `numpy` - Komputasi numerik
   - `scikit-learn` - Machine learning (SVM)
   - `Pillow` - Manipulasi gambar
   - `matplotlib` - Visualisasi
   - `seaborn` - Visualisasi statistik

3. **Struktur Folder Dataset**

   ```
   dataset/
   ├── mentah/     # Gambar tomat mentah
   ├── muda/       # Gambar tomat muda
   └── matang/     # Gambar tomat matang
   ```

4. **Jalankan Aplikasi**
   ```bash
   python main.py
   ```

### 📖 Cara Penggunaan

#### **Mode 1: Auto-Training (Recommended)**

Program akan otomatis melatih model saat startup jika folder `dataset` tersedia.

1. Jalankan `python main.py`
2. Tunggu proses training selesai
3. ✨ **Model otomatis tersimpan** ke folder `models/` dengan timestamp
4. 📊 **Akurasi ditampilkan** di badge header dengan color-coding
5. Klik **"🖼️ Pilih Gambar Tomat"**
6. Klik **"🔍 Proses Klasifikasi"**
7. 🎯 Lihat **Confidence Score** di hasil klasifikasi

#### **Mode 2: Manual Training**

1. Klik **"📚 Load Dataset & Training"**
2. Pilih folder dataset
3. Pilih metode: **HSV + GLCM** atau **RGB + GLCM**
4. Tunggu training selesai
5. 📊 **Akurasi badge** di header akan update otomatis
6. ✅ **Model auto-saved** dengan timestamp ke `models/tomato_model_[METHOD]_[TIMESTAMP].pkl`
7. Lihat hasil akurasi dan confusion matrix

#### **Mode 3: Load Model Tersimpan**

1. Klik **"📂 Load Model"**
2. Pilih file model (.pkl)
3. Langsung gunakan untuk klasifikasi

---

## 🎨 Fitur Aplikasi

### 1. **🎨 UI Modern & Premium**

- ✨ **Dark Theme** - Desain modern dengan skema warna gelap yang elegan
- 🏆 **Accuracy Badge** - Display akurasi prominently di header dengan color-coding:
  - 🟢 **Hijau** (≥85%): Akurasi sangat baik
  - 🟡 **Kuning** (<85%): Akurasi perlu improvement
- 🎯 **Gradient Backgrounds** - Background dengan gradient untuk tampilan premium
- 💎 **Raised Buttons** - Tombol dengan shadow effect dan hover states

### 2. **Panel Kontrol**

- ✅ Pilihan metode ekstraksi (HSV/RGB)
- ✅ Training otomatis dan manual
- ✅ **Auto-Save Model** - Model otomatis tersimpan setelah training
- ✅ Save/Load model dengan default path ke folder `models/`
- ✅ Reset aplikasi

### 3. **Visualisasi Hasil**

- 📊 Confusion Matrix (heatmap)
- 📈 Histogram RGB dan HSV
- 🖼️ Preview gambar input
- 📋 Classification report lengkap
- 🎯 **Confidence Score** - Persentase kepercayaan prediksi (baru!)

### 4. **Analisis Mendalam**

Setelah klasifikasi, sistem menampilkan:

- Citra RGB, HSV, dan Grayscale
- Histogram distribusi warna
- Interpretasi hasil berdasarkan nilai mean Hue/RGB

**Contoh Output Klasifikasi:**

```
=== HASIL KLASIFIKASI ===

Gambar: tomat_test.jpg
Metode: HSV + GLCM

HASIL PREDIKSI: Matang
CONFIDENCE: 94.23%

🔴 MATANG: Red channel tinggi (185.3),
           Hue rendah (12.5) → warna merah dominan

🟢 MUDA: Green channel tinggi (142.7),
         Hue tinggi (78.2) → warna hijau dominan

🟠 MENTAH: Red-Green seimbang (R:125.4, G:118.9),
           Saturation sedang (95.3)
```

---

## 📈 Evaluasi Model

### Metrik Evaluasi

1. **Accuracy**: Persentase prediksi benar

   ```
   Accuracy = (TP + TN) / Total Data
   ```

2. **Confusion Matrix**: Tabel prediksi vs aktual

   ```
              Predicted
              M  Mu Ma
   Actual  M  [TP FP FP]
           Mu [FN TP FP]
           Ma [FN FN TP]
   ```

3. **Classification Report**:
   - **Precision**: Ketepatan prediksi positif
   - **Recall**: Kemampuan mendeteksi kelas
   - **F1-Score**: Harmonic mean precision & recall

---

## 🔬 Hasil Eksperimen

### Perbandingan Metode

| Metode         | Akurasi | Kelebihan                                                 | Kekurangan                 |
| -------------- | ------- | --------------------------------------------------------- | -------------------------- |
| **HSV + GLCM** | ~85-95% | ✅ Robust terhadap pencahayaan<br>✅ Deteksi warna akurat | ⚠️ Sensitif terhadap noise |
| **RGB + GLCM** | ~75-85% | ✅ Sederhana<br>✅ Cepat                                  | ❌ Terpengaruh pencahayaan |

### Rekomendasi

🏆 **HSV + GLCM** adalah pilihan terbaik untuk klasifikasi kematangan tomat karena:

- Memisahkan informasi warna dari kecerahan
- Lebih robust terhadap variasi pencahayaan
- Akurasi lebih tinggi dan konsisten

---

## 📁 Struktur File

```
project-akhir/
│
├── main.py                 # File utama aplikasi (Enhanced!)
├── requirements.txt        # Dependencies Python
├── installation.txt        # Panduan instalasi singkat
├── README.md              # Dokumentasi lengkap (file ini)
├── LAPORAN.md             # Laporan teknis
│
├── dataset/               # Dataset training
│   ├── mentah/            # Tomat mentah (hijau kekuningan)
│   ├── muda/              # Tomat setengah matang (hijau)
│   └── matang/            # Tomat matang (merah)
│
├── test_images/           # Gambar untuk testing
│
└── models/                # Model tersimpan (.pkl)
    └── tomato_model_HSV_[timestamp].pkl  # Auto-saved models
```

---

## 🎓 Pengembang

**Luthfi Shidqi H**  
Mata Kuliah: Pengolahan Citra Digital  
Semester 5

---

## 📚 Referensi

1. Haralick, R. M., et al. (1973). "Textural Features for Image Classification"
2. Stricker, M. A., & Orengo, M. (1995). "Similarity of color images"
3. Cortes, C., & Vapnik, V. (1995). "Support-vector networks"
4. OpenCV Documentation: https://docs.opencv.org/
5. Scikit-learn Documentation: https://scikit-learn.org/

---

## 📄 Lisensi

Project ini dibuat untuk keperluan akademik dan pembelajaran.

---

## 🤝 Kontribusi

Untuk pertanyaan atau saran perbaikan, silakan hubungi pengembang.

---

**Terakhir diupdate**: Januari 2026
