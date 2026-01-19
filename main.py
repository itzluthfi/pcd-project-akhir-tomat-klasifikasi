import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class TomatoClassifier:
    def __init__(self):
        self.model = None
        self.feature_method = 'COMBINED'  # Default: HSV + RGB
        self.accuracy = None  # Store accuracy
        self.confusion_matrix = None  # Store confusion matrix
        
    def extract_glcm_features(self, image):
        """
        Ekstraksi fitur GLCM (Gray Level Co-occurrence Matrix)
        
        GLCM adalah metode untuk menganalisis tekstur gambar dengan menghitung
        seberapa sering pasangan pixel dengan intensitas tertentu muncul.
        
        Langkah-langkah:
        1. Konversi gambar ke grayscale
        2. Buat matriks co-occurrence (GLCM)
        3. Normalisasi matriks
        4. Hitung fitur statistik (contrast, dissimilarity, homogeneity, energy)
        """
        # LANGKAH 1: Konversi gambar BGR ke Grayscale
        # Grayscale diperlukan karena GLCM bekerja dengan intensitas pixel (0-255)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # LANGKAH 2: Inisialisasi matriks GLCM (256x256)
        # Matriks ini akan menyimpan frekuensi kemunculan pasangan pixel
        glcm = np.zeros((256, 256))
        rows, cols = gray.shape
        
        # Hitung co-occurrence matrix dengan arah horizontal (0°)
        # Untuk setiap pixel, kita lihat pixel di sebelah kanannya
        for i in range(rows-1):
            for j in range(cols-1):
                # Ambil nilai pixel saat ini dan pixel di sebelah kanannya
                current_pixel = gray[i,j]
                next_pixel = gray[i,j+1]
                # Tambahkan hitungan pada matriks GLCM
                glcm[current_pixel, next_pixel] += 1
        
        # LANGKAH 3: Normalisasi GLCM
        # Ubah frekuensi menjadi probabilitas (jumlah total = 1)
        glcm = glcm / glcm.sum()
        
        # LANGKAH 4: Hitung fitur statistik dari GLCM
        contrast = 0      # Mengukur perbedaan intensitas lokal
        dissimilarity = 0 # Mengukur variasi intensitas
        homogeneity = 0   # Mengukur keseragaman tekstur
        energy = 0        # Mengukur keseragaman distribusi
        
        for i in range(256):
            for j in range(256):
                # Contrast: Mengukur kontras lokal (perbedaan kuadrat)
                contrast += glcm[i,j] * (i-j)**2
                
                # Dissimilarity: Perbedaan absolut intensitas
                dissimilarity += glcm[i,j] * abs(i-j)
                
                # Homogeneity: Kedekatan distribusi GLCM ke diagonal
                # Nilai tinggi = tekstur homogen
                homogeneity += glcm[i,j] / (1 + (i-j)**2)
                
                # Energy: Jumlah kuadrat elemen (uniformity)
                # Nilai tinggi = tekstur teratur
                energy += glcm[i,j]**2
        
        # Return 4 fitur GLCM sebagai array
        return [contrast, dissimilarity, homogeneity, energy]
    
    def extract_color_moment(self, image, color_space='HSV'):
        """
        Ekstraksi fitur Color Moment
        
        Color Moment adalah metode untuk merepresentasikan distribusi warna dalam gambar.
        Menggunakan 3 momen statistik: Mean, Standard Deviation, dan Skewness.
        
        Untuk klasifikasi tomat:
        - HSV lebih baik karena memisahkan warna (Hue) dari kecerahan (Value)
        - Hue menunjukkan tingkat kematangan (hijau -> kuning -> merah)
        - Saturation menunjukkan intensitas warna
        - Value menunjukkan kecerahan
        """
        # LANGKAH 1: Konversi color space
        if color_space == 'HSV':
            # HSV (Hue, Saturation, Value) lebih baik untuk deteksi warna
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            # RGB (Red, Green, Blue) - warna asli
            converted = image
        
        features = []
        
        # LANGKAH 2: Ekstraksi momen untuk setiap channel (H, S, V atau R, G, B)
        for channel in range(3):
            # Ambil data channel dan ubah menjadi array 1D
            channel_data = converted[:,:,channel].flatten()
            
            # MOMEN 1: Mean (Rata-rata)
            # Menunjukkan nilai warna dominan
            # Contoh: Mean Hue tinggi = warna hijau, rendah = merah
            mean = np.mean(channel_data)
            
            # MOMEN 2: Standard Deviation (Deviasi Standar)
            # Mengukur variasi/sebaran warna
            # Nilai tinggi = warna bervariasi, rendah = warna seragam
            std = np.std(channel_data)
            
            # MOMEN 3: Skewness (Kemencengan)
            # Mengukur asimetri distribusi warna
            # Positif = condong ke kanan, Negatif = condong ke kiri
            skewness = np.mean(((channel_data - mean) / std) ** 3) if std != 0 else 0
            
            # Tambahkan 3 fitur per channel (total 9 fitur untuk 3 channel)
            features.extend([mean, std, skewness])
        
        # Return 9 fitur Color Moment (3 momen × 3 channel)
        return features
    
    def extract_features(self, image_path, color_space='COMBINED'):
        """
        Gabungan ekstraksi fitur GLCM dan Color Moment (HSV + RGB)
        
        Total fitur: 22
        - GLCM: 4 fitur (texture)
        - HSV Color Moment: 9 fitur (3 momen × 3 channel)
        - RGB Color Moment: 9 fitur (3 momen × 3 channel)
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Resize untuk konsistensi
        image = cv2.resize(image, (128, 128))
        
        # Ekstraksi fitur GLCM (texture)
        glcm_features = self.extract_glcm_features(image)
        
        # Ekstraksi Color Moment dari HSV
        hsv_features = self.extract_color_moment(image, 'HSV')
        
        # Ekstraksi Color Moment dari RGB
        rgb_features = self.extract_color_moment(image, 'RGB')
        
        # Gabungkan semua fitur: GLCM + HSV + RGB
        all_features = glcm_features + hsv_features + rgb_features
        
        return all_features
    
    def load_dataset(self, dataset_path, color_space='HSV'):
        """Load dataset dari folder"""
        features = []
        labels = []
        
        categories = ['mentah', 'muda', 'matang']
        
        # Daftar ekstensi gambar yang didukung (termasuk yang tidak punya ekstensi)
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.avif')
        
        for label, category in enumerate(categories):
            category_path = os.path.join(dataset_path, category)
            if not os.path.exists(category_path):
                print(f"Warning: Folder '{category}' tidak ditemukan!")
                continue
            
            file_count = 0
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                
                # Skip jika bukan file
                if not os.path.isfile(file_path):
                    continue
                
                # Cek ekstensi file atau coba baca langsung
                filename_lower = filename.lower()
                is_valid = filename_lower.endswith(valid_extensions)
                
                # Jika tidak ada ekstensi atau ekstensi tidak dikenali, coba baca sebagai gambar
                if not is_valid and '.' not in filename:
                    is_valid = True
                
                if is_valid:
                    try:
                        feature = self.extract_features(file_path, color_space)
                        
                        if feature is not None:
                            features.append(feature)
                            labels.append(label)
                            file_count += 1
                    except Exception as e:
                        print(f"Error reading {filename}: {str(e)}")
                        continue
            
            print(f"Loaded {file_count} images dari folder '{category}'")
        
        return np.array(features), np.array(labels)
    
    def train(self, dataset_path, color_space='HSV', test_size=0.15):
        """
        Training model menggunakan Support Vector Machine (SVM)
        
        SVM adalah algoritma machine learning untuk klasifikasi yang bekerja dengan
        mencari hyperplane (bidang pemisah) terbaik antara kelas-kelas data.
        
        Parameter:
        - kernel='rbf': Radial Basis Function untuk data non-linear
        - C=1.0: Parameter regularisasi (trade-off antara margin dan error)
        - gamma='scale': Parameter kernel (seberapa jauh pengaruh satu data)
        """
        print(f"Loading dataset dengan metode {color_space}...")
        
        # LANGKAH 1: Load dan ekstraksi fitur dari dataset
        X, y = self.load_dataset(dataset_path, color_space)
        
        if len(X) == 0:
            raise ValueError("Dataset kosong! Pastikan struktur folder benar.")
        
        print(f"Total data: {len(X)}")
        
        # LANGKAH 2: Split data menjadi training dan testing
        # - 85% untuk training (melatih model)
        # - 15% untuk testing (menguji akurasi)
        # - stratify=y: memastikan proporsi kelas seimbang di train dan test
        # - test_size dikurangi karena dataset kecil
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Data training: {len(X_train)}")
        print(f"Data testing: {len(X_test)}")
        
        # LANGKAH 3: Training model SVM
        print("Training SVM...")
        # Inisialisasi SVM dengan kernel RBF (Radial Basis Function)
        # RBF cocok untuk data yang tidak linear separable
        self.model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        
        # Fit model dengan data training
        # Model akan belajar pola dari fitur (X_train) dan label (y_train)
        self.model.fit(X_train, y_train)
        
        # LANGKAH 4: Evaluasi model dengan data testing
        # Prediksi label untuk data test
        y_pred = self.model.predict(X_test)
        
        # Hitung metrik evaluasi
        # Accuracy: Persentase prediksi yang benar
        accuracy = accuracy_score(y_test, y_pred)
        
        # Confusion Matrix: Tabel yang menunjukkan prediksi benar dan salah
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification Report: Precision, Recall, F1-Score per kelas
        report = classification_report(y_test, y_pred, 
                                       target_names=['Mentah', 'Muda', 'Matang'])
        
        # Tampilkan hasil evaluasi
        print(f"\n{'='*60}")
        print(f"HASIL TRAINING MODEL")
        print(f"{'='*60}")
        print(f"\n🎯 AKURASI KESELURUHAN: {accuracy*100:.2f}%\n")
        
        # Format Confusion Matrix yang mudah dibaca
        print("📊 CONFUSION MATRIX (Matriks Kebingungan):")
        print("="*60)
        print(f"{'':>15} {'Prediksi →':^45}")
        print(f"{'Aktual ↓':>15} {'Mentah':>15} {'Muda':>15} {'Matang':>15}")
        print("-"*60)
        categories = ['Mentah', 'Muda', 'Matang']
        for i, category in enumerate(categories):
            print(f"{category:>15}", end="")
            for j in range(len(categories)):
                print(f"{cm[i][j]:>15}", end="")
            print()
        print("="*60)
        
        # Interpretasi Confusion Matrix
        print("\n📖 INTERPRETASI:")
        for i, category in enumerate(categories):
            total_actual = cm[i].sum()
            correct = cm[i][i]
            if total_actual > 0:
                class_acc = (correct / total_actual) * 100
                print(f"\n   {category}:")
                print(f"   ✓ Benar diprediksi: {correct}/{total_actual} ({class_acc:.1f}%)")
                
                # Tampilkan kesalahan prediksi
                for j, pred_cat in enumerate(categories):
                    if i != j and cm[i][j] > 0:
                        print(f"   ✗ Salah diprediksi sebagai {pred_cat}: {cm[i][j]}")
        
        print("\n" + "="*60)
        print("\n📋 CLASSIFICATION REPORT:")
        print(report)
        print("="*60)
        
        # Simpan metode yang digunakan
        self.feature_method = color_space
        self.accuracy = accuracy  # Simpan accuracy
        self.confusion_matrix = cm  # Simpan confusion matrix
        
        return accuracy, cm, report
    
    def predict(self, image_path):
        """Prediksi gambar baru dengan confidence score"""
        if self.model is None:
            raise ValueError("Model belum ditraining!")
        
        feature = self.extract_features(image_path, self.feature_method)
        if feature is None:
            return None, None, None
        
        prediction = self.model.predict([feature])[0]
        decision = self.model.decision_function([feature])[0]
        
        # Hitung confidence score menggunakan softmax pada decision values
        exp_scores = np.exp(decision - np.max(decision))
        probabilities = exp_scores / np.sum(exp_scores)
        confidence = probabilities[prediction] * 100  # Dalam persen
        
        categories = ['Mentah', 'Muda', 'Matang']
        return categories[prediction], prediction, confidence
    
    def save_model(self, filepath):
        """Simpan model dengan accuracy dan confusion matrix"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model, 
                'method': self.feature_method,
                'accuracy': self.accuracy,
                'confusion_matrix': self.confusion_matrix
            }, f)
    
    def load_model(self, filepath):
        """Load model dengan accuracy dan confusion matrix"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_method = data['method']
            self.accuracy = data.get('accuracy', None)
            self.confusion_matrix = data.get('confusion_matrix', None)


class TomatoClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Klasifikasi Kematangan Buah Tomat - Luthfi Shidqi H")
        self.root.geometry("1400x750")
        
        self.classifier = TomatoClassifier()
        self.current_image_path = None
        self.dataset_path = "dataset_augmented"  # Path ke folder dataset AUGMENTED (100+ gambar per kelas)
        # Untuk pakai dataset lama: gunakan "dataset"
        self.accuracy_label = None  # Label untuk menampilkan akurasi
        
        self.setup_gui()
        
        # Auto-load dataset saat program start
        self.auto_load_dataset()
    
    def setup_gui(self):
        """Setup GUI dengan desain modern dan profesional"""
        # Konfigurasi warna tema modern dengan gradient
        self.root.configure(bg='#1a1a2e')
        
        # ========== HEADER dengan Accuracy Badge ==========
        header_frame = tk.Frame(self.root, bg='#16213e', height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, 
                              text="🍅 Klasifikasi Kematangan Tomat",
                              font=("Segoe UI", 18, "bold"),
                              bg='#16213e',
                              fg='#e94560')
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        # Accuracy Badge (akan diupdate setelah training)
        self.accuracy_label = tk.Label(header_frame,
                                       text="Akurasi: ---%",
                                       font=("Segoe UI", 14, "bold"),
                                       bg='#0f3460',
                                       fg='#00ff88',
                                       padx=20,
                                       pady=10,
                                       relief=tk.RAISED,
                                       borderwidth=3)
        self.accuracy_label.pack(side=tk.RIGHT, padx=20, pady=15)
        
        # Frame utama dengan background gradient-like
        main_frame = tk.Frame(self.root, padx=20, pady=20, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ========== FRAME KIRI - PANEL KONTROL ==========
        left_frame = tk.LabelFrame(main_frame, text="🎛️ Panel Kontrol", 
                                   padx=20, pady=20, 
                                   font=("Segoe UI", 12, "bold"),
                                   bg='#16213e',
                                   fg='#00d9ff',
                                   relief=tk.GROOVE,
                                   borderwidth=3)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0,15))
        
        # Header metode ekstraksi
        method_header = tk.Label(left_frame, 
                                text="Metode Ekstraksi:", 
                                font=("Segoe UI", 11, "bold"),
                                bg='#16213e',
                                fg='#ffffff')
        method_header.pack(anchor=tk.W, pady=(5,10))
        
        # Info metode (tidak lagi radio button karena fixed ke COMBINED)
        method_info = tk.Label(left_frame,
                              text="✨ HSV + RGB + GLCM\n ",
                              font=("Segoe UI", 9),
                              bg='#16213e',
                              fg='#00ff88',
                              justify=tk.LEFT)
        method_info.pack(anchor=tk.W, pady=(0,15))
        
        # Separator
        separator = tk.Frame(left_frame, height=2, bg='#0f3460')
        separator.pack(fill=tk.X, pady=12)
        
        # Tombol training dengan ikon dan shadow effect
        btn_train = tk.Button(left_frame, 
                             text="📚 Load Dataset & Training", 
                             command=self.train_model, 
                             width=26, 
                             height=2,
                             font=("Segoe UI", 10, "bold"),
                             bg="#00b894", 
                             fg="white",
                             activebackground="#00a383",
                             activeforeground="white",
                             relief=tk.RAISED,
                             cursor="hand2",
                             borderwidth=2)
        btn_train.pack(pady=8)
        
        # Tombol pilih gambar
        btn_load = tk.Button(left_frame, 
                            text="🖼️ Pilih Gambar Tomat", 
                            command=self.load_image, 
                            width=26, 
                            height=2,
                            font=("Segoe UI", 10, "bold"),
                            bg="#0984e3", 
                            fg="white",
                            activebackground="#0770c7",
                            activeforeground="white",
                            relief=tk.RAISED,
                            cursor="hand2",
                            borderwidth=2)
        btn_load.pack(pady=8)
        
        # Tombol klasifikasi
        btn_classify = tk.Button(left_frame, 
                                text="🔍 Proses Klasifikasi", 
                                command=self.classify_image, 
                                width=26, 
                                height=2,
                                font=("Segoe UI", 10, "bold"),
                                bg="#fd79a8", 
                                fg="white",
                                activebackground="#e66b8f",
                                activeforeground="white",
                                relief=tk.RAISED,
                                cursor="hand2",
                                borderwidth=2)
        btn_classify.pack(pady=8)
        
        # Separator
        separator2 = tk.Frame(left_frame, height=2, bg='#0f3460')
        separator2.pack(fill=tk.X, pady=12)
        
        # Tombol save model
        btn_save = tk.Button(left_frame, 
                            text="💾 Save Model", 
                            command=self.save_model, 
                            width=26, 
                            height=1,
                            font=("Segoe UI", 9),
                            bg="#6c5ce7", 
                            fg="white",
                            activebackground="#5f4dcd",
                            activeforeground="white",
                            relief=tk.RAISED,
                            cursor="hand2",
                            borderwidth=2)
        btn_save.pack(pady=5)
        
        # Tombol load model
        btn_load_model = tk.Button(left_frame, 
                                   text="📂 Load Model", 
                                   command=self.load_model, 
                                   width=26, 
                                   height=1,
                                   font=("Segoe UI", 9),
                                   bg="#6c5ce7", 
                                   fg="white",
                                   activebackground="#5f4dcd",
                                   activeforeground="white",
                                   relief=tk.RAISED,
                                   cursor="hand2",
                                   borderwidth=2)
        btn_load_model.pack(pady=5)
        
        # Tombol reset
        btn_reset = tk.Button(left_frame, 
                             text="🔄 RESET", 
                             command=self.reset, 
                             width=26, 
                             height=1,
                             font=("Segoe UI", 9, "bold"),
                             bg="#d63031", 
                             fg="white",
                             activebackground="#c0392b",
                             activeforeground="white",
                             relief=tk.RAISED,
                             cursor="hand2",
                             borderwidth=2)
        btn_reset.pack(pady=5)
        
        # ========== FRAME KANAN - HASIL ==========
        right_frame = tk.Frame(main_frame, bg='#1a1a2e')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Frame gambar dengan styling modern
        image_frame = tk.LabelFrame(right_frame, 
                                    text="🖼️ Gambar Input", 
                                    padx=15, 
                                    pady=15,
                                    font=("Segoe UI", 12, "bold"),
                                    bg='#16213e',
                                    fg='#00d9ff',
                                    relief=tk.GROOVE,
                                    borderwidth=3)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0,15))
        
        # Label gambar dengan border
        image_container = tk.Frame(image_frame, bg='#0f3460', relief=tk.SUNKEN, borderwidth=3)
        image_container.pack(expand=True, fill=tk.BOTH, padx=8, pady=8)
        
        self.image_label = tk.Label(image_container, 
                                    text="📷\n\nBelum ada gambar\n\nKlik 'Pilih Gambar Tomat' untuk memulai", 
                                    bg="#0f3460",
                                    fg="#74b9ff",
                                    font=("Segoe UI", 11),
                                    width=40, 
                                    height=15)
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        # Frame hasil dengan styling
        result_frame = tk.LabelFrame(right_frame, 
                                     text="📊 Hasil Klasifikasi", 
                                     padx=15, 
                                     pady=15,
                                     font=("Segoe UI", 12, "bold"),
                                     bg='#16213e',
                                     fg='#00d9ff',
                                     relief=tk.GROOVE,
                                     borderwidth=3)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text widget dengan scrollbar
        text_container = tk.Frame(result_frame, bg='#16213e')
        text_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(text_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(text_container, 
                                   height=10, 
                                   font=("Consolas", 10),
                                   bg='#0f3460',
                                   fg='#dfe6e9',
                                   relief=tk.FLAT,
                                   borderwidth=0,
                                   yscrollcommand=scrollbar.set,
                                   wrap=tk.WORD,
                                   insertbackground='#00ff88')
        self.result_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)
    
    def train_model(self):
        dataset_path = filedialog.askdirectory(title="Pilih Folder Dataset")
        if not dataset_path:
            return
        
        try:
            method = 'COMBINED'  # Fixed to COMBINED (HSV + RGB + GLCM)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Training dengan metode {method}...\n")
            self.root.update()
            
            accuracy, cm, report = self.classifier.train(dataset_path, method)
            
            # Update accuracy badge
            self.accuracy_label.config(
                text=f"Akurasi: {accuracy*100:.2f}%",
                bg='#00b894' if accuracy >= 0.85 else '#fdcb6e',
                fg='white'
            )
            
            result = f"\n{'='*50}\n"
            result += f"HASIL TRAINING\n"
            result += f"{'='*50}\n"
            result += f"Metode: HSV + RGB + GLCM \n"
            result += f"🎯 AKURASI: {accuracy*100:.2f}%\n\n"
            
            # Format Confusion Matrix yang readable
            result += "📊 CONFUSION MATRIX:\n"
            result += "="*50 + "\n"
            result += f"{'':>12} {'Prediksi →':^38}\n"
            result += f"{'Aktual ↓':>12} {'Mentah':>12} {'Muda':>12} {'Matang':>12}\n"
            result += "-"*50 + "\n"
            categories = ['Mentah', 'Muda', 'Matang']
            for i, category in enumerate(categories):
                result += f"{category:>12}"
                for j in range(len(categories)):
                    result += f"{cm[i][j]:>12}"
                result += "\n"
            result += "="*50 + "\n"
            
            # Interpretasi per kelas
            result += "\n📖 INTERPRETASI PER KELAS:\n"
            for i, category in enumerate(categories):
                total_actual = cm[i].sum()
                correct = cm[i][i]
                if total_actual > 0:
                    class_acc = (correct / total_actual) * 100
                    result += f"\n  {category}:\n"
                    result += f"  ✓ Benar: {correct}/{total_actual} ({class_acc:.1f}%)\n"
                    
                    for j, pred_cat in enumerate(categories):
                        if i != j and cm[i][j] > 0:
                            result += f"  ✗ Salah sbg {pred_cat}: {cm[i][j]}\n"
            
            result += "\n" + "="*50 + "\n"
            result += "Classification Report:\n"
            result += report
            
            self.result_text.insert(tk.END, result)
            
            # Tampilkan confusion matrix
            self.show_confusion_matrix(cm)
            
            # Auto-save model
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"tomato_model_{method}_{timestamp}.pkl"
            model_path = os.path.join("models", model_filename)
            
            # Buat folder models jika belum ada
            os.makedirs("models", exist_ok=True)
            
            self.classifier.save_model(model_path)
            
            success_msg = f"Training selesai!\n"
            success_msg += f"Akurasi: {accuracy*100:.2f}%\n"
            success_msg += f"Model tersimpan: {model_filename}"
            
            messagebox.showinfo("✅ Sukses", success_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Training gagal: {str(e)}")
    
    def show_confusion_matrix(self, cm):
        """Tampilkan confusion matrix sebagai diagram bar dengan persentase"""
        categories = ['Mentah', 'Muda', 'Matang']
        
        # Hitung persentase untuk setiap kelas
        percentages = []
        class_names = []
        colors_data = []
        
        for i, category in enumerate(categories):
            total_actual = cm[i].sum()
            if total_actual > 0:
                correct = cm[i][i]
                percentage = (correct / total_actual) * 100
                percentages.append(percentage)
                class_names.append(category)
                
                # Warna berdasarkan persentase
                if percentage >= 90:
                    colors_data.append('#27AE60')  # Hijau tua
                elif percentage >= 70:
                    colors_data.append('#2ECC71')  # Hijau
                elif percentage >= 50:
                    colors_data.append('#F39C12')  # Orange
                else:
                    colors_data.append('#E74C3C')  # Merah
        
        # Hitung akurasi overall
        total_correct = np.trace(cm)
        total_samples = np.sum(cm)
        overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        
        # Create figure dengan 2 subplot
        fig = plt.figure(figsize=(16, 7), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
        
        # Subplot 1: Bar chart persentase per kelas
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(class_names, percentages, color=colors_data, edgecolor='black', linewidth=2, alpha=0.85)
        
        # Tambahkan label persentase di atas bar
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=16, fontweight='bold')
            
            # Tambahkan jumlah benar/total di dalam bar
            total_actual = cm[i].sum()
            correct = cm[i][i]
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{correct}/{int(total_actual)}',
                    ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        ax1.set_ylabel('Akurasi (%)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Kelas', fontsize=14, fontweight='bold')
        ax1.set_title('📊 Akurasi Per Kelas', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
        ax1.set_facecolor('#f9f9f9')
        ax1.tick_params(labelsize=12)
        
        # Subplot 2: Pie chart akurasi overall dan detail
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Data untuk pie chart
        pie_data = [overall_accuracy, 100 - overall_accuracy]
        pie_labels = [f'Benar\n{overall_accuracy:.1f}%', f'Salah\n{100-overall_accuracy:.1f}%']
        pie_colors = ['#27AE60', '#E74C3C']
        
        wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, colors=pie_colors, 
                                            autopct='', startangle=90, 
                                            textprops={'fontsize': 14, 'fontweight': 'bold'},
                                            wedgeprops={'edgecolor': 'black', 'linewidth': 2})
        
        ax2.set_title('🎯 Akurasi Keseluruhan', fontsize=16, fontweight='bold', pad=20)
        
        # Tambahkan teks di tengah pie chart
        centre_circle = plt.Circle((0, 0), 0.70, fc='white', edgecolor='black', linewidth=2)
        ax2.add_artist(centre_circle)
        ax2.text(0, 0, f'{overall_accuracy:.1f}%', ha='center', va='center', 
                fontsize=32, fontweight='bold', color='#27AE60')
        ax2.text(0, -0.25, 'Akurasi', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='#555')
        
        # Suptitle dengan info tambahan
        fig.suptitle(f'Evaluasi Model - Total Sampel: {total_samples} | Benar: {total_correct} | Salah: {total_samples - total_correct}',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.show()
    
    def load_image(self):
        filepath = filedialog.askopenfilename(
            title="Pilih Gambar Tomat",
            filetypes=[
                ("All Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.avif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("WebP files", "*.webp"),
                ("AVIF files", "*.avif"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            self.current_image_path = filepath
            
            # Tampilkan gambar
            image = Image.open(filepath)
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Gambar loaded: {os.path.basename(filepath)}\n")
            self.result_text.insert(tk.END, "Klik 'Proses Klasifikasi' untuk mengklasifikasi.\n")
    
    def classify_image(self):
        if self.current_image_path is None:
            messagebox.showwarning("Peringatan", "Pilih gambar terlebih dahulu!")
            return
        
        if self.classifier.model is None:
            messagebox.showwarning("Peringatan", "Model belum ditraining!")
            return
        
        try:
            result, label, confidence = self.classifier.predict(self.current_image_path)
            
            self.result_text.delete(1.0, tk.END)
            
            # Header dengan separator
            self.result_text.insert(tk.END, "="*55 + "\n")
            self.result_text.insert(tk.END, "   HASIL KLASIFIKASI KEMATANGAN TOMAT\n")
            self.result_text.insert(tk.END, "="*55 + "\n\n")
            
            # Info gambar
            self.result_text.insert(tk.END, f"🖼️  Gambar: {os.path.basename(self.current_image_path)}\n")
            self.result_text.insert(tk.END, f"⚙️  Metode: HSV + RGB + GLCM \n\n")
            
            # Separator
            self.result_text.insert(tk.END, "-"*55 + "\n\n")
            
            # HASIL PREDIKSI (paling menonjol)
            self.result_text.insert(tk.END, "🎯 PREDIKSI: ", "label_title")
            self.result_text.insert(tk.END, f"{result}\n\n", "result")
            
            # CONFIDENCE SCORE dengan interpretasi
            self.result_text.insert(tk.END, "📊 CONFIDENCE SCORE: ", "conf_title")
            self.result_text.insert(tk.END, f"{confidence:.2f}%\n\n", "confidence")
            
            # Interpretasi confidence
            if confidence >= 90:
                conf_level = "⭐⭐⭐ SANGAT YAKIN"
                conf_desc = "Model sangat percaya diri dengan prediksi ini"
            elif confidence >= 70:
                conf_level = "⭐⭐ CUKUP YAKIN"
                conf_desc = "Model cukup percaya diri dengan prediksi ini"
            elif confidence >= 50:
                conf_level = "⭐ KURANG YAKIN"
                conf_desc = "Model kurang percaya diri, hasil mungkin tidak akurat"
            else:
                conf_level = "⚠️ TIDAK YAKIN"
                conf_desc = "Model sangat tidak yakin, hasil kemungkinan salah"
            
            self.result_text.insert(tk.END, f"📈 Tingkat Keyakinan: {conf_level}\n", "conf_level")
            self.result_text.insert(tk.END, f"   {conf_desc}\n\n", "conf_desc")
            
            # Separator
            self.result_text.insert(tk.END, "="*55 + "\n\n")
            
            # Penjelasan singkat
            self.result_text.insert(tk.END, "📖 Penjelasan:\n", "explanation_title")
            
            if result == "Matang":
                self.result_text.insert(tk.END, "🔴 Tomat ini sudah MATANG (merah)\n")
                self.result_text.insert(tk.END, "   Warna dominan merah, siap dikonsumsi\n")
            elif result == "Muda":
                self.result_text.insert(tk.END, "🟢 Tomat ini masih MUDA (hijau)\n")
                self.result_text.insert(tk.END, "   Warna dominan hijau, belum matang\n")
            else:  # Mentah
                self.result_text.insert(tk.END, "🟠 Tomat ini MENTAH (hijau kekuningan)\n")
                self.result_text.insert(tk.END, "   Warna hijau-kuning, perlu waktu untuk matang\n")
            
            # Styling untuk semua tag
            self.result_text.tag_config("label_title", font=("Arial", 11, "bold"), foreground="#74b9ff")
            self.result_text.tag_config("result", font=("Arial", 20, "bold"), foreground="#00ff88")
            
            self.result_text.tag_config("conf_title", font=("Arial", 11, "bold"), foreground="#74b9ff")
            self.result_text.tag_config("confidence", font=("Arial", 18, "bold"), foreground="#ffeaa7")
            
            self.result_text.tag_config("conf_level", font=("Arial", 10, "bold"), foreground="#fdcb6e")
            self.result_text.tag_config("conf_desc", font=("Arial", 9), foreground="#b2bec3")
            
            self.result_text.tag_config("explanation_title", font=("Arial", 10, "bold"), foreground="#74b9ff")
            
            # Tampilkan visualisasi
            self.show_prediction_visualization(self.current_image_path, result, confidence)
            
        except Exception as e:
            messagebox.showerror("Error", f"Klasifikasi gagal: {str(e)}")
    
    def show_prediction_visualization(self, image_path, prediction, confidence=None):
        """Tampilkan visualisasi hasil prediksi dengan histogram dan confidence"""
        # Load gambar
        img_bgr = cv2.imread(image_path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (300, 300))  # Lebih besar untuk lebih jelas
        
        # Convert ke HSV dan Grayscale
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hsv_resized = cv2.resize(hsv, (300, 300))  # Lebih besar
        
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (300, 300))  # Lebih besar
        
        # Create figure dengan 2 baris - LEBIH BESAR
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
        
        # Baris 1: Tampilan gambar - LEBIH PROMINENT
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_resized)
        ax1.set_title('Citra RGB', fontweight='bold', fontsize=14, color='darkblue', pad=10)
        ax1.axis('off')
        # Tambah border untuk lebih menonjol
        for spine in ax1.spines.values():
            spine.set_edgecolor('darkblue')
            spine.set_linewidth(3)
        ax1.spines['top'].set_visible(True)
        ax1.spines['right'].set_visible(True)
        ax1.spines['bottom'].set_visible(True)
        ax1.spines['left'].set_visible(True)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(hsv_resized)
        ax2.set_title('Citra HSV', fontweight='bold', fontsize=14, color='darkgreen', pad=10)
        ax2.axis('off')
        # Tambah border untuk lebih menonjol
        for spine in ax2.spines.values():
            spine.set_edgecolor('darkgreen')
            spine.set_linewidth(3)
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)
        ax2.spines['bottom'].set_visible(True)
        ax2.spines['left'].set_visible(True)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(gray_resized, cmap='gray')
        ax3.set_title('Citra Grayscale\n(untuk GLCM)', fontweight='bold', fontsize=14, color='darkred', pad=10)
        ax3.axis('off')
        # Tambah border untuk lebih menonjol
        for spine in ax3.spines.values():
            spine.set_edgecolor('darkred')
            spine.set_linewidth(3)
        ax3.spines['top'].set_visible(True)
        ax3.spines['right'].set_visible(True)
        ax3.spines['bottom'].set_visible(True)
        ax3.spines['left'].set_visible(True)
        
        # Baris 2: Histogram RGB - LEBIH TEBAL DAN JELAS
        ax4 = fig.add_subplot(gs[1, :])
        colors = ('#FF0000', '#00FF00', '#0000FF')  # Warna lebih vibrant
        labels = ('Red Channel', 'Green Channel', 'Blue Channel')
        for i, (color, label) in enumerate(zip(colors, labels)):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax4.plot(hist, color=color, label=label, linewidth=3.5, alpha=0.8)  # Lebih tebal
        ax4.set_xlim([0, 256])
        ax4.set_xlabel('Nilai Pixel', fontweight='bold', fontsize=13)
        ax4.set_ylabel('Frekuensi', fontweight='bold', fontsize=13)
        ax4.set_title('📊 Histogram RGB - Distribusi Warna', fontweight='bold', fontsize=15, pad=15)
        ax4.legend(fontsize=12, loc='upper right', framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax4.set_facecolor('#f9f9f9')
        
        # Baris 3: Histogram HSV - WARNA LEBIH MENCOLOK
        ax5 = fig.add_subplot(gs[2, :])
        hsv_colors = ('#FF6B35', '#9B59B6', '#34495E')  # Warna lebih mencolok
        hsv_labels = ('Hue (Warna)', 'Saturation (Kejenuhan)', 'Value (Kecerahan)')
        for i, (color, label) in enumerate(zip(hsv_colors, hsv_labels)):
            hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
            ax5.plot(hist, color=color, label=label, linewidth=3.5, alpha=0.8)  # Lebih tebal
        ax5.set_xlim([0, 256])
        ax5.set_xlabel('Nilai Pixel', fontweight='bold', fontsize=13)
        ax5.set_ylabel('Frekuensi', fontweight='bold', fontsize=13)
        ax5.set_title('📊 Histogram HSV - Analisis Kematangan', fontweight='bold', fontsize=15, pad=15)
        ax5.legend(fontsize=12, loc='upper right', framealpha=0.9)
        ax5.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        ax5.set_facecolor('#f9f9f9')
        
        # Analisis warna untuk penjelasan
        mean_red = np.mean(img[:,:,0])
        mean_green = np.mean(img[:,:,1])
        mean_hue = np.mean(hsv[:,:,0])
        mean_saturation = np.mean(hsv[:,:,1])
        
        # Tentukan interpretasi
        if prediction == 'Matang':
            interpretation = f"🔴 MATANG: Red channel tinggi ({mean_red:.1f}), Hue rendah ({mean_hue:.1f}) menunjukkan warna merah dominan"
        elif prediction == 'Muda':
            interpretation = f"🟢 MUDA: Green channel tinggi ({mean_green:.1f}), Hue tinggi ({mean_hue:.1f}) menunjukkan warna hijau dominan"
        else:  # Mentah
            interpretation = f"🟠 MENTAH: Red-Green seimbang (R:{mean_red:.1f}, G:{mean_green:.1f}), Saturation sedang ({mean_saturation:.1f})"
        
        # Interpretasi confidence for suptitle
        if confidence is not None:
            if confidence >= 90:
                conf_level = "SANGAT YAKIN"
            elif confidence >= 70:
                conf_level = "CUKUP YAKIN"
            elif confidence >= 50:
                conf_level = "KURANG YAKIN"
            else:
                conf_level = "TIDAK YAKIN"
            confidence_str = f" (Confidence: {confidence:.2f}% - {conf_level})"
        else:
            confidence_str = ""
        
        fig.suptitle(f'Hasil Prediksi: {prediction}{confidence_str}\n{interpretation}', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
    def save_model(self):
        if self.classifier.model is None:
            messagebox.showwarning("Peringatan", "Tidak ada model untuk disimpan!")
            return
        
        # Default filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"tomato_model_{self.classifier.feature_method}_{timestamp}.pkl"
        default_path = os.path.join("models", default_filename)
        
        # Buat folder models jika belum ada
        os.makedirs("models", exist_ok=True)
        
        filepath = filedialog.asksaveasfilename(
            initialdir="models",
            initialfile=default_filename,
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")]
        )
        
        if filepath:
            self.classifier.save_model(filepath)
            messagebox.showinfo("✅ Sukses", f"Model berhasil disimpan!\n{os.path.basename(filepath)}")

    
    def load_model(self):
        filepath = filedialog.askopenfilename(
            title="Pilih Model File",
            filetypes=[("Pickle files", "*.pkl")]
        )
        
        if filepath:
            try:
                self.classifier.load_model(filepath)
                
                # Update accuracy badge jika ada data accuracy
                if self.classifier.accuracy is not None:
                    self.accuracy_label.config(
                        text=f"Akurasi: {self.classifier.accuracy*100:.2f}%",
                        bg='#00b894' if self.classifier.accuracy >= 0.85 else '#fdcb6e',
                        fg='white'
                    )
                
                msg = f"Model loaded!\nMetode: {self.classifier.feature_method}"
                if self.classifier.accuracy:
                    msg += f"\nAkurasi: {self.classifier.accuracy*100:.2f}%"
                
                messagebox.showinfo("✅ Sukses", msg)
                self.method_var.set(self.classifier.feature_method)
            except Exception as e:
                messagebox.showerror("Error", f"Gagal load model: {str(e)}")
    
    def reset(self):
        self.current_image_path = None
        self.image_label.config(image="", 
                               text="📷\n\nBelum ada gambar\n\nKlik 'Pilih Gambar Tomat' untuk memulai",
                               bg="#0f3460",
                               fg="#74b9ff")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "✅ Program direset.\n")
    
    def auto_load_dataset(self):
        """Auto-load dataset saat program start"""
        if os.path.exists(self.dataset_path):
            try:
                method = 'COMBINED'  # Fixed to COMBINED
                self.result_text.insert(tk.END, "=== AUTO-LOADING DATASET ===\n")
                self.result_text.insert(tk.END, f"Mencari dataset di folder: {self.dataset_path}\n")
                self.result_text.insert(tk.END, f"Training dengan HSV + RGB + GLCM...\n\n")
                self.root.update()
                
                accuracy, cm, report = self.classifier.train(self.dataset_path, method)
                
                # Update accuracy badge
                self.accuracy_label.config(
                    text=f"Akurasi: {accuracy*100:.2f}%",
                    bg='#00b894' if accuracy >= 0.85 else '#fdcb6e',
                    fg='white'
                )
                
                result = "="*50 + "\n"
                result += "TRAINING SELESAI\n"
                result += "="*50 + "\n"
                result += f"Metode: HSV + RGB + GLCM \n"
                result += f"🎯 AKURASI: {accuracy*100:.2f}%\n\n"
                
                # Format Confusion Matrix yang readable
                result += "📊 CONFUSION MATRIX:\n"
                result += "="*50 + "\n"
                result += f"{'':>12} {'Prediksi →':^38}\n"
                result += f"{'Aktual ↓':>12} {'Mentah':>12} {'Muda':>12} {'Matang':>12}\n"
                result += "-"*50 + "\n"
                categories = ['Mentah', 'Muda', 'Matang']
                for i, category in enumerate(categories):
                    result += f"{category:>12}"
                    for j in range(len(categories)):
                        result += f"{cm[i][j]:>12}"
                    result += "\n"
                result += "="*50 + "\n"
                
                # Interpretasi per kelas
                result += "\n📖 INTERPRETASI:\n"
                for i, category in enumerate(categories):
                    total_actual = cm[i].sum()
                    correct = cm[i][i]
                    if total_actual > 0:
                        class_acc = (correct / total_actual) * 100
                        result += f"\n  {category}:\n"
                        result += f"  ✓ Benar: {correct}/{total_actual} ({class_acc:.1f}%)\n"
                        
                        for j, pred_cat in enumerate(categories):
                            if i != j and cm[i][j] > 0:
                                result += f"  ✗ Salah sbg {pred_cat}: {cm[i][j]}\n"
                
                result += "\n" + "="*50 + "\n"
                result += "Program siap digunakan!\n"
                result += "Klik 'Pilih Gambar Tomat' untuk mulai klasifikasi.\n"
                
                self.result_text.insert(tk.END, result)
                
                # Tampilkan confusion matrix
                self.show_confusion_matrix(cm)
                
                # Auto-save model saat startup
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"tomato_model_{method}_autoload_{timestamp}.pkl"
                model_path = os.path.join("models", model_filename)
                
                # Buat folder models jika belum ada
                os.makedirs("models", exist_ok=True)
                
                self.classifier.save_model(model_path)
                self.result_text.insert(tk.END, f"\n✅ Model auto-saved: {model_filename}\n")
                
            except Exception as e:
                self.result_text.insert(tk.END, f"\n⚠️ Auto-load gagal: {str(e)}\n")
                self.result_text.insert(tk.END, "Silakan klik 'Load Dataset & Training' secara manual.\n")
        else:
            self.result_text.insert(tk.END, f"⚠️ Folder 'dataset' tidak ditemukan!\n")
            self.result_text.insert(tk.END, "Silakan klik 'Load Dataset & Training' untuk memilih folder dataset.\n")



if __name__ == "__main__":
    root = tk.Tk()
    app = TomatoClassifierGUI(root)
    root.mainloop()