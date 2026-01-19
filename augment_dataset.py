import cv2
import numpy as np
import os
from pathlib import Path
import random

class DataAugmentation:
    def __init__(self, input_folder='dataset', output_folder='dataset_augmented'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.categories = ['mentah', 'muda', 'matang']
        
    def augment_image(self, image, augmentation_type):
        """
        Aplikasikan augmentasi ke gambar
        
        Augmentation types:
        1. rotate_90: Rotasi 90 derajat
        2. rotate_180: Rotasi 180 derajat
        3. rotate_270: Rotasi 270 derajat
        4. flip_horizontal: Flip horizontal (mirror)
        5. flip_vertical: Flip vertical
        6. brightness_up: Tingkatkan brightness
        7. brightness_down: Kurangi brightness
        8. zoom_in: Zoom in (crop tengah)
        9. zoom_out: Zoom out (padding)
        """
        h, w = image.shape[:2]
        
        if augmentation_type == 'rotate_90':
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        elif augmentation_type == 'rotate_180':
            return cv2.rotate(image, cv2.ROTATE_180)
        
        elif augmentation_type == 'rotate_270':
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        elif augmentation_type == 'flip_horizontal':
            return cv2.flip(image, 1)
        
        elif augmentation_type == 'flip_vertical':
            return cv2.flip(image, 0)
        
        elif augmentation_type == 'brightness_up':
            # Tingkatkan brightness 30%
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 2] = hsv[:, :, 2] * 1.3
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            hsv = hsv.astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        elif augmentation_type == 'brightness_down':
            # Kurangi brightness 30%
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 2] = hsv[:, :, 2] * 0.7
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            hsv = hsv.astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        elif augmentation_type == 'zoom_in':
            # Zoom in 20% (crop tengah)
            crop_size = int(min(h, w) * 0.8)
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
            cropped = image[start_h:start_h+crop_size, start_w:start_w+crop_size]
            return cv2.resize(cropped, (w, h))
        
        elif augmentation_type == 'zoom_out':
            # Zoom out 20% (padding)
            new_size = int(min(h, w) * 1.2)
            canvas = np.zeros((new_size, new_size, 3), dtype=np.uint8)
            start = (new_size - h) // 2
            canvas[start:start+h, start:start+w] = image
            return cv2.resize(canvas, (w, h))
        
        else:
            return image
    
    def augment_dataset(self, target_per_class=100):
        """
        Augmentasi seluruh dataset
        
        Args:
            target_per_class: Target jumlah gambar per kelas (default: 100)
        """
        # Daftar augmentasi yang akan digunakan
        augmentation_types = [
            'rotate_90', 'rotate_180', 'rotate_270',
            'flip_horizontal', 'flip_vertical',
            'brightness_up', 'brightness_down',
            'zoom_in', 'zoom_out'
        ]
        
        print("="*60)
        print("🎨 DATA AUGMENTATION - Tomato Classifier")
        print("="*60)
        print(f"\nFolder Input: {self.input_folder}")
        print(f"Folder Output: {self.output_folder}")
        print(f"Target per kelas: {target_per_class} gambar\n")
        
        # Buat folder output
        os.makedirs(self.output_folder, exist_ok=True)
        
        total_original = 0
        total_augmented = 0
        
        for category in self.categories:
            input_path = os.path.join(self.input_folder, category)
            output_path = os.path.join(self.output_folder, category)
            
            if not os.path.exists(input_path):
                print(f"⚠️  Folder '{category}' tidak ditemukan, skip...")
                continue
            
            # Buat folder output untuk kategori ini
            os.makedirs(output_path, exist_ok=True)
            
            # Ambil semua file gambar
            valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.avif')
            image_files = [f for f in os.listdir(input_path) 
                          if f.lower().endswith(valid_extensions) or '.' not in f]
            
            num_original = len(image_files)
            total_original += num_original
            
            print(f"\n📁 Kategori: {category.upper()}")
            print(f"   Gambar asli: {num_original}")
            
            if num_original == 0:
                print(f"   ⚠️  Tidak ada gambar, skip...")
                continue
            
            # Hitung berapa augmentasi per gambar
            augmentations_per_image = (target_per_class - num_original) // num_original
            
            saved_count = 0
            
            # Copy gambar asli dulu
            for idx, filename in enumerate(image_files):
                src_path = os.path.join(input_path, filename)
                
                # Tentukan ekstensi file
                if '.' in filename:
                    ext = filename.split('.')[-1]
                else:
                    ext = 'jpg'  # Default
                
                # Copy original
                dst_path = os.path.join(output_path, f"{category}_original_{idx+1:03d}.{ext}")
                image = cv2.imread(src_path)
                
                if image is not None:
                    cv2.imwrite(dst_path, image)
                    saved_count += 1
                    
                    # Generate augmented images
                    aug_count = 0
                    for aug_type in augmentation_types:
                        if aug_count >= augmentations_per_image:
                            break
                        
                        augmented = self.augment_image(image, aug_type)
                        aug_filename = f"{category}_aug_{idx+1:03d}_{aug_type}.{ext}"
                        aug_path = os.path.join(output_path, aug_filename)
                        cv2.imwrite(aug_path, augmented)
                        saved_count += 1
                        aug_count += 1
            
            print(f"   ✅ Total gambar: {saved_count}")
            total_augmented += saved_count
        
        print("\n" + "="*60)
        print("✅ AUGMENTASI SELESAI!")
        print("="*60)
        print(f"\nRingkasan:")
        print(f"  • Gambar asli: {total_original}")
        print(f"  • Total setelah augmentasi: {total_augmented}")
        print(f"  • Peningkatan: {total_augmented - total_original} gambar (+{(total_augmented/total_original-1)*100:.1f}%)")
        print(f"\nDataset baru tersimpan di: {self.output_folder}/")
        print("\n🚀 Sekarang training ulang dengan:")
        print(f"   dataset_path = '{self.output_folder}'")
        print("="*60)


def main():
    """
    Main function untuk menjalankan augmentasi
    """
    # Inisialisasi augmentation
    augmenter = DataAugmentation(
        input_folder='dataset',           # Folder input
        output_folder='dataset_augmented' # Folder output
    )
    
    # Jalankan augmentasi dengan target 100 gambar per kelas
    augmenter.augment_dataset(target_per_class=100)
    
    print("\n💡 Tips:")
    print("   - Cek folder 'dataset_augmented' untuk melihat hasilnya")
    print("   - Jika sudah OK, training ulang model dengan dataset baru")
    print("   - Akurasi akan meningkat signifikan!\n")


if __name__ == "__main__":
    main()
