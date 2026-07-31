Berikut adalah catatan kuliah Pengolahan Citra Digital dengan topik Klasifikasi Sampah Organik dan Anorganik, yang disampaikan oleh Bu Yuita pada tanggal 3 Juli 2026.

---

# Catatan Kuliah: Klasifikasi Sampah Organik dan Anorganik

## Topik Utama
*   Pembahasan proyek klasifikasi sampah menjadi dua kategori (organik dan anorganik) menggunakan kombinasi teknik Pengolahan Citra Digital (PCD) dan Visi Komputer (VK).
*   Penjelasan detail mengenai pipeline sistem yang lengkap, meliputi tahap preprocessing, ekstraksi fitur, dan klasifikasi.
*   Penggunaan Convolutional Neural Network (CNN) sebagai model utama untuk klasifikasi.
*   Metode PCD yang diterapkan pada tahap preprocessing (resize, konversi warna, smoothing, segmentasi) dan ekstraksi fitur (warna, tekstur, bentuk).

## Definisi & Konsep Kunci
*   **Preprocessing Citra:** Tahap awal dalam pengolahan citra untuk mempersiapkan data sebelum analisis lebih lanjut. Meliputi:
    *   **Resize Gambar:** Mengubah ukuran citra agar konsisten.
    *   **Konversi Warna (RGB ke HSV):** Mengubah representasi warna dari model RGB (Red, Green, Blue) ke HSV (Hue, Saturation, Value) yang lebih intuitif untuk analisis warna dan robust terhadap perubahan pencahayaan.
    *   **Smoothing (Image Denoising):** Proses mengurangi noise atau gangguan pada citra menggunakan filter seperti Gaussian dan Median.
    *   **Segmentasi Objek:** Proses memisahkan objek utama dari latar belakang citra, sering menggunakan teknik thresholding dan operasi morfologi.
*   **Ekstraksi Fitur:** Proses mendapatkan informasi deskriptif dari citra yang akan digunakan untuk klasifikasi. Fitur yang diekstrak meliputi:
    *   **Fitur Warna:** Mean, standar deviasi (STD) HSV, dan histogram warna.
    *   **Fitur Tekstur:** GLCM (Gray-Level Co-occurrence Matrix), yang menghasilkan fitur seperti contrast, correlation, energy, dan homogeneity.
    *   **Fitur Bentuk:** Area ratio, solidity, dan circularity.
*   **Klasifikasi:** Proses mengkategorikan objek berdasarkan fitur yang telah diekstrak.
*   **Convolutional Neural Network (CNN):** Sebuah jenis model deep learning yang sangat efektif untuk tugas-tugas pengolahan citra, digunakan sebagai model klasifikasi utama dalam proyek ini.
*   **StandardScaler:** Teknik standarisasi fitur yang mengubah data sehingga memiliki rata-rata 0 dan variansi 1. Ini penting untuk memastikan semua fitur memiliki skala yang sama sebelum dimasukkan ke model klasifikasi.

## Contoh Kode / Algoritma
Dalam sesi ini disebutkan penggunaan algoritma dan metode berikut:
*   Gaussian Filter
*   Median Filter
*   Thresholding
*   Operasi Morfologi
*   GLCM (Gray-Level Co-occurrence Matrix)
*   Convolutional Neural Network (CNN)
*   StandardScaler

Tidak ada contoh kode spesifik yang ditampilkan dalam transkrip atau materi pendukung.

## Ringkasan Alur Pembahasan
Kuliah ini membahas proyek akhir mata kuliah Pengolahan Citra Digital dan Visi Komputer, yaitu "Smart Waste Classifier" untuk mengklasifikasikan sampah organik dan anorganik. Pembahasan dimulai dengan deskripsi ide proyek dan alasan di baliknya, menyoroti tantangan pemilahan sampah dan potensi solusi otomatis.

Selanjutnya, dijelaskan pipeline lengkap sistem yang terbagi menjadi tiga tahap utama:
1.  **Preprocessing:** Tahap ini meliputi resize gambar ke ukuran yang konsisten, konversi warna dari RGB ke HSV, smoothing menggunakan Gaussian dan Median Filter untuk mengurangi noise, serta segmentasi objek menggunakan thresholding dan operasi morfologi untuk memisahkan objek sampah dari latar belakang.
2.  **Ekstraksi Fitur:** Setelah preprocessing, fitur-fitur penting diekstrak dari citra. Ini mencakup fitur warna (mean, standar deviasi HSV, dan histogram warna), fitur tekstur (menggunakan GLCM untuk mendapatkan contrast, correlation, energy, dan homogeneity), serta fitur bentuk (area ratio, solidity, dan circularity).
3.  **Klasifikasi:** Fitur-fitur yang telah diekstrak kemudian distandarisasi menggunakan StandardScaler sebelum dimasukkan ke model klasifikasi. Model klasifikasi utama yang digunakan adalah Convolutional Neural Network (CNN).

Sesi ini diakhiri dengan demo hasil dari sistem yang telah dibangun.

## Catatan Tambahan
Tidak disebutkan di sesi ini.

---