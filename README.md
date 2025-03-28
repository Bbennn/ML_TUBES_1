# Feedforward Neural Network (FFNN) - IF3270 Pembelajaran Mesin

Repository ini berisi implementasi **Feedforward Neural Network (FFNN)** dari awal (*from scratch*) sebagai bagian dari **Tugas Besar 1 IF3270 Pembelajaran Mesin**.

## 📂 Struktur Direktori

```
├── src/                    # Folder berisi implementasi FFNN
│   ├── FFNN.py             # Implementasi utama FFNN
│   ├── Activation.py       # Implementasi fungsi aktivasi
│   ├── Initialization.py   # Implementasi fungsi initilizer bobot
│   ├── Layer.py            # Implementasi layer network
│   ├── Loss.py             # Implementasi Loss function
│   └── utils.py            # Fungsi tambahan untuk visualisasi plot loss
├── test/               # Folder berisi pengujian model
│   ├── file.ipynb      # Notebook contoh penggunaan dan evaluasi model
│   └── ...
├── doc/                # Folder untuk laporan tugas besar
│   ├── laporan.pdf     # Laporan hasil implementasi dan analisis
│   └── ...
├── README.md           # Dokumentasi proyek
└── requirements.txt    # Daftar dependensi yang dibutuhkan
```

## 🚀 Cara Menjalankan Program

1. **Instalasi Dependensi**  
   Pastikan Anda memiliki Python 3.x terinstal, lalu instal dependensi yang diperlukan:

   ```sh
   pip install -r requirements.txt
   ```

2. **Menjalankan Model**  
   Untuk melatih model, silahkan import model FFNN ke dalam file jupyter notebook atau python. Jika Anda menggunakan python run program dengan command running biasa.

   ```sh
   python3 file-name.py
   ```

3. **Menjalankan Notebook**  
   Anda juga bisa membuka `test/file-name.ipynb` untuk melihat contoh penggunaan model dalam format Jupyter Notebook.

## 📊 Fitur yang Diimplementasikan

- **Dukungan jumlah neuron fleksibel** untuk tiap layer.
- **Beragam fungsi aktivasi**, termasuk:
  - Linear
  - ReLU
  - Sigmoid
  - Tanh
  - Softmax
  - Leaky ReLU (bonus)
  - SeLU (bonus)
- **Beragam fungsi loss**, termasuk:
  - Mean Squared Error (MSE)
  - Binary Cross-Entropy
  - Categorical Cross-Entropy
- **Inisialisasi bobot yang dapat dikustomisasi**, seperti:
  - Zero initialization
  - Random uniform
  - Random normal
  - Xavier (bonus)
  - He (bonus)
- **Forward propagation dan backward propagation** dengan chain rule.
- **Pembelajaran dengan Gradient Descent**.
- **Mekanisme penyimpanan dan pemuatan model**.
- **Visualisasi jaringan dan distribusi bobot**.

## 🔬 Eksperimen dan Analisis

Dalam proyek ini, kami melakukan eksperimen untuk menganalisis berbagai hyperparameter, termasuk:

1. **Pengaruh jumlah neuron dan layer** terhadap performa model.
2. **Pengaruh fungsi aktivasi** terhadap konvergensi dan hasil prediksi.
3. **Pengaruh learning rate** terhadap kecepatan dan stabilitas pelatihan.
4. **Pengaruh metode inisialisasi bobot** terhadap performa model.
5. **Perbandingan model dengan library `sklearn.MLPClassifier`**.

## 👥 Pembagian Tugas

| Nama  | NIM  | Tugas |
|--------|------|------|
| Benjamin Sihombing | 13522054 |Dokumen bagian deskripsi kelas, pengujian depth dan width, dan pengujian aktivasi. Program bagian kelas aktivasi, fungsi plotting distribusi, dan fungsi visualisasi graf.|
| M. Atpur Rafif | 13522086 | Dokumen bagian forward, backward, analisis softmax, pengujian inisialisasi bobot, dan perbandingan dengan sklearn. Program bagian fungsi forward, fungsi backward, dan kelas layer. |
| Suthasoma Mahardhika Munthe | 13522098 | Pengujian dan dokumen bagian variasi learning rate, save dan load, plot training dan validation loss, kelas loss function, kelas FFNN, dan kelas initializer. |