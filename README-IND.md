# APPLICATION OF AUGMENTED REALITY TECHNOLOGY USING CNN (CONVOLUTIONAL NEURAL NETWORK) IN LEARNING JAVANESE CHARACTERS

<br/>

- [ ] **TAHAP PERTAMA | MEMPERSIAPKAN MODEL**
> [!IMPORTANT]
> Pertama anda harus menyiapkan model terlebih dahulu untuk menjalankan aplikasi ini, atau bisa langsung menggunakan model saya yang sudah ada, [Unduh di sini](https://github.com/YudhaDevelops/JawaLearn-PyQt5/releases/tag/Models-JawaLearn)

> [!NOTE]
> Atau Anda bisa membuat model sendiri dengan database Anda sendiri, menggunakan program yang saya gunakan di kaggle,
> Lihat [Untuk Klasifikasi Gambar Model](https://www.kaggle.com/happyngoding/cnn-aksara-use-7-models-fix), dan lihat [Untuk Model AR / objek deteksi Aksara Jawa](https://www.kaggle.com/happyngoding/ssd-mobilenet-v2-python-3-10-12)

<br/>

- [x] **TAHAP PERTAMA | MEMPERSIAPKAN MODEL**
- [ ] **TAHAP KEDUA | MEMPERSIAPKAN LIBRARY DAN PYTHON**
      
# Library Yang Harus Di Install
> [!NOTE]
> Saya menggunakan python versi ```3.9.19```. Dengan menggunakan lingkungan / env di miniconda3.

```
pip install keras==2.12.0
```
```
pip install numpy==1.23.5
```
```
pip install opencv_python==4.9.0.80
```
```
pip install opencv_python_headless==4.9.0.80
```
```
pip install Pillow==9.5.0
```
```
pip install Pillow==10.3.0
```
```
pip install psutil==5.9.8
```
```
pip install PyQt5==5.15.10
```
```
pip install PyQt5_sip==12.13.0
```
```
pip install tensorflow==2.12.0
```
```
pip install tensorflow_intel==2.12.0
```
```
pip install tflite_runtime==2.14.0
```

<br/>

- [x] **TAHAP PERTAMA | MEMPERSIAPKAN MODEL**
- [x] **TAHAP KEDUA | MEMPERSIAPKAN LIBRARY DAN PYTHON**
- [ ] **TAHAP KETIGA | CLONE PROGRAM FROM MY GITHUB**
      
# Clone Program
> git clone https://github.com/YudhaDevelops/JawaLearn-PyQt5

<br/>

- [x] **TAHAP PERTAMA | MEMPERSIAPKAN MODEL**
- [x] **TAHAP KEDUA | MEMPERSIAPKAN LIBRARY DAN PYTHON**
- [x] **TAHAP KETIGA | CLONE PROGRAM FROM MY GITHUB**
- [ ] **TAHAP KE EMPAT | Ubah lokasi model **

# Ubah lokasi model yang sudah anda buat Atau download dari model rilis saya

<br/>

- [x] **TAHAP PERTAMA | MEMPERSIAPKAN MODEL**
- [x] **TAHAP KEDUA | MEMPERSIAPKAN LIBRARY DAN PYTHON**
- [x] **TAHAP KETIGA | CLONE PROGRAM FROM MY GITHUB**
- [x] **TAHAP KE EMPAT | UBAH LOKASI MODEL**
- [ ] **TAHAP KE LIMA | JALANKAN PROGRAM**

# Jalankan Program
```
python main.py
```

<br/>

- [x] **TAHAP PERTAMA | MEMPERSIAPKAN MODEL**
- [x] **TAHAP KEDUA | MEMPERSIAPKAN LIBRARY DAN PYTHON**
- [x] **TAHAP KETIGA | CLONE PROGRAM FROM MY GITHUB**
- [x] **TAHAP KE EMPAT | UBAH LOKASI MODEL**
- [x] **TAHAP KE LIMA | JALANKAN PROGRAM**
      
# Jika program berjalan dengan baik maka akan terlihat seperti gambar di bawah ini
> [!IMPORTANT]
> Anda harus membuat model terlebih dahulu dengan ekstensi model .tflite untuk fitur AR dan .h5 untuk fitur klasifikasi aksara jawa.

## 1. Menjalankan fitur Klasifikasi Aksara Jawa
![klasifikasi](https://github.com/YudhaDevelops/JawaLearn-PyQt5/assets/106727245/1c3d93c5-3727-441f-9d68-58131150f729)

## 2. Menjalankan fitur Deteksi AR Aksara Jawa
![objek_deteksi](https://github.com/YudhaDevelops/JawaLearn-PyQt5/assets/106727245/41b04ddc-e0f6-44e9-a9fd-94200fe83ab6)

# Ini adalah program AR yang menggunakan model sebagai otak atau database yang digunakan untuk melakukan transliterasi dari aksara Jawa ke aksara Latin.
Program akan menambahkan kotak di sekitar area aksara jawa dan memberikan arti karakter di kiri atas aksara jawa yang terdeteksi. Dapat dilihat pada video dibawah ini

https://github.com/YudhaDevelops/JawaLearn-PyQt5/assets/106727245/5e11c069-26fb-4b43-9a07-7d12ab41d471

