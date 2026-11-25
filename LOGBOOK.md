# Logbook Evaluasi CPMK 3

**Nama** Muhammad Farhan\
**NRP** 5054241018\
**Periode kegiatan** 15–25 November 2026

Logbook ini dirangkum secara retrospektif mulai 24 November 2026 dan diperbarui setelah evaluasi v2. Tanggal kegiatan mengikuti catatan mahasiswa. Notebook dan log yang tersedia digunakan sebagai bukti hasil dan kendala teknis.

## 15–21 November 2026

### 15 November — Memahami tugas dan menyusun asumsi awal

Saya membaca ketentuan tugas dan merencanakan perbandingan YOLO11, Mask R-CNN, dan RF-DETR pada Panoptic Indoor Segmentation v5. Ketiganya mewakili pendekatan CNN one-stage, CNN two-stage, dan transformer. Karena NRP berakhir dengan digit 8, eksperimen membandingkan semua kelas dengan tiga kelas tersulit yang dipilih dari hasil validation awal.

Saya memisahkan foto HP sebagai test set mandiri untuk melihat performa model pada ruangan dan kondisi pengambilan yang berbeda dari dataset sumber.

**Hipotesis awal yang didokumentasikan pada 25 November 2026.** Berdasarkan perbedaan karakteristik arsitektur ketiga model, YOLO11s-seg dihipotesiskan memiliki efisiensi komputasi dan kecepatan inferensi terbaik. Mask R-CNN dihipotesiskan menghasilkan segmentasi objek yang lebih presisi daripada YOLO11s-seg melalui pendekatan two-stage dan RoIAlign. RF-DETR Seg Small dihipotesiskan memperoleh akurasi segmentasi keseluruhan dan kemampuan generalisasi terbaik melalui arsitektur Transformer dan pretrained backbone DINOv2.

**Waktu pencatatan.** Saya sudah memiliki asumsi tersebut sebelum training pertama dan baru menuliskannya secara retrospektif pada 25 November 2026. Rumusan yang terdokumentasi berfokus pada tingkat model.\
**Rujukan** [Ketentuan tugas dan variabel NRP](README.md#23-variabel-eksperimen-berdasarkan-nrp).

### 16 November — Memulai training berdasarkan rancangan awal

Saya mulai menjalankan training di Kaggle menggunakan split asli Roboflow. Dataset berisi 397 gambar train, 166 validation, dan 359 test. Pipeline memeriksa anotasi dan mengoreksi empat gambar yang orientasinya tidak sesuai dengan dimensi anotasi pada salinan kerja.

Pada percobaan awal, pembuatan virtual environment gagal saat `ensurepip`. Setup worker kemudian diubah agar memakai pip dari environment utama. Run Kaggle yang tersimpan dalam repo tercatat pada **18 November**, setelah training pertama dimulai pada 16 November. Pada run tersimpan itu, YOLO11n-seg dan Mask R-CNN menyelesaikan training serta evaluasi. RF-DETR Seg Nano mencapai 30 epoch dan menyimpan best checkpoint, tetapi inferensi validation berhenti pada `KeyError: 74`.

**Bukti** [Audit dataset](results/reports/audit.json), [normalisasi data](results/reports/normalization.json), dan [notebook hasil Kaggle](results/segmentasi-instance-indoor-evaluasi-cpmk-3.ipynb).

### 19 November — Mengambil foto mandiri

Saya mengambil foto ruangan menggunakan kamera HP untuk test set mandiri. Salinan kerja awal berisi **44 foto**. Hasil tinjauan visual menandai **11 foto sulit** dan **33 foto normal**. Kondisi sulit yang dicatat mencakup cahaya redup, backlight, dan objek yang banyak tertutup.

**Bukti** `dataset/foto-sendiri/photo_metadata.csv` mencatat kondisi pada salinan kerja awal. Bukti kamera untuk paket uji final tersedia pada [CSV EXIF 56 foto](evidence/metadata-foto-mandiri.csv).

### 20–21 November — Mengerjakan anotasi manual

Saya mengerjakan anotasi manual foto sendiri untuk menyiapkan ground truth. Label mengikuti kelas dataset training, dan setiap instance memiliki mask tersendiri. Foto dan anotasinya disiapkan untuk evaluasi ketiga model.

**Bukti** [Screenshot editor Roboflow](evidence/bukti-anotasi-roboflow.png). Paket foto mandiri berkembang menjadi 56 foto pada evaluasi akhir.

## 23–24 November 2026

### 23 November — Meninjau evaluasi model lama

Saya mengevaluasi dan meninjau hasil model dari run awal. Tabel berikut memakai COCO Mask AP50–95 pada skala 0–100 dari CSV evaluasi Kaggle.

| Model | Validation, 166 gambar | Original test, 359 gambar |
| --- | ---: | ---: |
| YOLO11n-seg | 14,15 | 8,68 |
| Mask R-CNN ResNet-50 FPN | 16,08 | 7,15 |
| RF-DETR Seg Nano | Inferensi validation gagal | Belum tersedia |

Mask R-CNN memperoleh AP validation lebih tinggi, sedangkan YOLO11n-seg memperoleh AP original test lebih tinggi. Saya meninjau distribusi kelas sebagai salah satu kemungkinan penyebab selisih skor. Sebanyak 24 kelas yang muncul di train memiliki kurang dari 10 instance, dan 9 kelas terdaftar tidak muncul di train.

**Bukti** [CSV evaluasi Kaggle](results/reports/comparison.csv) dan [distribusi kelas](results/reports/class_distribution.csv).

### 24 November — Training ulang di Colab L4

Saya menjalankan profil `quality` di Colab dengan GPU NVIDIA L4. Dataset tetap memakai split Roboflow yang sama. Profil ini memakai YOLO11s-seg, scheduler Mask R-CNN berdasarkan AP validation, dan RF-DETR Seg Small. Checkpoint, log, serta hasil evaluasi disimpan di Drive.

- **YOLO11s-seg** selesai 30 epoch dan evaluasi validation. Log bawaan Ultralytics menunjukkan Mask mAP50–95 sebesar **23,3**. Saat notebook dijalankan ulang, worker menemukan output yang lengkap dan melewati YOLO.
- **Mask R-CNN** tercatat menyelesaikan epoch 13 dengan Mask AP50–95 validation **14,04**. Output sesi pertama berhenti saat inferensi berikutnya. Percobaan resume membaca checkpoint, tetapi gagal saat memulihkan state acak CUDA dengan `RNG state must be a torch.ByteTensor`. Kode pemulihan diperbaiki dan diuji lokal. Keputusan berikutnya adalah menjalankan Mask R-CNN saja dari epoch 1 dengan label `colab_quality02`. Run tersebut selesai 30 epoch. Mask AP validation terbaik mencapai **17,11** pada skala 0–100. [Log training Mask R-CNN](results/colab/colab_quality02_maskrcnn.ipynb)
- **RF-DETR Seg Small** selesai 30 epoch dan evaluasi validation. Best checkpoint EMA tersimpan. Mask AP50–95 dari evaluator bawaan RF-DETR sebesar **33,17**. Evaluasi bersama memakai checkpoint EMA yang dipertahankan. [Evaluasi v2](04_evaluasi_foto_pribadi_colab_v2.ipynb)
- **Tabel laporan** menulis CSV, lalu gagal ditampilkan dengan `KeyError: 'model'`. Penyebabnya adalah kolom `model` dikeluarkan dari tampilan sebelum dipakai untuk mengurutkan tabel. Notebook utama diperbaiki agar mengurutkan data sebelum memilih kolom tampilan.

**Bukti** [Colab sesi pertama](results/colab/colab_quality01_session1.ipynb), [Colab sesi kedua](results/colab/colab_quality01_session2.ipynb), dan [notebook utama](01_instance_segmentation_kaggle.ipynb).

#### Log training run final semua kelas

| Model | Run | Berkas di Google Drive |
| --- | --- | --- |
| YOLO11s-seg | `colab_quality01` | [training_log.csv](https://drive.google.com/file/d/1IAhz5Iw1eeHmef0u75kyz18VEOrBl2cC/view?usp=drive_link) · [results.csv dari Ultralytics](https://drive.google.com/file/d/1zN5f4Fg4gX1kkIk_dR69FVwGPJGaA5EJ/view?usp=drive_link) |
| Mask R-CNN | `colab_quality02` | [training_log.csv](https://drive.google.com/file/d/1gmMSFbj-uzc-Bri4D-50K2n5qNd_wkyU/view?usp=drive_link) |
| RF-DETR Seg Small | `colab_quality01` | [Log TensorBoard](https://drive.google.com/file/d/1BRpkx9Z3BTyCZO9CQ2NIeo-C_j4eSFZl/view?usp=drive_link) |

Untuk YOLO, notebook menyalin `native/results.csv` menjadi `training_log.csv`. Kedua CSV `training_log.csv` dan berkas TensorBoard RF-DETR dapat dibuka melalui tautan Drive di atas.

### Catatan durasi pada log

Durasi worker mencakup persiapan dan evaluasi yang dijalankan oleh worker. Durasi Mask R-CNN sesi pertama di Colab dihitung dari total waktu 13 epoch yang tercatat.

| Run dan GPU | Model | Durasi tercatat | Status |
| --- | --- | ---: | --- |
| Kaggle `run01`, T4 | YOLO11n-seg | 10 menit 13 detik | Selesai training dan evaluasi |
| Kaggle `run01`, T4 | Mask R-CNN | 56 menit 48 detik | Selesai training dan evaluasi |
| Kaggle `run01`, T4 | RF-DETR Seg Nano | 33 menit 27 detik | Training selesai, inferensi validation gagal |
| Colab `colab_quality01`, L4 | YOLO11s-seg | 8 menit 51 detik | Selesai training dan validation |
| Colab `colab_quality01`, L4 | Mask R-CNN | 14 menit 59 detik | Total 13 epoch beserta validation per epoch |
| Colab `colab_quality01`, L4 | RF-DETR Seg Small | 35 menit 52 detik | Selesai training dan validation |
| Colab `colab_quality02`, L4 | Mask R-CNN | 34 menit 34 detik | Total 30 epoch beserta validation per epoch |

## 25 November 2026

### Evaluasi kedua test set dan eksperimen NRP

Evaluasi awal membandingkan 359 gambar Roboflow dengan 56 foto pribadi. Untuk eksperimen NRP, saya memakai Mask AP validation untuk memilih `cabinet-merged`, `blanket`, dan `rug-merged`. Ketiga model menyelesaikan run `colab_hard3_01` dalam total waktu worker sekitar 53 menit 30 detik. Perbandingan dilakukan pada tiga kelas dan gambar test yang sama. [Evaluasi awal](results/colab/eval_personal56_v1.ipynb) · [Eksperimen NRP](05_eksperimen_hard3_colab.ipynb)

### Revisi anotasi dan analisis kegagalan

Satu mask piring muncul dengan beberapa label pada COCO awal. Saya meninjau hasil audit anotasi dan menyetujui koreksi. Paket v2 memuat 430 mask pada 56 foto, termasuk 17 foto kondisi sulit. Evaluasi ulang memakai checkpoint terlatih dan selesai dalam total sekitar 9 menit 33 detik waktu worker L4. Notebook 06 membandingkan 31 kelas bersama dan mengganti kasus `v2_023` dengan `v2_029` karena label pada contoh sebelumnya perlu pemeriksaan. Uji ambang confidence memulihkan dua kasus dan menunjukkan peningkatan false positive. [Evaluasi v2](04_evaluasi_foto_pribadi_colab_v2.ipynb) · [Notebook analisis](06_analisis_foto_pribadi_colab_v2.ipynb) · [Laporan](ANALISIS_CPMK3.md)

## Status demo

Demo inferensi dan tanya jawab 10 menit telah dilaksanakan bersama dosen. Status dicatat pada 25 November 2026. [Notebook demo](07_inferensi_foto_baru_colab.ipynb)

## Verifikasi EXIF foto uji

Seluruh 56 foto uji memiliki padanan berkas asli dengan EXIF dan checksum. Tag `DateTimeOriginal` pada berkas asli paket uji mencatat 24 November 2026. [Pemetaan dan metadata 56 foto](evidence/metadata-foto-mandiri.csv)
