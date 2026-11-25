# Evaluasi CPMK 3 — Segmentasi Instance dengan Deep Learning

| Identitas | Keterangan |
| --- | --- |
| NRP | **5054241018** |
| Mata kuliah | Visi Komputer (EA234503) |
| Program studi | Rekayasa Kecerdasan Artifisial, ITS |
| Dosen pada materi | Anny Yuniarti |
| Sumber | [10 - Segmentasi Instance dengan Deep Learning.pdf](<10 - Segmentasi Instance dengan Deep Learning.pdf>) |

README ini menuliskan kembali materi perkuliahan dan ketentuan tugas dari PDF. Ringkasan materi mengacu pada halaman 1–28, sedangkan ketentuan Evaluasi CPMK 3 mengacu pada halaman 29–32. Hasil evaluasi, eksperimen NRP, dan lima studi kasus tersedia di [Analisis CPMK 3](ANALISIS_CPMK3.md).

## 1. Ringkasan materi

### 1.1. Capaian pembelajaran

CPMK 3: mahasiswa mampu menerapkan **semantic segmentation** dan **instance segmentation** menggunakan deep learning. Fokus pertemuan ini adalah segmentasi instance, yaitu menghasilkan mask yang memisahkan setiap objek secara individual, termasuk objek-objek dengan kelas yang sama.

### 1.2. Perbedaan tugas dalam visi komputer

| Tugas | Keluaran | Contoh |
| --- | --- | --- |
| Image classification | Label untuk keseluruhan gambar. | Gambar mengandung kucing. |
| Object detection | Kelas dan lokasi setiap objek berupa bounding box. | Dua anjing memiliki dua kotak deteksi. |
| Semantic segmentation | Label kelas untuk setiap piksel, tanpa identitas objek individual. | Piksel kedua anjing sama-sama berlabel `anjing`. |
| Instance segmentation | Mask dan kelas untuk masing-masing objek individual. | `anjing #1` dan `anjing #2` memiliki mask berbeda. |
| Panoptic segmentation | Label semantik seluruh piksel sekaligus identitas instance untuk objek yang dapat dihitung. | Langit dan rumput diberi label, sedangkan kedua anjing tetap dipisahkan. |

Pada pembahasan panoptic segmentation, **things** adalah objek yang dapat dihitung satu per satu, misalnya anjing, kucing, atau mobil. **Stuff** adalah wilayah seperti langit, rumput, atau jalan. Dalam perbandingan pada slide, instance segmentation berfokus pada things, sedangkan semantic dan panoptic segmentation juga melabeli stuff.

### 1.3. Contoh pemahaman segmentasi

Halaman 8–9 memberikan contoh citra 10 × 10 piksel yang berisi langit sebanyak 20 piksel, rumput 50 piksel, anjing pertama 12 piksel, anjing kedua 10 piksel, dan kucing 8 piksel. Kedua anjing saling bersentuhan.

Berikut penjelasan untuk latihan tersebut berdasarkan konsep pada materi:

| Tugas | Hasil pelabelan | Jumlah piksel berlabel |
| --- | --- | --- |
| Semantic segmentation | Langit 20, rumput 50, anjing 22, dan kucing 8; identitas kedua anjing tidak dibedakan. | 100 piksel |
| Instance segmentation | Tiga mask objek: anjing #1 sebanyak 12, anjing #2 sebanyak 10, dan kucing sebanyak 8. Langit dan rumput tidak diberi label instance. | 30 piksel objek |
| Panoptic segmentation | Langit 20, rumput 50, anjing #1 sebanyak 12, anjing #2 sebanyak 10, dan kucing sebanyak 8. | 100 piksel |

Dengan menghitung satu mask untuk setiap wilayah stuff pada contoh ini, panoptic segmentation menghasilkan **5 mask**, yaitu **2 stuff** dan **3 things**. Untuk menghitung jumlah sapi sekaligus mengukur luas rumput yang tersisa, panoptic segmentation sesuai karena dapat memisahkan setiap sapi dan melabeli wilayah rumput.

### 1.4. Aplikasi segmentasi instance

- **Kendaraan otonom:** memahami jalan, mobil, rambu, dan pejalan kaki.
- **Medis:** segmentasi organ atau tumor pada citra MRI/CT.
- **Pertanian:** identifikasi tanaman dan gulma dari citra drone.
- **Augmented reality:** memahami lingkungan secara presisi untuk interaksi virtual.

### 1.5. Fully Convolutional Network dan pendekatan berbasis FCN

Fully Convolutional Network (FCN) menggunakan jaringan konvolusi untuk menghasilkan prediksi spasial. Proses downsampling, misalnya melalui pooling atau strided convolution, memperkecil feature map. Proses upsampling mengembalikan resolusi spasial untuk menghasilkan prediksi piksel.

Pada pendekatan instance segmentation berbasis FCN, urutan umumnya adalah:

1. Menghasilkan segmentasi semantik.
2. Memisahkan piksel objek sekelas menjadi instance yang berbeda.

Kelebihannya adalah prediksi dapat dilakukan sekali jalan dan batas mask dapat tajam. Tantangannya adalah memisahkan objek sekelas yang bersentuhan, sehingga diperlukan pengelompokan tambahan.

Contoh metode yang disebutkan pada slide:

- *Proposal-free Network for Instance-level Object Segmentation* — Liang dkk., 2015.
- *InstanceCut: from Edges to Instances with MultiCut* — Kirillov dkk., CVPR 2017.
- *Deep Watershed Transform for Instance Segmentation* — Bai dan Urtasun, CVPR 2017.

### 1.6. Pendekatan berbasis proposal

Pendekatan proposal-based memulai proses dengan kandidat lokasi objek, lalu menghasilkan segmentasi untuk setiap kandidat. Urutannya adalah **deteksi objek → segmentasi per objek**.

Konsep instance sudah ditentukan sejak awal, sehingga pendekatan ini membantu menangani objek yang saling menutupi. Namun, kualitas hasil bergantung pada proposal dan mask dibatasi oleh wilayah box.

**Simultaneous Detection and Segmentation (SDS).** Kandidat wilayah berupa box dan mask kasar dihasilkan oleh algoritma di luar neural network. Setiap kandidat diproses oleh dua CNN: Box CNN melihat isi box, sedangkan Region CNN melihat piksel dalam mask dengan latar dihapus. Fitur keduanya digabung untuk klasifikasi wilayah, kemudian mask kandidat diperhalus.

**Multi-task Network Cascades (MNC).** Segmentasi instance dipecah menjadi tiga tugas berurutan yang berbagi satu feature map konvolusi: menentukan box instance, menghasilkan mask instance, lalu menentukan kelas instance.

Materi juga menyebut *Hypercolumns for Object Segmentation and Fine-grained Localization* serta *Convolutional Feature Masking for Joint Object and Stuff Segmentation* sebagai bagian dari perkembangan pendekatan ini.

### 1.7. Faster R-CNN dan Mask R-CNN

Faster R-CNN mengekstraksi feature map dengan CNN, menghasilkan proposal melalui **Region Proposal Network (RPN)**, lalu memprediksi kelas dan koordinat bounding box untuk wilayah kandidat.

Mask R-CNN menambahkan cabang prediksi mask pada kerangka tersebut. Alur yang ditunjukkan pada materi adalah:

1. CNN mengekstraksi fitur citra dan RPN menghasilkan proposal objek.
2. Fitur setiap **Region of Interest (RoI)** diambil dengan **RoIAlign**.
3. Cabang deteksi memprediksi kelas dan bounding box.
4. Cabang mask yang menggunakan operasi konvolusi memprediksi mask untuk setiap kelas pada RoI.

Diagram menampilkan keluaran mask berukuran **C × 28 × 28**, dengan C sebagai jumlah kelas. Target mask saat training dibuat **per RoI**, yaitu mask ground truth objek dipotong sesuai box, lalu diubah ukurannya menjadi grid biner 28 × 28.

Resolusi mask menimbulkan kompromi antara detail bentuk objek dan biaya komputasi. Grid yang lebih kecil membutuhkan komputasi lebih sedikit, tetapi dapat kehilangan detail. Grid yang lebih besar dapat mempertahankan detail lebih banyak dengan biaya komputasi lebih tinggi.

### 1.8. Invariance, equivariance, dan RoIAlign

Untuk klasifikasi objek, representasi diharapkan bersifat **invariant**: label objek tetap sama walaupun posisi objek berubah. Untuk segmentasi, prediksi harus **equivariant**: ketika objek bergeser atau berubah skala, mask juga harus mengikuti pergeseran atau perubahan skala tersebut.

Materi menekankan penggunaan operasi yang mempertahankan informasi spasial pada cabang mask. RoIAlign digunakan untuk menyelaraskan fitur RoI dengan lebih tepat dibandingkan RoI pooling, sedangkan cabang mask menggunakan konvolusi. Diagram juga menunjukkan bahwa fully connected layer dan global pooling mendukung invariance pada klasifikasi.

Pada evaluasi berbasis piksel, objek kecil menyumbang lebih sedikit piksel. Instance segmentation menuntut perhatian pada setiap objek individual, termasuk objek berukuran kecil. Materi turut menunjukkan hasil segmentasi Mask R-CNN dan perluasannya untuk estimasi pose.

### 1.9. Kualitas mask dan Mask Scoring R-CNN

Confidence deteksi yang tinggi belum tentu berarti mask memiliki bentuk yang akurat. Materi membahas keterbatasan penggunaan skor deteksi sebagai skor kualitas mask, terutama ketika objek sekelas berdekatan atau saling menutupi.

**Mask Scoring R-CNN** menambahkan **MaskIoU head** untuk memperkirakan kualitas mask terhadap ground truth. Konsep pengukurannya adalah Intersection over Union (IoU):

```text
IoU = luas irisan mask prediksi dan ground truth
      ----------------------------------------
      luas gabungan mask prediksi dan ground truth
```

Skor mask kemudian mempertimbangkan perkiraan kualitas tersebut. Karena mask tidak selalu sempurna, skornya dapat lebih rendah dan lebih mencerminkan kualitas segmentasi dibandingkan skor deteksi saja. Contoh pada slide berasal dari *Mask Scoring R-CNN* — Huang dkk., CVPR 2019.

## 2. Ketentuan tugas Evaluasi CPMK 3

Bagian ini menuliskan kembali persyaratan pada halaman 29–32 PDF.

### 2.1. Dataset dan perbandingan model

- Gunakan **Panoptic Indoor Segmentation v5** sebagai data latih.
- Bandingkan **minimal 3 model segmentasi instance**.
- Model harus berasal dari **setidaknya 2 keluarga arsitektur berbeda**, misalnya one-stage dan two-stage, atau CNN dan transformer.
- Jika mengalami kesulitan akses GPU, tetapkan batas waktu training **maksimal 2 jam GPU Colab per model**, sesuai catatan pada slide.

Nama dataset mengikuti PDF. URL dataset, daftar kelas, dan pembagian datanya perlu dipastikan saat persiapan karena detail tersebut tidak tercantum dalam teks ketentuan tugas.

### 2.2. Test set buatan sendiri — wajib

1. Ambil **minimal 40 foto ruangan nyata** dengan kamera HP sendiri, misalnya kamar kos, rumah, laboratorium, atau ruang kelas di ITS.
2. Dari foto tersebut, **minimal 10 foto** harus diambil dalam kondisi sulit: cahaya redup, backlight, objek saling menutupi, atau sudut ekstrem.
3. Lakukan **anotasi sendiri di Roboflow** menggunakan kelas yang sama dengan dataset.
4. Lampirkan **screenshot proses anotasi** dan **EXIF/metadata foto**.
5. Laporkan performa **setiap model pada dua test set**: test set asli dan test set buatan sendiri.
6. Analisis perbedaan performa sebagai **domain shift**, yaitu perubahan karakteristik antara data asal dan foto ruangan yang diambil sendiri.

### 2.3. Variabel eksperimen berdasarkan NRP

Digit terakhir NRP menentukan variabel yang wajib diselidiki:

| Digit terakhir NRP | Variabel eksperimen wajib |
| --- | --- |
| 0–1 | Ukuran input: 320 / 640 / 960. |
| 2–3 | Jumlah data latih: 25% / 50% / 100%. |
| 4–5 | Dengan dibandingkan tanpa pretrained weights. |
| 6–7 | Augmentasi: tanpa / default / rancangan sendiri. |
| **8–9** | **Hanya 3 kelas tersulit berdasarkan analisis awal dibandingkan semua kelas.** |

**NRP 5054241018 berakhir dengan digit 8. Jadi, eksperimen wajib untuk tugas ini adalah membandingkan skenario hanya 3 kelas tersulit dengan skenario semua kelas.**

Tiga kelas terpilih adalah `cabinet-merged`, `blanket`, dan `rug-merged`. Pemilihan memakai rata-rata Mask AP validation dan mensyaratkan dukungan minimal lima gambar pada train serta validation. [Eksperimen NRP](05_eksperimen_hard3_colab.ipynb)

### 2.4. Analisis kegagalan berbasis bukti

1. Pilih **5 foto dari test set buatan sendiri** yang menunjukkan kegagalan model, misalnya objek tidak terdeteksi, mask bocor, salah kelas, atau satu objek terpecah.
2. Untuk **setiap foto**, jelaskan penyebab kegagalan dan kaitkan dengan mekanisme model. Contoh pada slide adalah resolusi prototype mask di YOLO atau grid 28 × 28 pada Mask R-CNN.
3. Uji penjelasan dengan **satu eksperimen kecil** agar analisis memiliki bukti dan tidak hanya berupa spekulasi.

### 2.5. Bukti proses

Lampirkan:

- **Log training asli**, misalnya `results.csv`, TensorBoard, atau Weights & Biases (W&B).
- **Catatan waktu penggunaan GPU**.
- **Logbook** yang mencatat kendala dan keputusan selama pengerjaan.

### 2.6. Penggunaan AI

AI diperbolehkan dengan ketentuan:

- Penggunaannya **wajib dideklarasikan**.
- Lampirkan **prompt utama** yang digunakan.
- Tunjukkan **minimal satu kasus nyata** ketika kode atau saran AI salah atau tidak cocok, lalu jelaskan cara memperbaikinya.

Penyusunan ringkasan README ini dibantu AI berdasarkan PDF yang disediakan. Prompt permintaan penyusunan:

> oke coba anda baca [10 - Segmentasi Instance dengan Deep Learning.pdf] nanti tolong tulis ulang apa materinya di readme dan apa tugasnya di readme juga, saya nrp 5054241018

Deklarasi penggunaan AI, tiga prompt utama, dan contoh perbaikan kode tersedia di [Analisis CPMK 3](ANALISIS_CPMK3.md).

### 2.7. Durasi pengerjaan

Durasi pengerjaan yang tercantum pada PDF adalah **2 minggu**. Tanggal mulai, tanggal pengumpulan, format pengumpulan, dan rubrik penilaian rinci tidak disebutkan pada halaman ketentuan tugas.

## 3. Checklist pengerjaan

- [x] Menyiapkan Panoptic Indoor Segmentation v5 sebagai data latih.
- [x] Memilih tiga model dari lebih dari satu keluarga arsitektur.
- [x] Mengambil 56 foto ruangan nyata dengan kamera HP, termasuk 17 foto kondisi sulit.
- [x] Menyiapkan dan mengaudit COCO foto pribadi dengan kelas training.
- [x] Melampirkan [screenshot proses anotasi di Roboflow](evidence/bukti-anotasi-roboflow.png).
- [x] Melatih ketiga model dan mengarsipkan notebook dengan output training.
- [x] Mengevaluasi test Roboflow dan foto mandiri serta menganalisis domain shift pada 31 kelas bersama.
- [x] Menentukan tiga kelas tersulit dari validation dan membandingkan skenario sesuai NRP 5054241018.
- [x] Menganalisis lima kegagalan dan menjalankan eksperimen confidence.
- [x] Menyertakan durasi GPU, [logbook](LOGBOOK.md), dan deklarasi penggunaan AI.
- [x] Menautkan log training ketiga model dan melampirkan [EXIF untuk 44 foto](evidence/metadata-foto-mandiri.csv).
- [ ] Melengkapi EXIF untuk 12 foto yang padanan berkas aslinya belum ditemukan jika tersedia.
- [ ] Menautkan catatan hipotesis sebelum training pertama jika tersedia.
- [x] Melaksanakan demo inferensi dan tanya jawab 10 menit bersama dosen. [Notebook demo](07_inferensi_foto_baru_colab.ipynb)

Rincian status dan bukti ada di [laporan analisis](ANALISIS_CPMK3.md).

## 4. Peta sumber dalam PDF

| Halaman PDF | Pokok bahasan |
| --- | --- |
| 1–3 | Identitas mata kuliah, CPMK 3, dan bahan kajian. |
| 4–9 | Jenis tugas visi komputer, aplikasi, serta latihan semantic, instance, dan panoptic segmentation. |
| 10–15 | Pendekatan FCN dan proposal-based, SDS, serta MNC. |
| 16–20 | Faster R-CNN, Mask R-CNN, target mask per RoI, dan kompromi resolusi terhadap komputasi. |
| 21–25 | Invariance, equivariance, RoIAlign, contoh segmentasi, dan estimasi pose. |
| 26–28 | Kualitas mask, MaskIoU head, dan Mask Scoring R-CNN. |
| 29 | Dataset, perbandingan model, test set mandiri, domain shift, dan catatan GPU. |
| 30 | Variabel eksperimen berdasarkan digit terakhir NRP. |
| 31 | Analisis kegagalan berbasis bukti. |
| 32 | Bukti proses, penggunaan AI, dan durasi pengerjaan. |

## 5. Alur notebook Colab

| Notebook | Runtime | Fungsi |
| --- | --- | --- |
| [01 · Training](01_instance_segmentation_kaggle.ipynb) | L4 GPU | Persiapan data, training `all_classes`, validation, dan checkpoint recovery. |
| [04 · Evaluasi v2](04_evaluasi_foto_pribadi_colab_v2.ipynb) | GPU | Evaluasi tiga checkpoint pada test Roboflow dan COCO foto pribadi v2. |
| [05 · Eksperimen NRP](05_eksperimen_hard3_colab.ipynb) | L4 GPU | Memilih tiga kelas dari validation dan membandingkan `hard3` dengan `all_classes`. |
| [06 · Analisis v2](06_analisis_foto_pribadi_colab_v2.ipynb) | CPU | Menghitung selisih pada kelas bersama, menampilkan lima studi kasus, dan menguji threshold. |

Notebook 04, 05, dan 06 memuat output run yang dilaporkan. Arsip run terdahulu tersimpan di `results/`.

| Bukti run | Isi |
| --- | --- |
| [Kaggle `run01`](results/segmentasi-instance-indoor-evaluasi-cpmk-3.ipynb) dan [laporan](results/reports/) | Baseline awal. CSV memuat hasil validation dan original test untuk YOLO11n dan Mask R-CNN. |
| [Colab `colab_quality01`, sesi pertama](results/colab/colab_quality01_session1.ipynb) | YOLO11s selesai training dan validation. Mask R-CNN tercatat selesai sampai epoch 13. |
| [Colab `colab_quality01`, sesi kedua](results/colab/colab_quality01_session2.ipynb) | YOLO11s dilewati, resume Mask R-CNN gagal, dan RF-DETR Seg Small selesai training serta validation. |
| [Mask R-CNN `colab_quality02`](results/colab/colab_quality02_maskrcnn.ipynb) | Training ulang yang selesai 30 epoch. Checkpoint ini dipakai Notebook 04 v2. |
| [Evaluasi `eval_personal56`](results/colab/eval_personal56_v1.ipynb) | Evaluasi awal, termasuk prediksi test Roboflow yang dirujuk oleh Notebook 05. |

`tests/` berisi uji regresi untuk memeriksa notebook utama. Folder `dataset/` dan `outputs/` dipakai untuk data dan hasil kerja lokal. Keduanya diabaikan Git. Output dari training Colab tersimpan di Drive.

### 5.1. Menjalankan training jika diperlukan

1. Buka notebook di Colab, pilih **L4 GPU** pada pengaturan runtime, lalu jalankan cell. Hubungkan Google Drive saat diminta. Jenis GPU aktual dicatat dalam `environment_train.json`. [FAQ Google Colab](https://research.google.com/colaboratory/faq.html)
2. Konfigurasi `RUNTIME = "colab"` membaca `RAW_DATASET` langsung dari `/content/drive/MyDrive/Computer_Vision_Tugas_CPMK3/dataset`. Setiap split `train`, `valid`, dan `test` harus berisi `_annotations.coco.json` dan gambar yang dirujuk. Cell 2.1 menampilkan path yang digunakan dan daftar file jika ada yang hilang. Notebook menyalin gambar tersebut ke disk Colab dan mempertahankan split 397/166/359 serta fingerprint baseline. Untuk menjalankan di laptop, gunakan `RUNTIME = "local"`.
3. Set `SMOKE_TEST = True` untuk menjalankan satu epoch dengan subset dataset. Setelah pipeline lolos, kembalikan `SMOKE_TEST = False`. Konfigurasi siap training memakai `MODE = "train"`, `MODEL = "all"`, `TRAIN_PROFILE = "quality"`, `RUN_LABEL = "colab_quality01"`, dan `EPOCHS = 30`. Model berjalan satu per satu pada subprocess terpisah. Untuk satu model, pilih `yolo11`, `maskrcnn`, atau `rfdetr`.
4. Default `FINAL_TEST = False` memberi hasil validation untuk pemilihan konfigurasi. Gunakan `MODE = "prepare"` untuk audit saja. Setelah checkpoint tersedia, evaluasi akhir dilakukan melalui Notebook 04.

### 5.2. Profil training dan kriteria perbandingan

| Model | `baseline` | `quality` |
| --- | --- | --- |
| YOLO11 | YOLO11n-seg, input 640, mosaic 1,0, mask ratio 4 | YOLO11s-seg, input 640, mosaic 0,4, mask ratio 2, patience 10 |
| Mask R-CNN | ResNet-50 FPN, input 480–640, StepLR | Arsitektur dan input yang sama, ReduceLROnPlateau berdasarkan Mask AP validation |
| RF-DETR | Seg Nano, input 312 | Seg Small, input 384, early stopping berdasarkan validation |

Profil `quality` menguji perubahan recipe untuk input indoor dan data latih terbatas. Bandingkan Mask AP50–95, AP per ukuran objek, AP per kelas, dukungan kelas train, dan waktu GPU pada validation. Untuk mengetahui efek suatu parameter, gunakan label run baru dan ubah satu parameter dari profil pembanding. RF-DETR memakai `RFDETR_AMP_DTYPE = "auto"`. Jika warning numerik berulang, `None` tersedia untuk percobaan FP32. L4 memberi ruang untuk eksperimen, sementara besar peningkatan AP perlu diukur pada run. [Parameter YOLO](https://docs.ultralytics.com/modes/train/) · [Model RF-DETR Segmentation](https://rfdetr.roboflow.com/latest/learn/run/segmentation/) · [Precision RF-DETR](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/#training-precision)

### 5.3. Artifact, checkpoint, dan resume

Root hasil pada Drive adalah `MyDrive/Computer_Vision_Tugas_CPMK3/outputs`.

| Lokasi relatif | Isi |
| --- | --- |
| `artifacts/<model>/<skenario>/<run>/` | Best checkpoint, checkpoint resume, konfigurasi, log model, masks COCO RLE, serta metrics dan per-class CSV. `console_log.txt` disimpan saat model dijalankan melalui runner berurutan. |
| `sequence/<skenario>/<run>/` | Status ketiga worker dan durasi per worker. |
| `reports/<skenario>/<run>/` | Audit, distribusi kelas, normalisasi, dan tabel perbandingan. |

Checkpoint training ditulis ke Drive saat epoch selesai. Resume memakai `last.pt` untuk YOLO, `last.pth` untuk Mask R-CNN, dan `native/last.ckpt` untuk RF-DETR. Jika `native/last.ckpt` gagal dimuat setelah runtime terputus saat menulis, RF-DETR juga menyimpan checkpoint periodik pada epoch kelipatan sepuluh.

Pada sesi baru, jalankan kembali `MODEL = "all"` dengan label dan konfigurasi yang sama untuk auto-resume dan melewati model dengan output lengkap. Atur `RESUME_CHECKPOINT` jika melanjutkan satu model secara eksplisit. Gunakan label baru untuk konfigurasi atau source worker yang berubah. Jalankan satu sesi per direktori run. `EXPORT_ZIPS = True` mengaktifkan arsip tambahan, sedangkan bagian 1.2–1.4 membuat recovery ZIP manual untuk direktori run yang dipilih.

Jika resume Mask R-CNN sebelumnya gagal dengan `RNG state must be a torch.ByteTensor`, buka notebook versi perbaikan dan jalankan dengan label serta konfigurasi lama. Notebook memulihkan state RNG CUDA dari tensor CPU dan menerima signature lama untuk perubahan ini.

### 5.4. Memakai checkpoint RF-DETR dari run Kaggle

Pada run Kaggle sebelumnya, YOLO dan Mask R-CNN selesai dievaluasi. RF-DETR mencapai 30 epoch dan menyimpan best checkpoint sebelum inferensi terhenti pada `KeyError: 74`. Adapter sekarang menyaring slot background sebelum menerapkan batas prediksi. Indeks foreground 73 tetap dipetakan ke ID kategori asli 74. [Source RF-DETR 1.12.1](https://github.com/roboflow/rf-detr/blob/1.12.1/src/rfdetr/_prediction/core.py)

Untuk memakai checkpoint RF-DETR lama di Colab, ekstrak recovery ZIP ke Drive dan letakkan `run_config.json` di samping `best.pth`. Sesuaikan `INFERENCE_CHECKPOINT` dengan direktori ekstraksi.

```python
MODE = "predict"
MODEL = "rfdetr"
RUN_LABEL = "run01_evalfix"
INFERENCE_CHECKPOINT = "/content/drive/MyDrive/Computer_Vision_Tugas_CPMK3/previous/rfdetr/run01/best.pth"
FINAL_TEST = True
ENABLE_PERSONAL = False
```

Notebook membaca resolusi dan asal run dari konfigurasi checkpoint. Langkah yang sama berlaku untuk mengevaluasi checkpoint terpilih dari model lain. Untuk YOLO gunakan `best.pt` dan untuk Mask R-CNN gunakan `best.pth`.

### 5.5. Menjalankan evaluasi dan analisis akhir

Root proyek di Drive adalah `MyDrive/Computer_Vision_Tugas_CPMK3/`. Foto disimpan di `foto-sendiri/foto-sendiri-final/` dan anotasi revisi di `annotation-results-v2/annotations.coco.json`. CSV `photo_metadata_56.csv` dapat disimpan bersama foto untuk laporan kondisi pengambilan.

1. **Notebook 04 v2** membaca checkpoint `colab_quality01` untuk YOLO dan RF-DETR, serta `colab_quality02` untuk Mask R-CNN. Run `eval_personal56_v2` menulis prediksi ketiga model dan metrik kedua test set.
2. **Notebook 05** memakai `per_class_valid.csv` dari training dan prediksi test Roboflow dari evaluasi `eval_personal56` versi awal. Jalankan hanya jika ingin mereproduksi training `hard3` dari awal atau melanjutkan run yang belum selesai.
3. **Notebook 06 v2** memakai prediksi `eval_personal56_v2`, serta `analysis_config.json` dan `failure_case_notes.csv` dari laporan `analisis_personal56_v2` sebelumnya. Ini mempertahankan empat kasus lama dan mengganti kasus keempat. Run menghasilkan laporan `analisis_personal56_v2_kasus_revisi`.

Tabel dan lima panel yang digunakan untuk pengumpulan tersedia di [laporan analisis](ANALISIS_CPMK3.md). Uji regresi Notebook 01 dapat dijalankan dengan `python -m unittest discover -s tests -v` setelah NumPy terpasang.
