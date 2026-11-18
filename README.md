# Evaluasi CPMK 3 — Segmentasi Instance dengan Deep Learning

| Identitas | Keterangan |
| --- | --- |
| NRP | **5054241018** |
| Mata kuliah | Visi Komputer (EA234503) |
| Program studi | Rekayasa Kecerdasan Artifisial, ITS |
| Dosen pada materi | Anny Yuniarti |
| Sumber | [10 - Segmentasi Instance dengan Deep Learning.pdf](<10 - Segmentasi Instance dengan Deep Learning.pdf>) |

README ini menuliskan kembali materi perkuliahan dan ketentuan tugas dari PDF. Ringkasan materi mengacu pada halaman 1–28, sedangkan ketentuan Evaluasi CPMK 3 mengacu pada halaman 29–32. Dokumen ini merupakan catatan materi dan panduan pengerjaan; hasil eksperimen belum dicantumkan.

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

Nama ketiga kelas belum dapat ditentukan sebelum analisis awal dilakukan. Pemilihannya harus disertai bukti analisis, bukan berdasarkan perkiraan semata.

Sebagai panduan pelaksanaan tambahan, analisis awal dapat menggunakan performa per kelas pada data validasi untuk memilih tiga kelas tersulit. Catat dasar pemilihan kelas dan perbedaan konfigurasi kedua skenario agar perbandingannya dapat ditelusuri. Cara pemilihan ini adalah saran pengerjaan; slide hanya mewajibkan pemilihan dari analisis awal.

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

Deklarasi dan lampiran prompt perlu dilengkapi sesuai penggunaan AI selama implementasi. Kasus kesalahan atau ketidakcocokan saran AI beserta perbaikannya perlu didokumentasikan dari kejadian yang benar-benar terjadi.

### 2.7. Durasi pengerjaan

Durasi pengerjaan yang tercantum pada PDF adalah **2 minggu**. Tanggal mulai, tanggal pengumpulan, format pengumpulan, dan rubrik penilaian rinci tidak disebutkan pada halaman ketentuan tugas.

## 3. Checklist pengerjaan

Checklist berikut merangkum kewajiban tugas. Semua kotak masih kosong karena README ini belum melaporkan penyelesaian eksperimen.

- [ ] Menyiapkan Panoptic Indoor Segmentation v5 sebagai data latih.
- [ ] Memilih minimal 3 model dari setidaknya 2 keluarga arsitektur.
- [ ] Mengambil minimal 40 foto ruangan nyata dengan kamera HP sendiri.
- [ ] Memastikan minimal 10 foto diambil dalam kondisi sulit.
- [ ] Menganotasi foto sendiri di Roboflow dengan kelas yang sama dengan dataset.
- [ ] Menyimpan screenshot proses anotasi dan EXIF/metadata foto.
- [ ] Melatih model dan menyimpan log training asli.
- [ ] Mengevaluasi setiap model pada test set asli dan test set buatan sendiri.
- [ ] Menganalisis selisih performa atau domain shift.
- [ ] Menentukan 3 kelas tersulit berdasarkan analisis awal dan mencatat buktinya.
- [ ] Membandingkan hanya 3 kelas tersulit dengan semua kelas sesuai NRP 5054241018.
- [ ] Memilih 5 foto kegagalan dari test set buatan sendiri.
- [ ] Menjelaskan kegagalan setiap foto dengan mengaitkannya pada mekanisme model.
- [ ] Menguji penjelasan kegagalan melalui eksperimen kecil dan mencatat hasilnya.
- [ ] Menyertakan catatan waktu GPU dan logbook kendala serta keputusan.
- [ ] Mendeklarasikan penggunaan AI dan melampirkan prompt utama.
- [ ] Mendokumentasikan minimal satu kesalahan atau saran AI yang tidak cocok serta perbaikannya.

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

## 5. Notebook Kaggle

Seluruh alur implementasi berada di [01_instance_segmentation_kaggle.ipynb](01_instance_segmentation_kaggle.ipynb): dataset preparation, training, eksperimen NRP, evaluation, dan analisis. Setiap code cell didahului Markdown bernomor hierarkis, misalnya `1.1`, dengan penjelasan formal dan istilah teknis yang dipertahankan. Model yang disiapkan adalah **YOLO11n-seg**, **Mask R-CNN ResNet-50 FPN**, dan **RF-DETR Seg Nano**.

**Fokus pengerjaan sekarang adalah dataset Roboflow dan perbandingan ketiga model pada semua kelas. Foto HP belum diambil.** Pengambilan foto, anotasi mandiri, dan analisis domain shift dikerjakan pada tahap berikutnya. Biarkan `ENABLE_PERSONAL = False`; tahap saat ini menggunakan 397 gambar train, 166 gambar validasi, dan 359 gambar test asli.

### 5.1. Menjalankan notebook

1. Upload isi folder `dataset/` sebagai Kaggle Dataset, pertahankan direktori `train`, `valid`, dan `test`, lalu tambahkan sebagai Input notebook.
2. Upload file notebook dan aktifkan Internet untuk instalasi paket serta pengunduhan pretrained weights.
   Path dataset yang sudah disetel sesuai Input Kaggle: `RAW_DATASET = "/kaggle/input/datasets/mrmfrhann234/dataset-panoptic"`. Folder ini langsung berisi `train`, `valid`, dan `test`. Jika nama atau lokasi Input berubah, sesuaikan nilai tersebut; kosongkan untuk memakai pencarian otomatis.
3. Konfigurasi default sudah siap untuk training: `MODE = "train"`, `MODEL = "all"`, `SCENARIO = "all_classes"`, `EPOCHS = 30`, `SMOKE_TEST = False`, `FINAL_TEST = True`, dan `ENABLE_PERSONAL = False`.
4. Aktifkan GPU, lalu jalankan **Run All**. Dataset audit dan orientation correction dilakukan sebelum sequence training. Gunakan `MODE = "prepare"` bila hanya memerlukan audit; gunakan `SMOKE_TEST = True` untuk sequence dengan subset kecil dan satu epoch.
5. Runner menjalankan **YOLO11 → Mask R-CNN → RF-DETR** secara berurutan. Setiap model menggunakan subprocess terpisah agar CUDA context dilepas ketika worker selesai. Dependencies tambahan diinstal pada virtual environment per model dengan akses ke paket bawaan Kaggle. Environment dibuat dengan `with_pip=False` untuk menghindari kegagalan `ensurepip`; modul `pip` diakses dari parent site-packages. Environment yang belum lengkap dikonfigurasi ulang, lalu Python dan pip diperiksa sebelum worker berjalan.
6. Best checkpoint dipilih dari validation metrics. Setelah training, worker mengekspor validation predictions dan menjalankan test evaluation pada dataset asli. Foto HP dan daftar hard3 tidak diperlukan untuk tahap ini.
7. Notebook menggabungkan metrics dari seluruh worker yang berhasil. Status sequence tersedia pada `/kaggle/working/cpmk3/sequence/all_classes/run01/sequence_status.json` dan console log masing-masing model pada direktori artifact.

`CONTINUE_ON_ERROR = True` membuat ordinary error pada satu model dicatat tanpa membatalkan model berikutnya. KeyboardInterrupt menghentikan seluruh sequence. Jika semua worker gagal, notebook menampilkan error setelah ketiganya dicoba. `SKIP_COMPLETED = True` melewati model yang output dan configuration signature-nya sesuai. `AUTO_RESUME = True` menggunakan checkpoint optimizer yang tersedia; jika training sudah selesai dan evaluation gagal, worker menjalankan inference ulang.

Untuk melanjutkan sesi sebelumnya, tambahkan output atau recovery archive yang sudah diekstrak sebagai Input. Runner memulihkan direktori artifact yang memiliki signature sesuai ke direktori kerja. Jika terdapat beberapa backup untuk run yang sama, batasi `PREVIOUS_RESULTS` ke satu sumber yang dipilih. Parameter atau source code yang berubah memerlukan `RUN_LABEL` baru. Gunakan `MODEL = "yolo11"`, `"maskrcnn"`, atau `"rfdetr"` apabila hanya satu model yang perlu dijalankan.

Tahap berikutnya: gunakan `MODE = "select_classes"` untuk menghasilkan `hard3_classes.json` dari rata-rata AP validasi ketiga model. Ulangi training menggunakan `SCENARIO = "hard3"` dengan daftar kelas yang sama. Setelah foto sendiri dan anotasinya tersedia, aktifkan `ENABLE_PERSONAL = True`, isi lokasi datanya, lalu jalankan inferensi checkpoint yang sudah tersimpan. Section kegagalan menyediakan visualisasi lima foto, tabel penjelasan berdasarkan bukti, serta `MODE = "diagnose"` untuk eksperimen kecil. Bagian foto sendiri dan domain shift dilewati selama flag tersebut tidak aktif.

Gunakan label run yang sama untuk pasangan skenario yang dibandingkan, misalnya `run01`; direktori model dan skenario sudah terpisah. Gunakan label baru untuk percobaan tambahan. Pertahankan `run_config.json` di samping checkpoint. Jika ada beberapa baseline berbeda untuk satu model, batasi `PREVIOUS_RESULTS` ke folder run yang dipilih.

### 5.2. Data dan hasil yang disimpan

Notebook mempertahankan dataset sumber dan membuat salinan kerja di `/kaggle/working/cpmk3/prepared/`. Koreksi orientasi, perhitungan luas mask, dan penomoran ulang ID anotasi dicatat pada laporan normalisasi. Identitas dataset menyertakan anotasi dan hash gambar.

Hasil model disimpan di `/kaggle/working/cpmk3/artifacts/<model>/<skenario>/<run>/`, meliputi best checkpoint, resume checkpoint, konfigurasi, log asli, waktu training, dan mask predictions dalam COCO RLE terkompresi. Direktori ini juga memuat `console_log.txt`, worker script, configuration signature, serta completion marker jika seluruh output model tersedia. Laporan berada di `/kaggle/working/cpmk3/reports/`. Arsip ZIP model dan laporan tersedia di `/kaggle/working/` untuk diunduh bersama output notebook.

Untuk tahap test foto sendiri nanti, siapkan ekspor COCO, `photo_metadata.csv`, bukti metadata, dan folder `annotation_screenshots` sesuai petunjuk di notebook. Saat `ENABLE_PERSONAL = False`, notebook melewati pembacaan foto dan bukti tersebut. Analisis penyebab kegagalan, kesimpulan eksperimen kecil, logbook, dan kasus kesalahan AI dilengkapi berdasarkan proses yang benar-benar terjadi.

Notebook belum memuat hasil training. Kompatibilitas framework, kapasitas batch, dan waktu training tetap perlu diuji pada GPU Kaggle yang diperoleh.

### 5.3. Menyelamatkan progres training

Checkpoint resume disimpan setiap akhir epoch lengkap: `last.pt` untuk YOLO, `last.pth` untuk Mask R-CNN, dan `native/last.ckpt` untuk RF-DETR. Salinan YOLO dan Mask R-CNN ditulis melalui file sementara, kemudian mengganti file terakhir secara atomik agar kegagalan penulisan tidak langsung merusak checkpoint sebelumnya.

Saat cell training selesai, gagal, atau menerima interupsi Python, notebook mencoba membuat `recovery_<model>_<skenario>_<run>.zip` di `/kaggle/working/` melalui blok `finally`. Arsip berisi checkpoint, konfigurasi, dan log yang sudah tersedia; proses ini tidak menunggu inferensi atau seluruh section notebook selesai.

Untuk manual export, interupsi cell training tanpa menghentikan sesi Kaggle. Jalankan tiga cell pada bagian **Ekspor progres tanpa menunggu seluruh notebook**, ubah `EXPORT_RECOVERY_NOW = True`, lalu isi `RECOVERY_RUN` dengan direktori run atau gunakan run terbaru. Pada kernel baru, jalankan **Runtime cell registry** terlebih dahulu. Cell ini tidak membutuhkan instalasi framework maupun pembacaan dataset. Unduh arsip dari panel file/output. Jangan menjalankan ulang semua cell hanya untuk melakukan export.

File pada disk sesi aktif belum menjadi jaminan penyimpanan permanen. Unduh archive/checkpoint ke laptop atau simpan versi notebook dengan output. Jika kernel atau VM dimatikan paksa, blok `finally` tidak dapat dijamin berjalan; progres yang belum masuk checkpoint dan file sesi yang belum disalin dapat hilang. Untuk melanjutkan sequence, upload archive yang sudah diekstrak sebagai Input dan pertahankan configuration/signature files. Untuk single-model mode, `RESUME_CHECKPOINT` dapat diisi secara eksplisit.
