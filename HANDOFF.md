# Handoff — Evaluasi CPMK 3: Instance Segmentation

Tanggal pemeriksaan: **18 November 2026**. Pemilik: **Muhammad Farhan**, NRP **5054241018**.

## 1. Tujuan dan ruang lingkup

Membandingkan minimal tiga model instance segmentation dari setidaknya dua keluarga arsitektur menggunakan **Panoptic Indoor Segmentation v5** dari Roboflow. Ketentuan tugas dirangkum dalam [README.md](README.md), berdasarkan [PDF materi dan tugas](<10 - Segmentasi Instance dengan Deep Learning.pdf>).

Target tugas lengkap juga mencakup eksperimen **semua kelas vs tiga kelas tersulit** karena digit terakhir NRP adalah 8; evaluasi pada test set asli serta minimal 40 foto HP sendiri, termasuk minimal 10 foto kondisi sulit; anotasi manual; analisis domain shift; lima contoh kegagalan dan eksperimen kecil; log training, waktu GPU, logbook, serta deklarasi penggunaan AI.

**Fokus yang disepakati saat ini adalah dataset Roboflow dan baseline tiga model pada semua kelas.** Foto HP belum diambil. Tahap personal dataset dan eksperimen hard3 belum dilaksanakan, walaupun section pendukungnya sudah tersedia di notebook.

## 2. Status repository yang diverifikasi

Repository aktual berada satu tingkat di bawah direktori workspace:

```text
C:\Kuliah ITS Farhan\Semester 5\Computer Vision\Tugas-CPMK3\
└── Computer_Vision_Tugas-Evaluasi-CPMK-3-\
```

| Pemeriksaan | Hasil saat handoff dibuat |
| --- | --- |
| Branch | `main`, tracking `origin/main` |
| HEAD | `3ee3eecae2dd4a4b31b9bfe547a302090e4b42fe` — `fix bug` |
| Perubahan tracked sebelum handoff | Tidak ada; staged diff juga kosong |
| File untracked sebelum handoff | `DEMO_PROGRESS.md` |
| Data lokal | `dataset/` tersedia dan diabaikan Git |
| Hasil lokal | `outputs/` hanya berisi direktori; tidak ditemukan file hasil |
| Output notebook tersimpan | Tidak ada output pada seluruh 73 code cell |

Status branch tersebut berdasarkan referensi lokal; **tidak dilakukan fetch atau pemeriksaan langsung remote**. Pembuatan handoff hanya menambahkan file ini, tanpa perubahan kode atau commit. `DEMO_PROGRESS.md` sudah ada sebelum pekerjaan ini.

## 3. Pekerjaan yang sudah diimplementasikan

Implementasi utama berada dalam [01_instance_segmentation_kaggle.ipynb](01_instance_segmentation_kaggle.ipynb). File ini memiliki **174 cell**, termasuk **73 code cell**, dengan maksimum **35 baris per code cell**. Code cell didahului penjelasan Markdown bernomor, berbahasa Indonesia formal dengan istilah teknis dipertahankan.

Yang sudah tersedia dalam source notebook:

- Dataset audit, class mapping, distribusi kelas, ground-truth overlay, serta salinan data kerja terpisah dari raw dataset.
- Koreksi empat image yang orientasinya tidak sesuai dimensi anotasi; normalisasi `area` dan `bbox` berdasarkan polygon mask; annotation ID dinomori ulang mulai 1.
- Konversi COCO Segmentation ke YOLO Segmentation dengan mapping kategori yang dapat ditelusuri.
- Training adapter untuk YOLO11n-seg, Mask R-CNN ResNet-50 FPN, dan RF-DETR Seg Nano.
- Sequential runner **YOLO11 → Mask R-CNN → RF-DETR**, dengan subprocess dan virtual environment masing-masing model.
- Pemilihan best checkpoint menggunakan validation mask AP, ekspor predictions COCO RLE, evaluasi validation dan original test dengan evaluator bersama, serta penggabungan metrics.
- Checkpoint per epoch, auto-resume, skip completed run, pemulihan backup dari Input, pencatatan status dan console log, serta recovery ZIP yang tidak memerlukan seluruh model selesai.
- Section pemilihan hard3 dari validation, training scenario hard3, personal test set, domain shift, failure analysis, diagnostic brightness experiment, dan pengarsipan laporan.

**Ketersediaan source bukan bukti seluruh alur sudah berhasil dijalankan.** Section hard3, personal dataset, dan analisis kegagalan masih merupakan fasilitas implementasi untuk tahap berikutnya.

Perbaikan yang sudah tampak pada source saat ini:

- Pembuatan virtual environment memakai `with_pip=False` dan akses ke parent site-packages melalui `.pth`, lalu memeriksa Python dan pip. Ini menghindari kegagalan `ensurepip` yang sebelumnya menghentikan ketiga worker.
- COCO evaluator menerima `copy.deepcopy(predictions)` untuk menghindari perubahan objek predictions oleh `loadRes`.
- Pemuatan ulang Mask R-CNN mempertahankan arsitektur `FrozenBatchNorm2d` yang sesuai checkpoint.
- Penulisan JSON dan salinan checkpoint tertentu memakai file sementara dan penggantian atomik.

## 4. Data yang diverifikasi dari file lokal

Raw dataset berformat COCO Segmentation, dengan setiap split berisi image dan `_annotations.coco.json`.

| Split | Image dalam JSON / file image | Annotation | Kelas dengan annotation |
| --- | --- | --- | --- |
| Train | 397 / 397 | 4.479 | 65 |
| Validation | 166 / 166 | 2.022 | 58 |
| Test | 359 / 359 | 5.344 | 69 |
| Total | 922 / 922 | 11.845 | — |

Setiap JSON memuat 75 kategori. Kategori induk `indoor-objects` tidak memiliki annotation dan dikeluarkan dari mapping model, sehingga scenario `all_classes` menggunakan **74 kelas**.

Keputusan: **pertahankan original split** sekitar 43% train, 18% validation, dan 39% test. Jumlah test yang hampir sama dengan train berasal dari dataset sumber; jangan memasukkan original test ke training untuk memperbesar data setelah hasil test dilihat.

Temuan yang masih relevan:

- Sebanyak **24 kelas yang muncul pada train** memiliki kurang dari 10 instance.
- Sembilan kelas segmentasi tidak muncul pada train: `flower`, `hair drier`, `keyboard`, `mouse`, `scissor`, `scissors`, `teddy bear`, `toaster`, dan `toothbrush`.
- Pemeriksaan ulang exact duplicate annotation, dengan mengabaikan field `id`, menemukan **0 tambahan pada train, 1 pada validation, dan 3 pada test**. Fungsi normalisasi saat ini **belum melakukan deduplication**.
- Jumlah annotation pada evaluator native YOLO dapat berbeda dari COCO karena perilaku pengolahan duplicate. Jangan langsung membandingkan angka native antar-framework sebagai hasil akhir.
- Koreksi orientasi yang sudah dikodekan berlaku pada empat nama file yang tercantum dalam `ROTATE_CLOCKWISE`. Pemeriksaan visual ulang terhadap image dan mask tidak dilakukan pada sesi pembuatan handoff ini.

## 5. Keputusan teknis dan konfigurasi aktif

| Model | Keluarga | Input / batch default |
| --- | --- | --- |
| YOLO11n-seg | CNN one-stage | 640 / 4 |
| Mask R-CNN ResNet-50 FPN | CNN two-stage | Short side 480, long side maksimal 640 / 2 |
| RF-DETR Seg Nano | Transformer | 312 / 2 |

Ketiganya memakai pretrained weights. Pilihan ini memenuhi variasi keluarga arsitektur; belum ada bukti bahwa kombinasi ini merupakan pilihan terbaik dari seluruh model yang tersedia. Resolusi dan recipe training antar-model berbeda, sehingga perbandingan perlu menjelaskan konfigurasi dan biaya komputasi tersebut.

Konfigurasi yang tersimpan:

```python
MODE = "train"
MODEL = "all"
MODEL_ORDER = ["yolo11", "maskrcnn", "rfdetr"]
SCENARIO = "all_classes"
EPOCHS = 30
SEED = 42
RUN_LABEL = "run01"
SMOKE_TEST = False
FINAL_TEST = True
ENABLE_PERSONAL = False
SKIP_COMPLETED = True
AUTO_RESUME = True
CONTINUE_ON_ERROR = True
RAW_DATASET = "/kaggle/input/datasets/mrmfrhann234/dataset-panoptic"
```

- Pengembangan dilakukan lokal melalui Antigravity, kemudian **upload notebook ke Kaggle** untuk GPU. Aktifkan GPU dan Internet; dataset ditambahkan sebagai Kaggle Input dengan struktur `train`, `valid`, `test`.
- Paket model dipin ke `ultralytics==8.4.215` dan `rfdetr[train]==1.12.1`. Paket dasar seperti PyTorch menggunakan environment Kaggle yang tersedia; environment lengkap belum dikunci dalam lockfile repository.
- Evaluasi bersama memakai `COCOeval(..., "segm")`, `SCORE_FLOOR = 0.001`, dan `MAX_DETECTIONS = 100`. Bandingkan saved metrics dari protokol yang sama, split yang sama, dan checkpoint terpilih.
- Hard3 dipilih dari **rata-rata AP validation ketiga baseline**, dengan syarat minimal 5 image train dan 5 image validation per kelas. Daftar dikunci berdasarkan identitas dataset; original test tidak dipakai memilih kelas.
- Signature run mencakup konfigurasi dan source worker. Perubahan source atau parameter dapat memerlukan `RUN_LABEL` baru; jangan mengasumsikan checkpoint lama otomatis kompatibel.
- Ordinary error dapat diteruskan ke model berikutnya; `KeyboardInterrupt` menghentikan sequence. Jika ketiga worker gagal, runner baru melempar error setelah semuanya dicoba.

## 6. Training yang teramati dari log Kaggle

Sumber: log yang diberikan pengguna, bukan eksekusi GPU lokal atau akses langsung ke sesi Kaggle:

```text
C:\Users\Farhan\.codex\attachments\c81b55c4-f3b2-46eb-9c0a-3dfef4c9186b\Pasted text.txt
```

| Model | Bukti dalam log terakhir | Batas kepastian |
| --- | --- | --- |
| YOLO11 | Training 30 epoch selesai; proses evaluasi validation dan original test selesai sebelum worker Mask R-CNN dimulai | Checkpoint, metrics JSON, dan completion marker belum diunduh untuk diperiksa lokal |
| Mask R-CNN | Training 30 epoch dan proses evaluasi selesai sebelum worker RF-DETR dimulai | Artifact final belum diperiksa lokal |
| RF-DETR | Validation EMA sudah mencapai epoch 25/30 | Log berhenti pada warning; penyelesaian epoch berikutnya dan run belum diketahui |

Angka sementara yang pernah teramati: native validation YOLO menunjukkan mask AP50–95 sekitar **0,145**, sedangkan validation Mask R-CNN terbaik dalam log sekitar **0,1608 pada epoch 14**. Angka ini **belum menjadi tabel perbandingan final** karena provenance evaluator/checkpoint harus disamakan dari artifact. Nilai AP bukan persentase image yang seluruh prediksinya benar.

Warning terakhir RF-DETR:

```text
Non-finite values detected in matcher cost matrix;
replacing with finite sentinel. Check for numerical instability.
```

Ini menunjukkan NaN/Inf pada matching cost. Penggantian nilai oleh library belum membuktikan loss dan weights tetap sehat. Belum ada bukti dalam log yang dikirim bahwa proses pulih setelah warning tersebut. Mixed precision FP16 pada T4 merupakan kemungkinan pemicu, **belum merupakan diagnosis terkonfirmasi**. Source notebook saat ini belum menetapkan `amp_dtype` secara eksplisit dan belum memiliki perbaikan khusus warning ini.

Log juga menampilkan pesan pip dependency conflict, tetapi detail paket tidak terlihat pada salinan output notebook. RF-DETR tetap sempat berjalan hingga epoch 25; hal itu tidak membuktikan seluruh dependency kompatibel. Periksa full `console_log.txt` sebelum menyimpulkan penyebab.

Pesan `compute_val_loss='auto'` hanya menyatakan validation loss tidak dihitung untuk logging; validation metrics tetap dijalankan. Peringatan direktori checkpoint sudah berisi file sesuai dengan source yang menulis `native/run_config.json` sebelum training.

## 7. Lokasi hasil dan mekanisme recovery

```text
/kaggle/working/cpmk3/
├── prepared/
├── artifacts/<model>/<scenario>/<run>/
└── reports/

/kaggle/working/cpmk3/sequence/all_classes/run01/sequence_status.json
/kaggle/working/recovery_<model>_<scenario>_<run>.zip
```

Artifact yang perlu disimpan dan diperiksa:

- Best weights: `best.pt` untuk YOLO; `best.pth` untuk Mask R-CNN dan RF-DETR.
- Resume checkpoint: `last.pt`, `last.pth`, atau RF-DETR **`native/last.ckpt`**. Root `last.ckpt` RF-DETR baru disalin setelah fungsi training selesai.
- `run_config.json`, `job_signature.json`, `worker_config.json`, `worker.py`, `console_log.txt`, `training_sessions.json`, log native, dan `completion.json` jika worker lengkap.
- `metrics_valid.json`, `per_class_valid.csv`, `metrics_test.json`, `per_class_test.csv`, serta predictions `.json.gz`.

Checkpoint resume diperbarui setelah epoch lengkap; progres di tengah epoch belum tentu tersimpan. Recovery ZIP dicoba pada penyelesaian, error, atau interupsi Python, dan tersedia manual export pada section 1.2–1.4. Pada kernel baru, jalankan Runtime cell registry dahulu; untuk export gunakan `EXPORT_RECOVERY_NOW = True` dan tentukan `RECOVERY_RUN`.

**File sesi aktif belum merupakan backup permanen.** Unduh archive/checkpoint atau simpan versi Kaggle dengan output. Hard kill kernel/VM tidak menjamin blok `finally` berjalan. Jangan menghentikan seluruh sesi hanya untuk mengekspor progres.

Untuk melanjutkan sesi lain, ekstrak recovery archive dan tambahkan sebagai Input, pertahankan konfigurasi/signature, lalu batasi `PREVIOUS_RESULTS` ke sumber yang benar bila ada beberapa backup. Gunakan `MODEL = "rfdetr"` apabila hanya RF-DETR yang memerlukan penanganan; jangan mengulang baseline yang sudah lengkap tanpa alasan eksperimen.

## 8. File penting dan status perubahannya

| File | Peran dan status aktual |
| --- | --- |
| [01_instance_segmentation_kaggle.ipynb](01_instance_segmentation_kaggle.ipynb) | Seluruh pipeline; tracked dan tidak diubah pada sesi handoff. Commit HEAD `fix bug` mencakup perbaikan notebook dan README. |
| [README.md](README.md) | Materi, ketentuan tugas, panduan training dan recovery; tracked dan tidak diubah pada sesi ini. Checklist masih seluruhnya kosong dan belum mencerminkan progres implementasi/training terbaru. |
| [DEMO_PROGRESS.md](DEMO_PROGRESS.md) | Naskah demo 4–5 menit; sudah ada tetapi untracked. Deskripsi training masih umum dan perlu disesuaikan dengan hasil terbaru sebelum dipakai. |
| [.gitignore](.gitignore) | Mengabaikan `dataset/`, `outputs/`, dan `.ipynb_checkpoints/`; tracked dan tidak diubah. |
| [PDF tugas](<10 - Segmentasi Instance dengan Deep Learning.pdf>) | Sumber materi dan persyaratan; tracked dan tidak diubah. Tidak dibaca ulang pada sesi handoff. |
| `dataset/` | Raw dataset tersedia lokal, ignored; tidak diubah. |
| `outputs/` | Struktur direktori lokal tanpa file artifact training; ignored melalui aturan `.gitignore`. |
| `HANDOFF.md` | Satu-satunya file yang ditambahkan pada sesi ini. |

Ada builder dan skrip pengujian historis **di luar repository**, dalam `C:\Users\Farhan\AppData\Local\Temp\`: `build_cpmk3_notebook.py`, `test_cpmk3_notebook.py`, `test_cpmk3_sequence.py`, dan `test_cpmk3_kernel.py`. Keberadaannya diperiksa, tetapi kesesuaian dan hasil eksekusinya tidak diverifikasi ulang pada sesi ini. Jangan mengandalkan file temporary tersebut tersedia di mesin lain atau menjalankan builder tanpa memeriksa apakah isinya sesuai notebook terbaru.

## 9. Verifikasi pada sesi handoff dan batasnya

Dilakukan pemeriksaan Git status, HEAD, tracked files, staged/unstaged diff, isi notebook dan dokumentasi, jumlah file/image/annotation/kategori dataset, serta exact duplicate annotation. Seluruh **73 code cell lolos parsing Python AST** menggunakan Python di WSL Ubuntu. Notebook tidak dieksekusi untuk training, dependency tidak diinstal, dan koneksi langsung ke GPU/sesi Kaggle tidak dilakukan.

Parsing hanya membuktikan syntax valid, bukan kompatibilitas framework, kapasitas GPU, kualitas training, atau keberhasilan recovery end-to-end. Hasil pengujian historis tidak dianggap sebagai pengujian ulang source saat ini.

## 10. Masalah terbuka dan langkah berikutnya

1. **Pastikan kelanjutan RF-DETR.** Ambil log setelah epoch 25 dan periksa loss/metrics serta pengulangan warning non-finite. Jika berulang atau muncul NaN/Inf, interupsi cell, ekspor recovery, dan periksa checkpoint sehat sebelum menentukan perubahan precision atau learning rate. Belum ada perubahan recipe untuk masalah ini.
2. **Amankan output dua model yang sudah selesai.** Unduh checkpoint, full console log, metrics, predictions, konfigurasi, dan status sequence. Periksa keberadaan serta isi artifact; jangan menganggap log progress saja membuktikan seluruh file ekspor lengkap.
3. **Bangun tabel baseline dari evaluator bersama.** Kumpulkan `metrics_valid.json`, `metrics_test.json`, per-class CSV, waktu GPU, dan identitas run. Angka RF-DETR final serta perbandingan tiga model belum tersedia lokal.
4. **Investigasi kualitas baseline.** Periksa distribusi kelas, per-class AP, kurva train/validation dan overlay prediksi. AP yang masih rendah belum dapat dijelaskan oleh satu penyebab pasti. Perubahan epoch, augmentation, kapasitas model, atau recipe perlu menjadi eksperimen terpisah dengan label run baru; jangan memilih konfigurasi berdasarkan original test.
5. **Tentukan kebijakan duplicate annotation secara konsisten.** Normalisasi belum menghapus duplicate. Jangan mengubah data baseline diam-diam ketika training masih berjalan; jika memperbaikinya, dokumentasikan identitas dataset baru dan ulangi evaluasi yang diperlukan secara konsisten.
6. Setelah tiga baseline valid tersedia, jalankan `MODE = "select_classes"`, simpan ranking dan `hard3_classes.json`, lalu latih ketiga model dengan `SCENARIO = "hard3"` memakai tiga kelas yang sama. Bandingkan dengan hasil semua kelas pada kelas evaluasi yang sama.
7. Ambil dan anotasi minimal 40 foto HP, termasuk 10 kondisi sulit, serta simpan metadata dan screenshot anotasi. Baru kemudian isi path personal dataset, aktifkan `ENABLE_PERSONAL`, dan evaluasi checkpoint pada domain baru.
8. Lengkapi lima contoh kegagalan, penjelasan mekanisme dan eksperimen kecil berbasis bukti; logbook, waktu GPU, deklarasi AI dan kasus kesalahan AI nyata. Kegagalan `ensurepip` merupakan kandidat kasus terdokumentasi, tetapi narasi buktinya belum dilengkapi.
9. Perbarui checklist README, naskah demo, dan laporan berdasarkan hasil yang sudah diverifikasi. Penyelesaian baseline tiga model belum sama dengan penyelesaian seluruh tugas CPMK 3.

Tidak ada commit yang dibuat dalam sesi penyusunan handoff ini.
