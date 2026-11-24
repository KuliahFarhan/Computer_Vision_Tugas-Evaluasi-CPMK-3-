# Naskah Demo Progress Evaluasi CPMK 3

Target durasi: 4–5 menit. Arahan tampilan dalam tanda kurung siku tidak perlu dibacakan. Posisi pekerjaan: persiapan pipeline selesai dan proses training berlangsung di Kaggle; hasil akhir belum dilaporkan.

## 0:00–0:35 — Tujuan dan cakupan pekerjaan

[Tampilkan README bagian tujuan tugas dan identitas NRP.]

Pada demo ini, saya akan menjelaskan progress pengerjaan Evaluasi CPMK 3 untuk mata kuliah Visi Komputer, dengan NRP 5054241018. Tujuan tugas ini adalah membandingkan minimal tiga model instance segmentation dari setidaknya dua keluarga arsitektur.

Fokus pekerjaan saat ini adalah menyiapkan dataset dari Roboflow dan menjalankan training ketiga model. Pengambilan serta anotasi foto ruangan menggunakan kamera HP belum dilakukan dan akan dikerjakan pada tahap berikutnya.

## 0:35–1:25 — Dataset dan pemeriksaan data

[Tampilkan struktur dataset, kemudian tabel audit pada notebook.]

Dataset yang digunakan adalah Panoptic Indoor Segmentation versi lima, dengan format COCO Segmentation. Dataset ini berisi 922 gambar, yang terbagi menjadi 397 gambar training, 166 gambar validation, dan 359 gambar test. Pembagian asli dataset dipertahankan.

Anotasi memuat class label, bounding box, dan polygon mask untuk setiap instance. Dari file kategori, terdapat 74 kelas segmentasi setelah kategori induk yang tidak memiliki anotasi dipisahkan.

Pemeriksaan data mencakup keberadaan file, ukuran gambar, class mapping, serta koordinat polygon. Ditemukan empat gambar dengan orientasi yang tidak sesuai metadata. Koreksi dilakukan pada salinan kerja dan diperiksa melalui overlay mask. Luas mask juga dihitung ulang dari segmentasi untuk kebutuhan evaluator.

## 1:25–2:10 — Pemilihan tiga model

[Tampilkan section YOLO11, Mask R-CNN, dan RF-DETR pada notebook.]

Tiga model yang dipilih adalah YOLO11n-seg, Mask R-CNN dengan backbone ResNet-50 FPN, dan RF-DETR Seg Nano.

YOLO11 mewakili pendekatan CNN one-stage. Mask R-CNN mewakili CNN two-stage yang menggunakan region proposal dan prediksi mask pada setiap region of interest. RF-DETR memberikan pembanding berbasis transformer.

Dengan kombinasi tersebut, ketentuan tiga model dari minimal dua keluarga arsitektur sudah terpenuhi. Semua model menggunakan pretrained weights, kemudian disesuaikan dengan kelas dataset. Model terbaik belum ditentukan karena keputusan tersebut harus berdasarkan hasil evaluasi.

## 2:10–3:20 — Implementasi notebook dan penyimpanan progress

[Tampilkan konfigurasi training, sequential runner, dan direktori Output Kaggle.]

Implementasi disusun dalam satu notebook Kaggle. Kode dipisahkan menjadi cell pendek, dengan Markdown bernomor dan penjelasan singkat sebelum setiap code cell.

Konfigurasi awal menggunakan 30 epoch per model. Melalui satu Run All, runner menjalankan YOLO11, dilanjutkan Mask R-CNN, kemudian RF-DETR secara berurutan. Setiap model dijalankan dalam subprocess terpisah agar penggunaan memori GPU tidak bertumpuk antar-model.

Checkpoint, training log, dan configuration disimpan dalam direktori masing-masing model. Checkpoint resume diperbarui setiap akhir epoch yang selesai. Notebook juga menyediakan recovery export, sehingga progress yang sudah tersimpan dapat diunduh dan dilanjutkan pada sesi berikutnya.

Salah satu kendala yang ditemukan adalah kegagalan ensurepip saat membuat virtual environment. Implementasinya sudah diperbaiki dengan menonaktifkan bootstrap ensurepip dan mengakses pip dari environment Kaggle. Mekanisme setup tersebut sudah diuji, termasuk pemulihan environment yang terbentuk sebagian.

## 3:20–4:45 — Posisi saat ini dan langkah berikutnya

[Tampilkan log training terbaru yang benar-benar tersedia di Kaggle.]

Saat ini, pekerjaan berada pada tahap training model di Kaggle. Proses training masih berlangsung, sehingga saya belum menyampaikan nilai performa akhir atau menyimpulkan model mana yang paling baik.

Selama training, validation set digunakan untuk memantau performa dan memilih best checkpoint. Setelah training selesai, checkpoint tersebut digunakan untuk evaluation pada test set asli Roboflow. Perbandingan akan menggunakan mask AP, termasuk AP50 dan AP pada rentang IoU 0,50 sampai 0,95, serta performa per kelas dan waktu training.

Untuk NRP saya yang berakhir dengan digit delapan, tahap eksperimen berikutnya adalah membandingkan semua kelas dengan tiga kelas tersulit yang dipilih dari analisis validation. Setelah itu, saya akan mengambil minimal 40 foto ruangan sendiri, termasuk minimal sepuluh foto dalam kondisi sulit, lalu melakukan anotasi di Roboflow.

Foto tersebut nantinya digunakan untuk menguji domain shift dan menganalisis kegagalan model. Dengan demikian, progress yang sudah tersedia adalah dataset yang diperiksa, pipeline tiga model, serta mekanisme checkpoint dan recovery. Pekerjaan yang sedang berjalan adalah training, sedangkan perbandingan hasil akhir dan pengujian foto sendiri menjadi tahap selanjutnya.
