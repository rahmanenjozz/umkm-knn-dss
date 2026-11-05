# UMKM KNN DSS

Sistem pendukung keputusan penerima modal UMKM dengan **K-Nearest Neighbors yang ditulis dari nol** (NumPy), lengkap dengan evaluasi cross-validation dan UI Streamlit.

> Proyek tugas kuliah (2018) yang dikembangkan ulang. Untuk demonstrasi, **bukan** untuk menentukan penerima modal sungguhan.

## Fitur
- **Prediksi**: masukkan data UMKM → keputusan YA / TUNDA / TIDAK, proporsi suara, tabel tetangga terdekat, dan posisi input di grafik PCA 2D
- **Evaluasi**: accuracy per nilai k (repeated stratified 5-fold × 10), confusion matrix, perbandingan dengan baseline
- **Dataset**: data asli (45 baris, soal tugas kuliah) dan data sintetis (300 baris, berbasis aturan)
- **Cara kerja KNN**: penjelasan normalisasi, jarak Euclidean, dan voting

## Hasil evaluasi
| Dataset | Baris | Accuracy terbaik | Baseline |
|---|---|---|---|
| Asli | 45 | 47.6% ± 3.3% | 35.6% |
| Sintetis | 300 | 73.9% ± 1.5% | 34.7% |

Data asli polanya lemah sehingga akurasinya hanya sedikit di atas baseline. Detail ada di halaman Evaluasi dan [`data/CODEBOOK.md`](data/CODEBOOK.md).

## Menjalankan
```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app/streamlit_app.py
```

Perintah lain:
```bash
pytest                               # semua test
python -m umkm_knn.evaluate          # hitung ulang reports/evaluasi_*.json
python scripts/generate_sintetis.py  # buat ulang data/umkm_sintetis.csv
```

## Struktur
```
app/                  UI Streamlit (streamlit_app.py + views/)
src/umkm_knn/         KNN, normalisasi, evaluasi, loader data
scripts/              generator dataset sintetis
data/                 dataset + codebook
reports/              hasil evaluasi (dibaca halaman Evaluasi)
tests/                pytest (termasuk perbandingan dengan scikit-learn)
```

## Lisensi
[Apache 2.0](LICENSE)
