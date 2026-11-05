import streamlit as st

st.title("Cara kerja KNN")

st.markdown(
    """
**K-Nearest Neighbors** menebak kelas sebuah UMKM baru dengan melihat UMKM lain
di data latih yang paling mirip dengannya.

### 1. Normalisasi
Rentang tiap fitur berbeda jauh (misalnya pekerja 1–30, lama usaha 0–10).
Tanpa normalisasi, fitur dengan angka besar akan mendominasi jarak. Karena itu setiap
fitur diskalakan ke 0–1 dengan min-max:
"""
)
st.latex(r"x' = \frac{x - \min(x)}{\max(x) - \min(x)}")

st.markdown("### 2. Hitung jarak\nJarak Euclidean antara input $q$ dan setiap data latih $p$:")
st.latex(r"d(p, q) = \sqrt{\sum_{i=1}^{4} (p_i - q_i)^2}")

st.markdown(
    """
### 3. Ambil k tetangga terdekat
Urutkan semua jarak, ambil k data dengan jarak terkecil.

### 4. Voting
- **Sama rata**: tiap tetangga punya 1 suara, kelas dengan suara terbanyak menang.
- **Bobot jarak**: suara tiap tetangga = 1 / jarak, jadi tetangga yang lebih dekat lebih berpengaruh.

Kalau seri, dipilih kelas yang urutannya pertama secara alfabet (sama dengan scikit-learn).

### Memilih k
- k kecil: sensitif terhadap noise, satu data aneh bisa mengubah hasil.
- k besar: lebih stabil, tapi batas antar kelas jadi kabur.

Nilai k terbaik dipilih dengan *repeated stratified cross-validation*, lihat halaman **Evaluasi**.

### Keterbatasan
- KNN tidak "belajar" aturan, hanya membandingkan dengan data yang ada. Kualitas hasil
  sepenuhnya bergantung pada kualitas dan jumlah data.
- Aplikasi ini untuk demonstrasi. Keputusan pemberian modal sungguhan perlu penilaian manusia.

Implementasinya ditulis dari nol dengan NumPy di `src/umkm_knn/knn.py`, dan diuji agar
hasilnya sama dengan `KNeighborsClassifier` dari scikit-learn.
"""
)
