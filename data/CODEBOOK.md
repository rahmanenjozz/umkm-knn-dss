# Codebook — data/umkm.csv

Sumber: Data bersifat contoh/latihan, bukan data UMKM sebenarnya.
Jumlah: 45 baris, 4 fitur, 1 label

| Kolom | Tipe | Satuan | Rentang |
|---|---|---|---|
| lama_usaha_tahun | int | tahun | 0–7 |
| jumlah_pekerja | int | orang | 1–30 |
| omzet_juta | int | juta Rp / bulan | 1–21 |
| jumlah_aset | int | unit | 1–28 |
| keputusan | kategori | — | YA, TIDAK, TUNDA |

## Label
- YA: layak menerima modal
- TUNDA: perlu pertimbangan lanjut
- TIDAK: tidak layak

Distribusi: YA 16, TIDAK 16, TUNDA 13

## Catatan kualitas
- Baris 41 (omzet 21, aset 28) jauh di atas baris lain. Belum diubah, perlu dicek.
- Rentang fitur berbeda jauh, jadi perlu normalisasi sebelum KNN.
- Dataset kecil; hasil hanya untuk demonstrasi, bukan keputusan nyata.
