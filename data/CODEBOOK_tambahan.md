---

# Dataset sintetis: data/umkm_sintetis.csv

**Bukan data UMKM nyata.** Dataset ini dibuat oleh `python scripts/generate_sintetis.py`
(seed 42) untuk demo, karena data asli terlalu kecil dan polanya lemah.

Jumlah: 300 baris. Kolom sama dengan dataset asli.
Distribusi label: TUNDA 104, TIDAK 98, YA 98.

| Kolom | Satuan | Rentang | Cara dibuat |
|---|---|---|---|
| lama_usaha_tahun | tahun | 0–10 | acak seragam |
| jumlah_pekerja | orang | 1–30 | acak seragam |
| omzet_juta | juta Rp / bulan | 1–60 | log-normal (median ±10) |
| jumlah_aset | juta Rp | 3–200 | log-normal (median ±30) |

## Aturan label

Setiap fitur dibagi "titik jenuh"-nya lalu dibatasi maksimal 1, kemudian dijumlah dengan bobot:

| Fitur | Bobot | Titik jenuh |
|---|---|---|
| omzet_juta | 0,35 | 20 juta/bulan |
| lama_usaha_tahun | 0,30 | 5 tahun |
| jumlah_aset | 0,20 | 80 juta |
| jumlah_pekerja | 0,15 | 15 orang |

Skor (0–1) ditambah noise normal (sd 0,03), lalu:
- skor ≥ 0,70 → **YA**
- 0,55 ≤ skor < 0,70 → **TUNDA**
- skor < 0,55 → **TIDAK**

Bobot dan batas ini adalah asumsi buatan sendiri, bukan kriteria resmi program modal mana pun.

## Catatan
- Satuan omzet dan aset di dataset sintetis sudah pasti (juta Rp), berbeda dengan
  dataset asli yang satuannya tidak dijelaskan di soal. Jangan menggabungkan kedua dataset.
- Kalau aturan atau seed diubah, jalankan ulang generator dan evaluasi, lalu commit
  CSV dan `reports/` yang baru. Test `test_csv_di_repo_sesuai_generator` akan gagal
  kalau CSV tidak sinkron dengan generator.
