"""Generator dataset sintetis berbasis aturan.

Data ini BUKAN data UMKM nyata. Labelnya dibuat dari skor kelayakan yang
bisa dijelaskan (lihat data/CODEBOOK.md), ditambah noise supaya tidak sempurna.

Jalankan: python scripts/generate_sintetis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from umkm_knn.data import DATASETS, FEATURES, LABEL

N_DEFAULT = 300
SEED_DEFAULT = 42

# Bobot dan "titik jenuh" tiap fitur: nilai di atas titik jenuh dapat poin penuh.
BOBOT = {"lama_usaha_tahun": 0.30, "omzet_juta": 0.35, "jumlah_pekerja": 0.15, "jumlah_aset": 0.20}
JENUH = {"lama_usaha_tahun": 5, "omzet_juta": 20, "jumlah_pekerja": 15, "jumlah_aset": 80}
BATAS_YA = 0.70
BATAS_TUNDA = 0.55
NOISE_SD = 0.03


def skor_kelayakan(df: pd.DataFrame) -> np.ndarray:
    """Skor 0–1: jumlah terbobot dari tiap fitur yang dibagi titik jenuhnya."""
    return sum(BOBOT[f] * np.clip(df[f] / JENUH[f], 0, 1) for f in FEATURES).to_numpy()


def label_dari_skor(skor: np.ndarray) -> np.ndarray:
    return np.where(skor >= BATAS_YA, "YA", np.where(skor >= BATAS_TUNDA, "TUNDA", "TIDAK"))


def generate(n: int = N_DEFAULT, seed: int = SEED_DEFAULT) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "lama_usaha_tahun": rng.integers(0, 11, n),
            "jumlah_pekerja": rng.integers(1, 31, n),
            "omzet_juta": np.clip(rng.lognormal(np.log(10), 0.8, n).round(), 1, 60).astype(int),
            "jumlah_aset": np.clip(rng.lognormal(np.log(30), 0.9, n).round(), 1, 200).astype(int),
        }
    )
    skor = skor_kelayakan(df) + rng.normal(0, NOISE_SD, n)
    df[LABEL] = label_dari_skor(skor)
    return df[FEATURES + [LABEL]]


def main() -> None:
    df = generate()
    path: Path = DATASETS["sintetis"]
    df.to_csv(path, index=False)
    print(f"{len(df)} baris disimpan ke {path.relative_to(path.parents[1])}")
    print(df[LABEL].value_counts().to_string())


if __name__ == "__main__":
    main()
