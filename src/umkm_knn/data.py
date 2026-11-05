from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["lama_usaha_tahun", "jumlah_pekerja", "omzet_juta", "jumlah_aset"]
LABEL = "keputusan"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASETS = {
    "asli": DATA_DIR / "umkm.csv",  # soal tugas kuliah, 45 baris
    "sintetis": DATA_DIR / "umkm_sintetis.csv",  # dibuat oleh scripts/generate_sintetis.py
}


def load_data(source: str | Path = "asli") -> tuple[np.ndarray, np.ndarray]:
    """Baca dataset dan kembalikan (X, y).

    `source` bisa nama dataset ("asli" / "sintetis") atau path ke CSV.
    """
    path = DATASETS.get(source, source) if isinstance(source, str) else source
    df = pd.read_csv(path)
    missing = set(FEATURES + [LABEL]) - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ada: {missing}")
    return df[FEATURES].to_numpy(dtype=float), df[LABEL].to_numpy(dtype=str)
