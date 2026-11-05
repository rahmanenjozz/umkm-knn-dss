from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["lama_usaha_tahun", "jumlah_pekerja", "omzet_juta", "jumlah_aset"]
LABEL = "keputusan"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "umkm.csv"


def load_data(path: str | Path = DEFAULT_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Baca CSV dan kembalikan (X, y)."""
    df = pd.read_csv(path)
    missing = set(FEATURES + [LABEL]) - set(df.columns)
    if missing:
        raise ValueError(f"Kolom tidak ada: {missing}")
    return df[FEATURES].to_numpy(dtype=float), df[LABEL].to_numpy(dtype=str)
