import numpy as np
import pandas as pd

from generate_sintetis import generate, label_dari_skor
from umkm_knn.data import DATASETS, FEATURES, LABEL, load_data
from umkm_knn.evaluate import cross_validate


def test_deterministik():
    pd.testing.assert_frame_equal(generate(seed=1), generate(seed=1))


def test_csv_di_repo_sesuai_generator():
    # kalau gagal: jalankan ulang `python scripts/generate_sintetis.py` lalu commit CSV-nya
    pd.testing.assert_frame_equal(pd.read_csv(DATASETS["sintetis"]), generate())


def test_kolom_dan_label_valid():
    df = generate(n=200)
    assert list(df.columns) == FEATURES + [LABEL]
    assert set(df[LABEL]) == {"YA", "TUNDA", "TIDAK"}
    assert (df[FEATURES] >= 0).all().all()


def test_kelas_cukup_seimbang():
    counts = generate()[LABEL].value_counts(normalize=True)
    assert counts.min() > 0.25


def test_batas_label():
    skor = np.array([0.1, 0.6, 0.9])
    assert label_dari_skor(skor).tolist() == ["TIDAK", "TUNDA", "YA"]


def test_knn_belajar_pola_sintetis():
    X, y = load_data("sintetis")
    acc = cross_validate(X, y, k=5, seeds=range(2))["accuracy_mean"]
    assert acc > 0.6  # jauh di atas baseline ~0.35
