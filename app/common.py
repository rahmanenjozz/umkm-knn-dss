"""Fungsi bersama untuk semua halaman Streamlit."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # jaga-jaga kalau package belum di-install

from umkm_knn.data import DATASETS, FEATURES, LABEL

WARNA = {"YA": "#2e9e5b", "TUNDA": "#e0a526", "TIDAK": "#d6453d"}
KELAS = ["YA", "TUNDA", "TIDAK"]
SINGKAT = {
    "lama_usaha_tahun": "Lama usaha",
    "jumlah_pekerja": "Pekerja",
    "omzet_juta": "Omzet",
    "jumlah_aset": "Aset",
}

INFO_DATASET = {
    "asli": {
        "nama": "Asli (Contoh)",
        "label": {
            "lama_usaha_tahun": "Lama usaha (tahun)",
            "jumlah_pekerja": "Jumlah pekerja (orang)",
            "omzet_juta": "Omzet (juta Rp)",
            "jumlah_aset": "Jumlah aset",
        },
    },
    "sintetis": {
        "nama": "Sintetis (berbasis aturan)",
        "label": {
            "lama_usaha_tahun": "Lama usaha (tahun)",
            "jumlah_pekerja": "Jumlah pekerja (orang)",
            "omzet_juta": "Omzet (juta Rp / bulan)",
            "jumlah_aset": "Jumlah aset (juta Rp)",
        },
    },
}


def dataset_aktif() -> str:
    return st.session_state.get("dataset", "asli")


@st.cache_data
def baca_df(nama: str) -> pd.DataFrame:
    return pd.read_csv(DATASETS[nama])


@st.cache_data
def baca_report(nama: str) -> dict | None:
    path = ROOT / "reports" / f"evaluasi_{nama}.json"
    return json.loads(path.read_text()) if path.exists() else None


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA sederhana dengan SVD. Mengembalikan (proyeksi, mean, komponen)."""
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    komponen = vt[:2]
    return (X - mean) @ komponen.T, mean, komponen


__all__ = ["FEATURES", "INFO_DATASET", "KELAS", "LABEL", "ROOT", "WARNA"]
