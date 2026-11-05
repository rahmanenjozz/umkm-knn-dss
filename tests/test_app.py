"""Smoke test UI: semua halaman bisa dirender tanpa error untuk kedua dataset."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
HALAMAN = ["views/prediksi.py", "views/evaluasi.py", "views/dataset.py", "views/cara_kerja.py"]


@pytest.mark.parametrize("dataset", ["asli", "sintetis"])
@pytest.mark.parametrize("halaman", HALAMAN)
def test_halaman_jalan(dataset, halaman):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["dataset"] = dataset
    at.run()
    at.switch_page(halaman).run()
    assert not at.exception, at.exception


def test_prediksi_berubah_saat_k_diubah():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.slider[0].set_value(1).run()
    assert not at.exception
    assert "1 tetangga terdekat" in [s.value for s in at.subheader]
