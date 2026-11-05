import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from common import (
    FEATURES,
    INFO_DATASET,
    KELAS,
    LABEL,
    SINGKAT,
    WARNA,
    baca_df,
    dataset_aktif,
    pca_2d,
)

from umkm_knn.knn import KNNClassifier
from umkm_knn.preprocessing import MinMaxScaler

nama = dataset_aktif()
info = INFO_DATASET[nama]
df = baca_df(nama)
X, y = df[FEATURES].to_numpy(float), df[LABEL].to_numpy(str)

st.title("Prediksi kelayakan modal UMKM")
st.caption(f"Dataset: **{info['nama']}** · {len(df)} data latih")

kiri, kanan = st.columns([1, 2], gap="large")

with kiri, st.container(border=True):
    nilai = {}
    for f in FEATURES:
        lo, hi = int(df[f].min()), int(df[f].max())
        nilai[f] = st.number_input(
            info["label"][f],
            min_value=0,
            max_value=hi * 2,
            value=int(df[f].median()),
            step=1,
            help=f"Rentang di data latih: {lo}–{hi}",
        )
    k = st.slider("Jumlah tetangga (k)", 1, 15, 5, step=2)
    weighted = st.toggle(
        "Bobot berdasarkan jarak", help="Tetangga yang lebih dekat bersuara lebih besar"
    )

scaler = MinMaxScaler().fit(X)
Xs = scaler.transform(X)
x_baru = scaler.transform(np.array([[nilai[f] for f in FEATURES]], dtype=float))
model = KNNClassifier(k=k, weighted=weighted).fit(Xs, y)
proba = dict(zip(model.classes_, model.predict_proba(x_baru)[0]))
hasil = str(model.predict(x_baru)[0])
jarak, idx = model.kneighbors(x_baru)
suara = int((y[idx[0]] == hasil).sum())
seri = [c for c, v in proba.items() if c != hasil and np.isclose(v, proba[hasil])]

with kanan:
    teks = {"YA": "Layak menerima modal", "TUNDA": "Ditunda", "TIDAK": "Belum layak"}
    st.markdown(
        f"<div style='background:{WARNA[hasil]};color:white;padding:1.2rem 1.5rem;"
        f"border-radius:12px'><div style='font-size:2rem;font-weight:700'>{hasil}</div>"
        f"<div>{teks[hasil]} · {suara} dari {k} tetangga terdekat</div></div>",
        unsafe_allow_html=True,
    )
    if seri:
        st.caption(
            f"Suara seri dengan {', '.join(seri)}. Pemenang dipilih berdasarkan urutan alfabet; "
            "coba ubah k atau aktifkan bobot jarak."
        )

    st.subheader("Proporsi suara", divider="gray")
    df_proba = pd.DataFrame({"Kelas": KELAS, "Proporsi": [proba.get(c, 0.0) for c in KELAS]})
    st.altair_chart(
        alt.Chart(df_proba)
        .mark_bar()
        .encode(
            x=alt.X("Proporsi:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Kelas:N", sort=KELAS, title=None),
            color=alt.Color(
                "Kelas:N",
                scale=alt.Scale(domain=KELAS, range=[WARNA[c] for c in KELAS]),
                legend=None,
            ),
            tooltip=["Kelas", alt.Tooltip("Proporsi:Q", format=".0%")],
        )
        .properties(height=150),
        width="stretch",
    )

    st.subheader(f"{k} tetangga terdekat", divider="gray")
    tetangga = df.iloc[idx[0]].copy()
    tetangga.insert(0, "jarak", jarak[0].round(3))
    st.dataframe(
        tetangga.rename(columns=SINGKAT | {LABEL: "Keputusan", "jarak": "Jarak"}),
        hide_index=True,
        width="stretch",
    )
    st.caption("Jarak Euclidean dihitung setelah semua fitur dinormalisasi ke 0–1.")

st.subheader("Posisi input di antara data latih (PCA 2D)", divider="gray")
proj, mean, komp = pca_2d(Xs)
titik = pd.DataFrame({"PC1": proj[:, 0], "PC2": proj[:, 1], "Keputusan": y})
titik["Tetangga"] = np.isin(np.arange(len(y)), idx[0])
p_baru = ((x_baru - mean) @ komp.T)[0]

warna = alt.Color("Keputusan:N", scale=alt.Scale(domain=KELAS, range=[WARNA[c] for c in KELAS]))
latar = (
    alt.Chart(titik)
    .mark_circle(size=60)
    .encode(
        x="PC1:Q",
        y="PC2:Q",
        color=warna,
        opacity=alt.condition("datum.Tetangga", alt.value(1), alt.value(0.35)),
        stroke=alt.condition("datum.Tetangga", alt.value("black"), alt.value(None)),
        tooltip=["Keputusan", "Tetangga"],
    )
)
input_ = (
    alt.Chart(pd.DataFrame({"PC1": [p_baru[0]], "PC2": [p_baru[1]], "Label": ["Input kamu"]}))
    .mark_point(shape="cross", size=400, color="black", strokeWidth=3)
    .encode(x="PC1:Q", y="PC2:Q", tooltip=["Label"])
)
st.altair_chart((latar + input_).properties(height=420), width="stretch")
st.caption(
    "PCA memadatkan 4 fitur jadi 2 dimensi supaya bisa digambar, jadi jarak di grafik "
    "hanya perkiraan. Titik bergaris hitam = tetangga yang dipakai untuk voting."
)
