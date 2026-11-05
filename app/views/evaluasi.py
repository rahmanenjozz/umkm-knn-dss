import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from common import INFO_DATASET, KELAS, baca_df, baca_report, dataset_aktif

from umkm_knn.data import LABEL
from umkm_knn.evaluate import metrics

nama = dataset_aktif()
report = baca_report(nama)

st.title("Evaluasi model")
st.caption(f"Dataset: **{INFO_DATASET[nama]['nama']}**")

if report is None:
    st.error("Belum ada hasil evaluasi. Jalankan `python -m umkm_knn.evaluate` lalu muat ulang.")
    st.stop()

hasil = pd.DataFrame(report["hasil"])
hasil["Voting"] = np.where(hasil["weighted"], "Bobot jarak", "Sama rata")
baseline = baca_df(nama)[LABEL].value_counts(normalize=True).max()
t = report["terbaik"]
best = hasil[(hasil["k"] == t["k"]) & (hasil["weighted"] == t["weighted"])].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy terbaik", f"{best['accuracy_mean']:.1%} ± {best['accuracy_std']:.1%}")
c2.metric("Macro-F1", f"{best['macro_f1_mean']:.2f}")
c3.metric("Baseline (tebak kelas terbanyak)", f"{baseline:.1%}")
st.caption(
    f"Repeated stratified {report['n_splits']}-fold × {report['n_repeats']} pengulangan. "
    f"Setting terbaik: k = {t['k']}, {'bobot jarak' if t['weighted'] else 'voting sama rata'}."
)

if nama == "asli":
    st.warning(
        "Dataset asli hanya 45 baris dan polanya lemah: akurasi cuma sedikit di atas baseline "
        "dan makin turun saat k membesar. Ini keterbatasan data, bukan bug.",
        icon="⚠️",
    )

st.subheader("Accuracy per nilai k", divider="gray")
hasil["atas"] = hasil["accuracy_mean"] + hasil["accuracy_std"]
hasil["bawah"] = hasil["accuracy_mean"] - hasil["accuracy_std"]
garis = alt.Chart(hasil).encode(
    x=alt.X("k:O", title="k", axis=alt.Axis(labelAngle=0)),
    color=alt.Color(
        "Voting:N",
        title=None,
        scale=alt.Scale(domain=["Sama rata", "Bobot jarak"], range=["#4c78a8", "#f58518"]),
        legend=alt.Legend(orient="top", symbolOpacity=1),
    ),
)
pita = garis.mark_area(opacity=0.15).encode(y="bawah:Q", y2="atas:Q")
titik = garis.mark_line(point=True).encode(
    y=alt.Y("accuracy_mean:Q", title="Accuracy", axis=alt.Axis(format="%")),
    tooltip=["k", "Voting", alt.Tooltip("accuracy_mean:Q", format=".1%", title="Accuracy")],
)
dasar = (
    alt.Chart(pd.DataFrame({"y": [baseline]}))
    .mark_rule(strokeDash=[4, 4], color="gray")
    .encode(y="y:Q")
)
st.altair_chart((pita + titik + dasar).properties(height=320), width="stretch")
st.caption("Pita = ±1 standar deviasi antar pengulangan. Garis putus-putus = baseline.")

st.subheader("Confusion matrix (setting terbaik)", divider="gray")
classes = np.array(best["classes"])
cm = np.array(best["confusion_matrix"])
urut = [list(classes).index(c) for c in KELAS]
cm, classes = cm[np.ix_(urut, urut)], np.array(KELAS)

kiri, kanan = st.columns([3, 2], gap="large")
with kiri:
    sel = pd.DataFrame(
        [(a, p, int(cm[i, j])) for i, a in enumerate(classes) for j, p in enumerate(classes)],
        columns=["Sebenarnya", "Tebakan", "Jumlah"],
    )
    dasar_cm = alt.Chart(sel).encode(
        x=alt.X("Tebakan:N", sort=KELAS, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Sebenarnya:N", sort=KELAS),
    )
    st.altair_chart(
        (
            dasar_cm.mark_rect().encode(
                color=alt.Color("Jumlah:Q", legend=None, scale=alt.Scale(scheme="blues"))
            )
            + dasar_cm.mark_text(fontSize=16).encode(
                text="Jumlah:Q",
                color=alt.condition(
                    alt.datum.Jumlah > cm.max() / 2, alt.value("white"), alt.value("black")
                ),
            )
        ).properties(height=300),
        width="stretch",
    )
    st.caption(f"Dijumlah dari {report['n_repeats']} pengulangan cross-validation.")
with kanan:
    m = metrics(cm, classes)["per_kelas"]
    st.dataframe(
        pd.DataFrame(m).T.rename(columns=str.capitalize).style.format("{:.2f}"),
        width="stretch",
    )
