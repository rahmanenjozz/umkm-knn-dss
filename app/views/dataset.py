import altair as alt
import streamlit as st
from common import (
    FEATURES,
    INFO_DATASET,
    KELAS,
    LABEL,
    ROOT,
    SINGKAT,
    WARNA,
    baca_df,
    dataset_aktif,
)

nama = dataset_aktif()
info = INFO_DATASET[nama]
df = baca_df(nama)

st.title("Dataset")
st.caption(f"**{info['nama']}** · {len(df)} baris · {len(FEATURES)} fitur")

if nama == "sintetis":
    st.info("Data sintetis dibuat dari aturan berbobot (lihat codebook). Bukan data UMKM nyata.")

kiri, kanan = st.columns([2, 1], gap="large")
with kiri:
    st.dataframe(
        df.rename(columns=info["label"] | {LABEL: "Keputusan"}),
        hide_index=True,
        width="stretch",
        height=400,
    )
with kanan:
    st.markdown("**Distribusi keputusan**")
    st.altair_chart(
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{LABEL}:N", sort=KELAS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count():Q", title="Jumlah"),
            color=alt.Color(
                f"{LABEL}:N",
                scale=alt.Scale(domain=KELAS, range=[WARNA[c] for c in KELAS]),
                legend=None,
            ),
        )
        .properties(height=250),
        width="stretch",
    )
    st.markdown("**Ringkasan**")
    ringkas = df[FEATURES].describe().T[["min", "50%", "max"]]
    st.dataframe(
        ringkas.rename(index=SINGKAT, columns={"50%": "median"}).style.format("{:g}"),
        width="stretch",
    )

with st.expander("Codebook"):
    st.markdown((ROOT / "data" / "CODEBOOK.md").read_text(encoding="utf-8"))
