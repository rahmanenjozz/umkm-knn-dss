"""Entry point: streamlit run app/streamlit_app.py"""

import streamlit as st
from common import INFO_DATASET

st.set_page_config(page_title="UMKM KNN DSS", page_icon="🏪", layout="wide")

with st.sidebar:
    st.selectbox(
        "Dataset",
        options=list(INFO_DATASET),
        format_func=lambda k: INFO_DATASET[k]["nama"],
        key="dataset",
    )
    st.caption(
        "Alat bantu keputusan untuk demonstrasi KNN. "
        "Bukan untuk menentukan penerima modal sungguhan."
    )

nav = st.navigation(
    [
        st.Page("views/prediksi.py", title="Prediksi", icon="🎯", default=True),
        st.Page("views/evaluasi.py", title="Evaluasi", icon="📊"),
        st.Page("views/dataset.py", title="Dataset", icon="🗂️"),
        st.Page("views/cara_kerja.py", title="Cara kerja KNN", icon="📖"),
    ]
)
nav.run()
