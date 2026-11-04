import pandas as pd
import streamlit as st
import base64
from pathlib import Path

# Import fungsi render dari masing-masing file Halaman
from Halaman_1 import render_overview_page
from Halaman_2 import render_eda_page
from Halaman_3 import render_evaluation_page
from Halaman_4 import render_prediction_page
from Halaman_5 import render_about_page


# 1. Setup konfigurasi halaman
st.set_page_config(
    page_title="Dashboard TA - Regresi Ridge & LASSO",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. Styling tampilan utama
st.markdown(
    """
    <style>
    /* Latar utama dan ruang konten */
    .stApp {
        background: linear-gradient(135deg, #f7faf9 0%, #eef5f3 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #123b35;
        border-right: 1px solid #d8e6e1;
    }

    [data-testid="stSidebar"] * {
        color: #f4f8f6;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c9ded7;
    }

    /* Radio navigasi */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.55rem 0.7rem;
        border-radius: 0.6rem;
        transition: background 0.2s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #24564e;
    }

    /* Tombol */
    .stButton > button {
        border: 1px solid #c5d8d1;
        border-radius: 0.7rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #287c6b;
        color: #176653;
    }

    /* Garis pemisah sidebar */
    [data-testid="stSidebar"] hr {
        border-color: #37675e;
    }

    /* Judul */
    h1, h2, h3 {
        color: #173f38;
        letter-spacing: -0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 3. Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")


try:
    df = load_data()
    data_loaded = True
except Exception:
    df = None
    data_loaded = False


# 4. Sidebar: identitas dan navigasi
st.sidebar.markdown(
    """
    <div style="padding: 0.4rem 0 1rem;">
        <div style="font-size: 2rem;">🏡</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">
            House Price Analytics
        </div>
        <div style="font-size: 0.8rem; color: #c9ded7; margin-top: 0.25rem;">
            Dashboard Tugas Akhir · Matematika ITB
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_page = st.sidebar.radio(
    "Pilih Halaman:",
    options=[
        "🏠 Overview",
        "📊 EDA",
        "⚖️ Model Comparison",
        "🔮 House Price Predictor",
        "👨🏻‍🎓 Profile",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()

# Tombol clear cache
if st.sidebar.button("🔄 Reset / Clear Cache", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.toast("Cache berhasil dibersihkan!", icon="🧹")

st.sidebar.divider()
st.sidebar.caption("Tugas Akhir Seminar Matematika ITB")


# 5. Routing / pemanggilan halaman
if selected_page == "🏠 Overview":
    render_overview_page(df)

elif selected_page == "📊 EDA":
    render_eda_page(df)

elif selected_page == "⚖️ Model Comparison":
    render_evaluation_page(df)

elif selected_page == "🔮 House Price Predictor":
    render_prediction_page(df)

elif selected_page == "👨🏻‍🎓 Profile":
    render_about_page(df)