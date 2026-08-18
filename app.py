import pandas as pd
import streamlit as st

# Import fungsi render dari masing-masing file Halaman
from Halaman_1 import render_overview_page
from Halaman_2 import render_eda_page
from Halaman_3 import render_evaluation_page
from Halaman_4 import render_prediction_page
from Halaman_5 import render_about_page

# 1. Setup Konfigurasi Halaman (Dipanggil pertama kali)
st.set_page_config(
    page_title='Dashboard TA - Regresi Ridge & Lasso',
    page_icon='🏠',
    layout='wide',
    initial_sidebar_state='expanded',
)


# 2. Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv('train.csv')


try:
    df = load_data()
    data_loaded = True
except Exception as e:
    df = None
    data_loaded = False


# ============================================================
# SIDEBAR MODERN & MINIMALIS (Tanpa Kotak Status Kaku)
# ============================================================
st.sidebar.markdown('### 📌 Navigasi Utama')

# Pilihan Halaman dengan Radio Button
selected_page = st.sidebar.radio(
    'Pilih Halaman:',
    options=[
        '🏠 Overview',
        '📊 EDA',
        '⚖️ Model Comparison',
        '🔮 House Price Predictor',
        '👨🏻‍🎓 Profile',
    ],
    label_visibility='collapsed',
)

st.sidebar.divider()

# --- Tombol Aksi Cepat (Clear Cache) yang Bersih ---
if st.sidebar.button('🔄 Reset / Clear Cache', use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.toast('Cache berhasil dibersihkan!', icon='🧹')

st.sidebar.divider()
st.sidebar.caption('Tugas Akhir Seminar Matematika ITB')

# ============================================================
# ROUTING / PEMANGGILAN HALAMAN
# ============================================================
if selected_page == '🏠 Overview':
    render_overview_page(df)

elif selected_page == '📊 EDA':
    render_eda_page(df)

elif selected_page == '⚖️ Model Comparison':
    render_evaluation_page(df)

elif selected_page == '🔮 House Price Predictor':
    render_prediction_page(df)  # Memanggil file Halaman_4

elif selected_page == '👨🏻‍🎓 Profile':
    render_about_page(df)  # Memanggil file Halaman_5