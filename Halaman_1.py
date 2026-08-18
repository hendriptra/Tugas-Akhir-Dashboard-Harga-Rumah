import streamlit as st
import pandas as pd

def render_overview_page(df=None):

    # ======================================================
    # HEADER
    # ======================================================

    st.title("🏠 Prediksi Harga Jual Rumah")

    st.subheader(
        "Analisis Perbandingan Regresi Ridge dan LASSO "
        "dalam Mengatasi Masalah Multikolinearitas dan Dimensi Tinggi pada Prediksi Harga Jual Rumah"
    )

    st.caption(
        "Tugas Akhir Sarjana Matematika | Institut Teknologi Bandung | 2026"
    )

    # ======================================================
    # PROJECT INFORMATION
    # ======================================================

    with st.expander("ℹ️ Informasi Project", expanded=True):

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("""
            **📂 Dataset**
            
            - House Prices: Advanced Regression Techniques
            - **Sumber**: Kaggle
            """)

        with c2:
            st.markdown("""
            **📊 Model Utama**

            - Ridge Regression
            - LASSO Regression
            """)

        with c3:
            st.markdown("""
            **📈 Evaluasi**

            - R² Score
            - MAE
            - MSE dan RMSE
            """)

    st.divider()

    # ======================================================
    # BACKGROUND & OBJECTIVE
    # ======================================================

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 📖 Latar Belakang")
        st.markdown(
            """
            <div style="text-align: justify; color: inherit; line-height: 1.6;">
                Prediksi harga rumah merupakan permasalahan regresi yang melibatkan banyak 
                variabel prediktor. Banyaknya fitur pada data dapat menyebabkan permasalahan 
                <b>multikolinearitas</b>, yaitu kondisi ketika antarvariabel prediktor memiliki 
                korelasi yang tinggi sehingga dapat memengaruhi kestabilan model.<br><br>
                Penelitian ini juga mengevaluasi performa model pada kondisi 
                <b>high-dimensional data (p > n)</b> yang dibentuk melalui <b>random sampling</b>, 
                yaitu ketika jumlah variabel prediktor lebih besar dibandingkan jumlah observasi.<br><br>
                Penelitian ini menerapkan metode regularisasi <b>Ridge Regression</b>
                dan <b>LASSO Regression</b> untuk menghasilkan model prediksi yang lebih
                stabil serta mampu menangani kompleksitas data berdimensi tinggi.
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown("### 🎯 Tujuan Dashboard")
        st.markdown("""
        Dashboard ini bertujuan untuk:

        - Mengeksplorasi karakteristik dataset House Prices.
        - Menganalisis faktor yang memengaruhi harga rumah.
        - Membandingkan performa model regresi.
        - Melakukan simulasi prediksi harga rumah.
        """)

    st.divider()

    # ======================================================
    # HASIL PENELITIAN
    # ======================================================

    st.markdown("### 🏆 Ringkasan Hasil Penelitian")

    st.caption(
        "Hyperparameter λ dipilih berdasarkan nilai CV MSE terendah pada data training. "
        "R², MAE, MSE, dan RMSE kemudian digunakan untuk mengevaluasi performa "
        "model pada data testing."
    )

    a, b = st.columns(2)

    with a:
        st.info("""
        **🏆 Multikolinearitas: LASSO Regression**
        
        - λ Optimal = **1000**
        - R² = **0.85351**
        - MAE = **$18,897.55**
        - MSE = **1.1236 × 10⁹**
        - RMSE = **$33,520.02**
        """)

    with b:
        st.info("""
        **🏆 High-Dimensional: Ridge Regression**

        - λ Optimal = **100**
        - R² = **0.87764**
        - MAE = **$20,589.37**
        - MSE = **0.6845 × 10⁹**
        - RMSE = **$26,162.37**
        """)

    st.divider()

    # ======================================================
    # DATASET SUMMARY
    # ======================================================

    st.markdown("### 📊 Ringkasan Dataset")

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Jumlah Observasi",
            f"{df.shape[0]:,}"
        )

        c2.metric(
            "Jumlah Variabel",
            f"{df.shape[1]-1}"
        )

        if "SalePrice" in df.columns:
            c3.metric(
                "Rata-rata Harga Jual Rumah",
                f"${df['SalePrice'].mean():,.0f}"
            )

            c4.metric(
                "Median Harga Jual Rumah",
                f"${df['SalePrice'].median():,.0f}"
            )

        st.markdown("#### Preview Dataset")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # ==================================================
        # DATASET DESCRIPTION
        # ==================================================

        st.markdown(
            """
            <div style="text-align: justify; color: inherit; line-height: 1.6;">
                <b>Tentang Dataset:</b> Dataset <i>House Prices – Advanced Regression Techniques</i>
                berisi informasi mengenai berbagai karakteristik rumah di Ames, Iowa, Amerika Serikat.
                Dataset mencakup <b>79 variabel penjelas</b> yang menggambarkan hampir seluruh aspek properti,
                mulai dari kualitas dan kondisi rumah, luas bangunan, jumlah kamar, hingga karakteristik
                lingkungan sekitar. Data ini digunakan untuk memahami faktor yang memengaruhi harga dan
                memprediksi <b>harga jual akhir rumah</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "📚 **Sumber Dataset:** Kaggle — *House Prices: Advanced Regression Techniques*"
        )

    else:
        st.info("Dataset belum dimuat.")