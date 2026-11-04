import streamlit as st
import pandas as pd


def apply_overview_page_style():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .overview-hero {
            display: flex;
            align-items: center;
            gap: 28px;
            background: linear-gradient(120deg, #102a43 0%, #1d4e89 100%);
            padding: 30px;
            border-radius: 18px;
            margin-bottom: 26px;
            color: white;
        }

        .overview-hero-content {
            flex: 1;
            min-width: 0;
        }

        .overview-tag {
            display: inline-block;
            padding: 5px 11px;
            margin-bottom: 13px;
            border: 1px solid rgba(255,255,255,.35);
            border-radius: 999px;
            color: #e7f0fa;
            font-size: 12px;
            letter-spacing: .4px;
        }

        .overview-hero h1 {
            color: white;
            font-size: 30px;
            line-height: 1.25;
            margin: 0 0 12px 0;
            font-weight: 750;
        }

        .overview-hero .overview-subtitle {
            color: #e6eef7;
            font-size: 15px;
            line-height: 1.65;
            margin: 0 0 15px 0;
        }

        .overview-hero .overview-meta {
            color: #c8d8e8;
            font-size: 13px;
            margin: 0;
        }

        .overview-hero-image {
            width: 34%;
            min-width: 220px;
        }

        .overview-hero-image img {
            width: 100%;
            height: 220px;
            object-fit: cover;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,.2);
        }

        .overview-section-title {
            font-size: 21px;
            font-weight: 700;
            margin: 10px 0 5px 0;
        }

        .overview-section-caption {
            color: #718096;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 18px;
        }

        .overview-card {
            height: 100%;
            padding: 18px 20px;
            border: 1px solid rgba(128,128,128,.2);
            border-radius: 14px;
            background: rgba(128,128,128,.045);
            line-height: 1.65;
        }

        .overview-card h4 {
            margin: 0 0 10px 0;
            font-size: 16px;
        }

        .overview-card p {
            margin: 0;
            font-size: 14px;
        }

        .overview-result {
            height: 100%;
            padding: 20px;
            border: 1px solid rgba(128,128,128,.2);
            border-radius: 14px;
            background: rgba(128,128,128,.045);
        }

        .overview-result .result-label {
            font-size: 12px;
            letter-spacing: .5px;
            text-transform: uppercase;
            color: #718096;
            margin-bottom: 8px;
        }

        .overview-result h3 {
            font-size: 19px;
            margin: 0 0 14px 0;
        }

        .overview-result .result-value {
            font-size: 14px;
            line-height: 1.8;
        }

        div[data-testid="stMetric"] {
            background: rgba(128,128,128,.06);
            border: 1px solid rgba(128,128,128,.16);
            padding: 15px 17px;
            border-radius: 14px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(128,128,128,.18);
            border-radius: 12px;
            overflow: hidden;
        }

        hr {
            margin: 28px 0;
            border-color: rgba(128,128,128,.2);
        }

        @media (max-width: 700px) {
            .overview-hero {
                display: block;
                padding: 23px;
            }

            .overview-hero-image {
                width: 100%;
                min-width: 0;
                margin-top: 20px;
            }

            .overview-hero-image img {
                height: 190px;
            }

            .overview-hero h1 {
                font-size: 25px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def overview_section(title, caption=None):
    st.markdown(
        f'<div class="overview-section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if caption:
        st.markdown(
            f'<div class="overview-section-caption">{caption}</div>',
            unsafe_allow_html=True
        )


def render_overview_page(df=None):

    apply_overview_page_style()

    # ======================================================
    # HEADER
    # ======================================================

    left, right = st.columns([1.8, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="overview-hero">
                <div class="overview-hero-content">
                    <div class="overview-tag">TUGAS AKHIR • MATEMATIKA ITB</div>
                    <h1>Prediksi Harga Jual Rumah</h1>
                    <p class="overview-subtitle">
                        Analisis perbandingan regresi Ridge dan LASSO dalam
                        mengatasi multikolinearitas dan dimensi tinggi pada
                        prediksi harga jual rumah.
                    </p>
                    <p class="overview-meta">
                        Sarjana Matematika · Institut Teknologi Bandung · 2026
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            '<div class="overview-hero-image">',
            unsafe_allow_html=True
        )
        st.image("rumah_iowa.jpg", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # PROJECT INFORMATION
    # ======================================================

    overview_section(
        "Informasi Project",
        "Ringkasan dataset, metode pemodelan, dan metrik evaluasi yang digunakan."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="overview-card">
                <h4>📂 Dataset</h4>
                <p>
                    <b>House Prices: Advanced Regression Techniques</b><br>
                    Sumber: Kaggle
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="overview-card">
                <h4>📊 Model</h4>
                <p>
                    Regresi Linear (OLS)<br>
                    Regresi Ridge<br>
                    Regresi LASSO
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="overview-card">
                <h4>📈 Evaluasi</h4>
                <p>
                    R² Score<br>
                    Mean Absolute Error (MAE)<br>
                    Mean Squared Error (MSE) dan RMSE
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ======================================================
    # BACKGROUND & OBJECTIVE
    # ======================================================

    overview_section("Latar Belakang dan Tujuan")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### 📖 Latar Belakang")
        st.markdown(
            """
            <div style="text-align: justify; line-height: 1.7;">
                Prediksi harga rumah merupakan permasalahan regresi yang
                melibatkan banyak variabel prediktor. Korelasi tinggi antar
                prediktor dapat menimbulkan <b>multikolinearitas</b> dan
                membuat estimasi koefisien kurang stabil.
                <br><br>
                Penelitian ini juga mengevaluasi performa model pada kondisi
                <b>high-dimensional data (p &gt; n)</b>, yaitu ketika jumlah
                prediktor lebih besar daripada jumlah observasi. Kondisi ini
                dibentuk melalui <b>random sampling</b>.
                <br><br>
                Regresi <b>Ridge</b> dan <b>LASSO</b> diterapkan sebagai metode
                regularisasi untuk membandingkan kestabilan dan kemampuan
                prediksi model pada kedua kondisi data tersebut.
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown("#### 🎯 Tujuan Dashboard")
        st.markdown(
            """
            Dashboard ini menyediakan:

            - Eksplorasi karakteristik dataset *House Prices*.
            - Analisis faktor yang berkaitan dengan harga rumah.
            - Perbandingan performa model regresi.
            - Simulasi prediksi harga rumah berdasarkan karakteristik properti.
            """
        )

    st.divider()

    # ======================================================
    # RESEARCH RESULTS
    # ======================================================

    overview_section(
        "Ringkasan Hasil Penelitian",
        "Nilai λ dipilih berdasarkan CV MSE terendah pada data training. "
        "R², MAE, dan RMSE digunakan untuk mengevaluasi performa pada data testing."
    )

    result1, result2 = st.columns(2, gap="large")

    with result1:
        st.markdown(
            """
            <div class="overview-result">
                <div class="result-label">Multikolinearitas</div>
                <h3>🏆 LASSO Regression</h3>
                <div class="result-value">
                    λ optimal: <b>1000</b><br>
                    R²: <b>0.8533</b><br>
                    MAE: <b>$18,923.73</b><br>
                    RMSE: <b>$33,546.96</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with result2:
        st.markdown(
            """
            <div class="overview-result">
                <div class="result-label">High-Dimensional (p &gt; n)</div>
                <h3>🏆 Ridge Regression</h3>
                <div class="result-value">
                    λ optimal: <b>100</b><br>
                    R²: <b>0.8763</b><br>
                    MAE: <b>$20,618.14</b><br>
                    RMSE: <b>$26,310.17</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ======================================================
    # DATASET SUMMARY
    # ======================================================

    overview_section(
        "Ringkasan Dataset",
        "Gambaran umum jumlah observasi dan tipe variabel pada dataset yang dimuat."
    )

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Jumlah Observasi", f"{df.shape[0]:,}")
        c2.metric("Jumlah Kolom", f"{df.shape[1]}")

        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        c3.metric("Variabel Numerik", f"{len(numeric_cols)}")
        c4.metric("Variabel Kategorik", f"{len(categorical_cols)}")

        st.markdown("#### Preview Dataset")
        st.dataframe(df.head(), use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="overview-card">
                <h4>📚 Tentang Dataset</h4>
                <p style="text-align: justify;">
                    Dataset <i>House Prices – Advanced Regression Techniques</i>
                    berisi karakteristik rumah di Ames, Iowa, Amerika Serikat.
                    Dataset mencakup 79 variabel prediktor yang menggambarkan
                    kualitas dan kondisi rumah, luas bangunan, jumlah kamar,
                    serta karakteristik lingkungan. Data ini digunakan untuk
                    menganalisis faktor yang berkaitan dengan harga dan
                    memprediksi harga jual akhir rumah.
                </p>
                <p style="margin-top: 12px;">
                    <b>Sumber:</b> Kaggle — <i>House Prices: Advanced Regression Techniques</i>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.info("Dataset belum dimuat.")