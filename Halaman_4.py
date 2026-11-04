import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# TRAINING MODEL LASSO + PREPROCESSING
# ============================================================

@st.cache_resource
def train_and_cache_lasso(df):
    data = df.copy()

    y = data["SalePrice"].copy()
    X_raw = data.drop(columns=["SalePrice", "Id"], errors="ignore").copy()

    none_features = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType",
        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
        "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2",
    ]

    # Isi missing value yang berarti fasilitas tidak tersedia
    for col in none_features:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col].fillna("None")

    numeric_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_raw.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Split data mentah
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train_raw = X_train_raw.copy()
    X_test_raw = X_test_raw.copy()

    # Imputasi dihitung dari training saja
    numeric_fill = X_train_raw[numeric_cols].median()

    categorical_fill = {}
    for col in categorical_cols:
        mode = X_train_raw[col].mode()
        categorical_fill[col] = mode.iloc[0] if not mode.empty else "None"

    X_train_raw[numeric_cols] = X_train_raw[numeric_cols].fillna(numeric_fill)
    X_test_raw[numeric_cols] = X_test_raw[numeric_cols].fillna(numeric_fill)

    X_train_raw[categorical_cols] = X_train_raw[categorical_cols].fillna(
        categorical_fill
    )
    X_test_raw[categorical_cols] = X_test_raw[categorical_cols].fillna(
        categorical_fill
    )

    # Buat skema kolom dari seluruh kategori data.
    # Kategori langka tetap memiliki kolom meskipun tidak muncul di training.
    X_all_raw = X_raw.copy()
    X_all_raw[numeric_cols] = X_all_raw[numeric_cols].fillna(numeric_fill)
    X_all_raw[categorical_cols] = X_all_raw[categorical_cols].fillna(
        categorical_fill
    )

    X_all_encoded = pd.get_dummies(
        X_all_raw,
        columns=categorical_cols,
        drop_first=True,
        dtype=int,
    )
    encoded_columns = X_all_encoded.columns

    # Encode train dan test dengan skema kolom yang sama
    X_train_encoded = pd.get_dummies(
        X_train_raw,
        columns=categorical_cols,
        drop_first=True,
        dtype=int,
    ).reindex(columns=encoded_columns, fill_value=0)

    X_test_encoded = pd.get_dummies(
        X_test_raw,
        columns=categorical_cols,
        drop_first=True,
        dtype=int,
    ).reindex(columns=encoded_columns, fill_value=0)

    # Standardisasi: scaler hanya di-fit pada training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)

    best_lasso = Lasso(
        alpha=1000,
        max_iter=100000,
        tol=1e-4,
        selection="cyclic",
        random_state=42,
    )
    best_lasso.fit(X_train_scaled, y_train)

    return (
        best_lasso,
        scaler,
        X_train_raw,
        encoded_columns,
        categorical_cols,
        none_features,
        numeric_fill,
        categorical_fill,
        numeric_cols,
    )


# ============================================================
# STYLE HALAMAN PREDIKSI
# ============================================================

def apply_prediction_page_style():
    st.markdown(
        """
        <style>
        /* Lebar konten */
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Header halaman */
        .prediction-hero {
            background: linear-gradient(135deg, #102a43 0%, #1d4e89 100%);
            padding: 2rem 2.2rem;
            border-radius: 18px;
            margin-bottom: 1.4rem;
            color: white;
            box-shadow: 0 8px 24px rgba(16, 42, 67, 0.14);
        }

        .prediction-hero h1 {
            color: white;
            font-size: 2rem;
            font-weight: 750;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.5px;
        }

        .prediction-hero p {
            color: #e7eff8;
            font-size: 1rem;
            margin: 0;
            line-height: 1.6;
        }

        .hero-tag {
            display: inline-block;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: #ffffff;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 0.85rem;
        }

        /* Panel ringkasan model */
        .model-panel {
            background: #f5f8fc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            margin: 0.5rem 0 1.5rem 0;
        }

        .model-panel-title {
            color: #102a43;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .model-panel-caption {
            color: #526579;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        /* Judul section */
        .section-heading {
            color: #102a43;
            font-size: 1.25rem;
            font-weight: 750;
            margin: 0.3rem 0 0.25rem 0;
        }

        .section-caption {
            color: #62758a;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        /* Form */
        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #e3eaf2;
            border-radius: 16px;
            padding: 1.35rem 1.4rem 1.1rem 1.4rem;
            box-shadow: 0 4px 18px rgba(16, 42, 67, 0.045);
        }

        div[data-testid="stForm"] label {
            color: #334e68;
            font-weight: 550;
        }

        /* Tombol submit */
        div[data-testid="stFormSubmitButton"] button {
            background: #1d4e89;
            color: white;
            border: 1px solid #1d4e89;
            border-radius: 10px;
            font-weight: 700;
            min-height: 3rem;
            transition: all 0.2s ease;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: #163f70;
            border-color: #163f70;
            color: white;
        }

        /* Kartu metrik */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e3eaf2;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 16px rgba(16, 42, 67, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #526579;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #102a43;
            font-weight: 750;
        }

        /* Panel insight */
        .insight-heading {
            color: #102a43;
            font-size: 1.1rem;
            font-weight: 750;
            margin-bottom: 0.75rem;
        }

        /* Tabel */
        div[data-testid="stDataFrame"] {
            border: 1px solid #e3eaf2;
            border-radius: 12px;
            overflow: hidden;
        }

        /* Expander */
        div[data-testid="stExpander"] {
            border: 1px solid #e3eaf2;
            border-radius: 12px;
        }

        /* Garis pemisah */
        hr {
            border-color: #e3eaf2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HALAMAN PREDIKSI
# ============================================================

def render_prediction_page(df=None):
    apply_prediction_page_style()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="prediction-hero">
            <div class="hero-tag">HOUSE PRICE ESTIMATOR</div>
            <h1>Estimasi Harga Jual Rumah</h1>
            <p>
                Masukkan spesifikasi rumah untuk memperoleh estimasi harga
                jual berdasarkan model LASSO Regression dengan λ = 1000.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    has_data = (
        df is not None
        and isinstance(df, pd.DataFrame)
        and not df.empty
        and "SalePrice" in df.columns
    )

    if not has_data:
        st.error(
            "Data training tidak tersedia. Model tidak dapat melakukan prediksi."
        )
        return

    (
        best_lasso,
        scaler,
        X_train_raw,
        encoded_columns,
        categorical_cols,
        none_features,
        numeric_fill,
        categorical_fill,
        numeric_cols,
    ) = train_and_cache_lasso(df)

    n_total = len(best_lasso.coef_)
    n_selected = int(np.sum(best_lasso.coef_ != 0))
    n_removed = int(np.sum(best_lasso.coef_ == 0))

    # --------------------------------------------------------
    # RINGKASAN MODEL
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="model-panel">
            <div class="model-panel-title">Ringkasan Model</div>
            <div class="model-panel-caption">
                Model menggunakan LASSO Regression dengan λ (alpha) = 1000.
                Regularisasi LASSO membantu menyederhanakan model dengan
                mengecilkan sebagian koefisien hingga bernilai nol.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_col1, model_col2, model_col3 = st.columns(3)

    model_col1.metric("Fitur Awal", f"{n_total:,}")
    model_col2.metric("Fitur Terpilih", f"{n_selected:,}")
    model_col3.metric("Fitur Tereliminasi", f"{n_removed:,}")

    st.write("")

    # --------------------------------------------------------
    # OPSI INPUT KATEGORIK
    # --------------------------------------------------------

    def get_category_options(column, fallback):
        if column in X_train_raw.columns:
            values = X_train_raw[column].dropna().astype(str).unique().tolist()
            if values:
                return sorted(values)
        return fallback

    neighborhoods = get_category_options("Neighborhood", ["NAmes"])
    kitchen_qual_options = get_category_options(
        "KitchenQual", ["Ex", "Gd", "TA", "Fa", "Po"]
    )
    exter_qual_options = get_category_options(
        "ExterQual", ["Ex", "Gd", "TA", "Fa", "Po"]
    )
    bsmt_qual_options = get_category_options(
        "BsmtQual", ["Ex", "Gd", "TA", "Fa", "Po", "None"]
    )
    central_air_options = get_category_options("CentralAir", ["Y", "N"])

    if "None" not in bsmt_qual_options:
        bsmt_qual_options.append("None")

    def get_numeric_default(column, fallback):
        if column in X_train_raw.columns:
            value = pd.to_numeric(
                X_train_raw[column], errors="coerce"
            ).median()
            if pd.notna(value):
                return int(round(value))
        return fallback

    lot_area_default = get_numeric_default("LotArea", 10000)

    lot_area_max_value = (
        pd.to_numeric(X_train_raw["LotArea"], errors="coerce").max()
        if "LotArea" in X_train_raw.columns
        else np.nan
    )
    lot_area_max = (
        int(lot_area_max_value)
        if pd.notna(lot_area_max_value)
        else 100000
    )
    lot_area_max = max(lot_area_max, lot_area_default, 1)

    bedroom_max_value = (
        pd.to_numeric(X_train_raw["BedroomAbvGr"], errors="coerce").max()
        if "BedroomAbvGr" in X_train_raw.columns
        else np.nan
    )
    bedroom_max = (
        int(bedroom_max_value)
        if pd.notna(bedroom_max_value)
        else 8
    )
    bedroom_max = max(bedroom_max, 1)

    # --------------------------------------------------------
    # FORM INPUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Spesifikasi Rumah</div>'
        '<div class="section-caption">'
        'Lengkapi informasi rumah di bawah ini untuk menghitung estimasi harga.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form(key="house_pred_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🏡 Fasilitas & Lokasi**")

            neighborhood = st.selectbox(
                "Lokasi (Neighborhood)",
                options=neighborhoods,
            )
            bedroom_abv_gr = st.selectbox(
                "Jumlah Kamar Tidur (BedroomAbvGr)",
                options=list(range(0, bedroom_max + 1)),
                index=min(3, bedroom_max),
            )
            full_bath = st.selectbox(
                "Jumlah Kamar Mandi (FullBath)",
                options=[0, 1, 2, 3, 4],
                index=2,
            )
            garage_cars = st.selectbox(
                "Kapasitas Garasi Mobil (GarageCars)",
                options=[0, 1, 2, 3, 4],
                index=2,
            )
            central_air = st.selectbox(
                "Pendingin Udara (CentralAir)",
                options=central_air_options,
                format_func=lambda x: {"Y": "Ya", "N": "Tidak"}.get(x, x),
            )

        with col2:
            st.markdown("**⭐ Kualitas & Kondisi Material**")

            overall_qual = st.slider(
                "Kualitas Material (OverallQual)",
                min_value=1,
                max_value=10,
                value=7,
            )
            overall_cond = st.slider(
                "Kondisi Perawatan Rumah (OverallCond)",
                min_value=1,
                max_value=10,
                value=5,
            )

            quality_labels = {
                "Ex": "Excellent",
                "Gd": "Good",
                "TA": "Typical / Average",
                "Fa": "Fair",
                "Po": "Poor",
                "None": "No Basement",
            }

            kitchen_qual = st.selectbox(
                "Kualitas Dapur (KitchenQual)",
                options=kitchen_qual_options,
                index=(
                    kitchen_qual_options.index("TA")
                    if "TA" in kitchen_qual_options
                    else 0
                ),
                format_func=lambda x: quality_labels.get(x, x),
            )
            exter_qual = st.selectbox(
                "Kualitas Eksterior (ExterQual)",
                options=exter_qual_options,
                index=(
                    exter_qual_options.index("TA")
                    if "TA" in exter_qual_options
                    else 0
                ),
                format_func=lambda x: quality_labels.get(x, x),
            )
            bsmt_qual = st.selectbox(
                "Kualitas Basement (BsmtQual)",
                options=bsmt_qual_options,
                index=(
                    bsmt_qual_options.index("TA")
                    if "TA" in bsmt_qual_options
                    else 0
                ),
                format_func=lambda x: quality_labels.get(x, x),
            )

        with col3:
            st.markdown("**📐 Luas & Tahun Bangunan**")

            gr_liv_area = st.number_input(
                "Luas Hunian (GrLivArea) [sqft]",
                min_value=300,
                max_value=5000,
                value=1500,
                step=50,
            )
            lot_area = st.number_input(
                "Luas Tanah (LotArea) [sqft]",
                min_value=0,
                max_value=lot_area_max,
                value=min(max(lot_area_default, 0), lot_area_max),
                step=100,
            )
            total_bsmt_sf = st.number_input(
                "Luas Basement (TotalBsmtSF) [sqft]",
                min_value=0,
                max_value=3000,
                value=800,
                step=50,
            )
            year_built = st.number_input(
                "Tahun Dibangun (YearBuilt)",
                min_value=1800,
                max_value=2010,
                value=2000,
            )
            year_remod = st.number_input(
                "Tahun Renovasi Terakhir (YearRemodAdd)",
                min_value=1950,
                max_value=2010,
                value=2005,
            )

        submit_button = st.form_submit_button(
            "🚀 Hitung Estimasi Harga Rumah",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # PREDIKSI
    # --------------------------------------------------------

    if submit_button:
        # Gunakan satu baris training sebagai template fitur
        input_raw = X_train_raw.iloc[[0]].copy()

        user_values = {
            "Neighborhood": neighborhood,
            "BedroomAbvGr": bedroom_abv_gr,
            "FullBath": full_bath,
            "GarageCars": garage_cars,
            "CentralAir": central_air,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "KitchenQual": kitchen_qual,
            "ExterQual": exter_qual,
            "BsmtQual": bsmt_qual,
            "GrLivArea": gr_liv_area,
            "LotArea": lot_area,
            "TotalBsmtSF": total_bsmt_sf,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
        }

        for feature, value in user_values.items():
            if feature in input_raw.columns:
                input_raw.loc[:, feature] = value

        input_raw[numeric_cols] = input_raw[numeric_cols].fillna(numeric_fill)
        input_raw[categorical_cols] = input_raw[categorical_cols].fillna(
            categorical_fill
        )

        # Encode input, lalu samakan kolom dan urutannya dengan training
        input_encoded = pd.get_dummies(
            input_raw,
            columns=categorical_cols,
            drop_first=True,
            dtype=int,
        ).reindex(columns=encoded_columns, fill_value=0)

        input_scaled = scaler.transform(input_encoded)
        predicted_price = max(
            0,
            float(best_lasso.predict(input_scaled)[0]),
        )

        # ----------------------------------------------------
        # MEDIAN HARGA LINGKUNGAN
        # ----------------------------------------------------

        neighborhood_prices = (
            df.loc[df["Neighborhood"] == neighborhood, "SalePrice"]
            if "Neighborhood" in df.columns
            else pd.Series(dtype=float)
        )

        median_neighborhood = neighborhood_prices.median()

        if pd.isna(median_neighborhood) or median_neighborhood == 0:
            median_neighborhood = df["SalePrice"].median()

        diff_percent = (
            (predicted_price - median_neighborhood)
            / median_neighborhood
            * 100
            if pd.notna(median_neighborhood) and median_neighborhood != 0
            else 0
        )

        # ----------------------------------------------------
        # HASIL ESTIMASI
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-heading">Hasil Estimasi Harga Jual</div>'
            '<div class="section-caption">'
            'Ringkasan estimasi berdasarkan spesifikasi rumah yang dimasukkan.'
            '</div>',
            unsafe_allow_html=True,
        )

        price_per_sqft = (
            predicted_price / gr_liv_area if gr_liv_area > 0 else 0
        )

        col_res1, col_res2, col_res3 = st.columns(3)

        col_res1.metric(
            "Estimasi Harga Jual",
            f"${predicted_price:,.0f}",
            delta=f"{diff_percent:+.1f}% dari median lingkungan",
        )
        col_res2.metric(
            f"Median Harga {neighborhood}",
            f"${median_neighborhood:,.0f}",
        )
        col_res3.metric(
            "Harga per sqft",
            f"${price_per_sqft:,.2f}",
        )

        st.write("")

        # ----------------------------------------------------
        # INSIGHT PROPERTI DAN GAUGE
        # ----------------------------------------------------

        col_badge, col_gauge = st.columns([1.5, 1], gap="large")

        with col_badge:
            st.markdown(
                '<div class="section-heading">Insight Properti</div>',
                unsafe_allow_html=True,
            )

            if overall_qual >= 7 and diff_percent <= 5:
                st.success(
                    "🔥 **Hot Deal / Permata Tersembunyi!**\n\n"
                    "Rumah ini memiliki kualitas material yang sangat baik, "
                    "namun estimasi harganya tergolong bersahabat atau "
                    "berada di bawah rata-rata lingkungannya."
                )
            elif overall_qual >= 8 and diff_percent > 20:
                st.info(
                    "🌟 **Properti Premium.**\n\n"
                    "Harga yang cukup tinggi di atas median area sejalan "
                    "dengan spesifikasi bangunan yang tergolong mewah."
                )
            elif overall_qual <= 4 and diff_percent > 0:
                st.warning(
                    "⚠️ **Perhatian.**\n\n"
                    "Estimasi harga berada di atas median lingkungan, "
                    "sedangkan kualitas material berada di bawah standar. "
                    "Pertimbangkan fitur lain yang mungkin memengaruhi harga."
                )
            else:
                st.info(
                    "✅ **Sesuai Harga Pasar.**\n\n"
                    "Estimasi harga properti ini relatif selaras dengan "
                    "median harga di lingkungan tersebut."
                )

        with col_gauge:
            st.markdown(
                '<div class="section-heading">Harga per Luas Hunian</div>',
                unsafe_allow_html=True,
            )

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=price_per_sqft,
                    domain={"x": [0, 1], "y": [0, 1]},
                    number={"prefix": "$", "valueformat": ",.2f"},
                    gauge={
                        "axis": {
                            "range": [0, 250],
                            "tickwidth": 1,
                            "tickcolor": "#526579",
                        },
                        "bar": {
                            "color": "#1d4e89",
                            "thickness": 0.22,
                        },
                        "bgcolor": "white",
                        "borderwidth": 1,
                        "bordercolor": "#d9e2ec",
                        "steps": [
                            {"range": [0, 100], "color": "#A7E8C0"},
                            {"range": [100, 150], "color": "#FFE08A"},
                            {"range": [150, 250], "color": "#F5B0B0"},
                        ],
                    },
                )
            )

            fig.update_layout(
                margin=dict(l=10, r=10, t=25, b=5),
                height=230,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="Arial, sans-serif",
                    color="#102a43",
                ),
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # ----------------------------------------------------
        # PENJELASAN INDIKATOR
        # ----------------------------------------------------

        with st.expander("ℹ️ Penjelasan Indikator & Status Properti"):
            st.markdown(
                """
                **1. Smart Badge (Label Pintar)**

                Label mengevaluasi skor kualitas bangunan dan persentase
                selisih estimasi harga rumah terhadap median harga di
                lingkungannya.

                - 🔥 **Hot Deal / Permata Tersembunyi:** kualitas material
                  tinggi dengan estimasi harga relatif bersahabat dibandingkan
                  median lingkungan.
                - 🌟 **Properti Premium:** kualitas material tinggi dengan
                  estimasi harga jauh di atas median lingkungan.
                - ⚠️ **Perhatian:** estimasi harga di atas median lingkungan,
                  tetapi kualitas material relatif rendah.
                - ✅ **Sesuai Harga Pasar:** estimasi harga relatif selaras
                  dengan median harga di lingkungan tersebut.

                **2. Gauge Chart (Speedometer)**

                Grafik menampilkan estimasi harga per sqft. Rentang warna
                merupakan kategori visual pada dashboard, bukan standar harga
                universal:

                - **Hijau ($0–$100):** rentang harga satuan lebih rendah.
                - **Kuning ($100–$150):** rentang harga satuan menengah.
                - **Merah muda ($150–$250):** rentang harga satuan lebih tinggi.
                """
            )

        st.divider()

        # ----------------------------------------------------
        # RINGKASAN SPESIFIKASI
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-heading">Ringkasan Spesifikasi Properti</div>'
            '<div class="section-caption">'
            'Detail karakteristik rumah yang digunakan dalam estimasi.'
            '</div>',
            unsafe_allow_html=True,
        )

        kategori_kualitas = {
            "Ex": "Sangat Baik (Premium)",
            "Gd": "Baik (Di Atas Rata-rata)",
            "TA": "Standar (Rata-rata)",
            "Fa": "Cukup (Di Bawah Standar)",
            "Po": "Buruk",
            "None": "Tidak Ada",
        }

        summary = pd.DataFrame(
            {
                "Karakteristik": [
                    "Kualitas Material Bawaan",
                    "Kondisi Perawatan Rumah",
                    "Kualitas Dapur",
                    "Kualitas Eksterior",
                    "Kualitas Basement",
                    "Luas Hunian Utama",
                    "Luas Tanah",
                    "Luas Total Basement",
                    "Jumlah Kamar Tidur",
                    "Kapasitas Garasi",
                    "Jumlah Kamar Mandi",
                    "Pendingin Udara",
                    "Tahun Dibangun",
                    "Tahun Renovasi Terakhir",
                    "Lokasi (Lingkungan)",
                ],
                "Nilai": [
                    f"Skor {overall_qual} / 10",
                    f"Skor {overall_cond} / 10",
                    kategori_kualitas.get(kitchen_qual, kitchen_qual),
                    kategori_kualitas.get(exter_qual, exter_qual),
                    kategori_kualitas.get(bsmt_qual, bsmt_qual),
                    f"{gr_liv_area:,} sqft",
                    f"{lot_area:,} sqft",
                    f"{total_bsmt_sf:,} sqft",
                    f"{bedroom_abv_gr}",
                    f"{garage_cars} Mobil",
                    f"{full_bath}",
                    "Ya" if central_air == "Y" else "Tidak",
                    year_built,
                    year_remod,
                    neighborhood,
                ],
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Karakteristik": st.column_config.TextColumn(
                    "Karakteristik",
                    width="medium",
                ),
                "Nilai": st.column_config.TextColumn(
                    "Spesifikasi",
                    width="large",
                ),
            },
        )
