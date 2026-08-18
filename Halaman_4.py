import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.graph_objects as go 

# ============================================================
# CACHING MODEL TRAINING (Hanya berjalan 1 kali)
# ============================================================
@st.cache_resource
def train_and_cache_lasso(df):
    data = df.copy()

    none_features = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "MasVnrType",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
    ]

    for col in none_features:
        if col in data.columns:
            data[col] = data[col].fillna("None")

    if "GarageYrBlt" in data.columns:
        data["GarageYrBlt"] = data["GarageYrBlt"].fillna(0)

    if "LotFrontage" in data.columns:
        data["LotFrontage"] = data["LotFrontage"].fillna(
            data["LotFrontage"].median()
        )

    if "MasVnrArea" in data.columns:
        data["MasVnrArea"] = data["MasVnrArea"].fillna(0)

    if "Electrical" in data.columns:
        data["Electrical"] = data["Electrical"].fillna(
            data["Electrical"].mode()[0]
        )

    target = "SalePrice"
    X_raw = data.drop(columns=[target], errors="ignore")

    if "Id" in X_raw.columns:
        X_raw = X_raw.drop(columns=["Id"])

    categorical_cols = X_raw.select_dtypes(exclude=np.number).columns.tolist()

    X_encoded = pd.get_dummies(
        X_raw, columns=categorical_cols, drop_first=True, dtype=int
    )

    y = data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    best_lasso = Lasso(alpha=1000, max_iter=10000, random_state=42)
    best_lasso.fit(X_scaled, y)

    return (
        best_lasso,
        scaler,
        X_raw,
        X_encoded.columns,
        categorical_cols,
        none_features,
    )


def render_prediction_page(df=None):

    # ============================================================
    # HEADER
    # ============================================================
    st.header("🔮 Real-Time House Price Estimator")
    st.caption(
        "Masukkan spesifikasi utama rumah untuk mendapatkan estimasi "
        "harga jual berdasarkan Lasso Regression dengan λ = 1000."
    )
    st.divider()

    # ============================================================
    # VALIDASI DATA
    # ============================================================
    has_data = (
        df is not None and isinstance(df, pd.DataFrame) and not df.empty
    )

    if not has_data:
        st.error(
            "Data training tidak tersedia. Model tidak dapat melakukan prediksi."
        )
        return

    # Load / Train model dari cache
    best_lasso, scaler, X_raw, encoded_columns, categorical_cols, none_features = (
        train_and_cache_lasso(df)
    )

    # ============================================================
    # INFORMASI MODEL
    # ============================================================
    n_total = len(best_lasso.coef_)
    n_selected = np.sum(best_lasso.coef_ != 0)
    n_removed = np.sum(best_lasso.coef_ == 0)

    st.info(
        f"""
        ℹ️ **Model Prediksi**
        
        Model yang digunakan adalah **Lasso Regression** dengan **λ (alpha) = 1000**.
        - Fitur awal: **{n_total}**
        - Fitur terpilih (koefisien ≠ 0): **{n_selected}**
        - Fitur tereliminasi (koefisien = 0): **{n_removed}**
        """
    )

    # Neighborhood options
    neighborhoods = (
        sorted(df["Neighborhood"].dropna().unique().tolist())
        if "Neighborhood" in df.columns
        else ["NAmes"]
    )

    # ============================================================
    # FORM INPUT PENGGUNA
    # ============================================================
    with st.form(key="house_pred_form"):
        st.subheader("📋 Spesifikasi Rumah")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📐 Dimensi & Ukuran**")
            gr_liv_area = st.number_input(
                "Luas Hunian (GrLivArea) [sqft]",
                min_value=300,
                max_value=5000,
                value=1500,
                step=50,
            )
            total_bsmt_sf = st.number_input(
                "Luas Basement (TotalBsmtSF) [sqft]",
                min_value=0,
                max_value=3000,
                value=800,
                step=50,
            )
            garage_cars = st.selectbox(
                "Kapasitas Garasi Mobil (GarageCars)",
                options=[0, 1, 2, 3, 4],
                index=2,
            )
            full_bath = st.selectbox(
                "Kamar Mandi (FullBath)", options=[0, 1, 2, 3, 4], index=2
            )

        with col2:
            st.markdown("**⭐ Kualitas & Kondisi Material**")
            overall_qual = st.slider(
                "Kualitas Material (OverallQual)", min_value=1, max_value=10, value=7
            )
            overall_cond = st.slider(
                "Kondisi Perawatan Rumah (OverallCond)", min_value=1, max_value=10, value=5
            )

            # Fitur kategorik terkuat ditambahkan ke form
            kitchen_qual = st.selectbox(
                "Kualitas Dapur (KitchenQual)",
                options=["Ex", "Gd", "TA", "Fa"],
                index=2,
                format_func=lambda x: {
                    "Ex": "Excellent",
                    "Gd": "Good",
                    "TA": "Typical / Average",
                    "Fa": "Fair",
                }.get(x, x),
            )
            exter_qual = st.selectbox(
                "Kualitas Eksterior (ExterQual)",
                options=["Ex", "Gd", "TA", "Fa"],
                index=2,
                format_func=lambda x: {
                    "Ex": "Excellent",
                    "Gd": "Good",
                    "TA": "Typical / Average",
                    "Fa": "Fair",
                }.get(x, x),
            )

        with col3:
            st.markdown("**📅 Waktu & Struktur**")
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
            bsmt_qual = st.selectbox(
                "Tinggi/Kualitas Basement (BsmtQual)",
                options=["Ex", "Gd", "TA", "Fa", "None"],
                index=2,
                format_func=lambda x: {
                    "Ex": "Excellent (100+ in)",
                    "Gd": "Good (90-99 in)",
                    "TA": "Typical (80-89 in)",
                    "Fa": "Fair (70-79 in)",
                    "None": "No Basement",
                }.get(x, x),
            )
            neighborhood = st.selectbox(
                "Lokasi (Neighborhood)", options=neighborhoods
            )

        submit_button = st.form_submit_button(
            "🚀 Hitung Estimasi Harga Rumah", use_container_width=True
        )

    # ============================================================
    # PREDICTION
    # ============================================================
    if submit_button:
        # Base input row
        input_raw = X_raw.iloc[[0]].copy()

        # Isi fitur lain yang tidak ada di form dengan median / mode
        for col in input_raw.columns:
            if pd.api.types.is_numeric_dtype(X_raw[col]):
                input_raw[col] = X_raw[col].median()
            else:
                mode_value = X_raw[col].mode()
                input_raw[col] = (
                    mode_value.iloc[0] if len(mode_value) > 0 else "None"
                )

        # Masukkan input pengguna
        user_values = {
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "GarageCars": garage_cars,
            "FullBath": full_bath,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "KitchenQual": kitchen_qual,
            "ExterQual": exter_qual,
            "BsmtQual": bsmt_qual,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "Neighborhood": neighborhood,
        }

        for feature, value in user_values.items():
            if feature in input_raw.columns:
                input_raw.loc[:, feature] = value

        # Handle missing value input pengguna
        for col in none_features:
            if col in input_raw.columns:
                input_raw[col] = input_raw[col].fillna("None")

        if "GarageYrBlt" in input_raw.columns:
            input_raw["GarageYrBlt"] = input_raw["GarageYrBlt"].fillna(0)
        if "LotFrontage" in input_raw.columns:
            input_raw["LotFrontage"] = input_raw["LotFrontage"].fillna(
                X_raw["LotFrontage"].median()
            )
        if "MasVnrArea" in input_raw.columns:
            input_raw["MasVnrArea"] = input_raw["MasVnrArea"].fillna(0)
        if "Electrical" in input_raw.columns:
            input_raw["Electrical"] = input_raw["Electrical"].fillna(
                X_raw["Electrical"].mode()[0]
            )

        # One-Hot Encoding & Alignment
        input_encoded = pd.get_dummies(
            input_raw, columns=categorical_cols, drop_first=True, dtype=int
        )
        input_encoded = input_encoded.reindex(
            columns=encoded_columns, fill_value=0
        )

        # Standardisasi & Prediksi
        input_scaled = scaler.transform(input_encoded)
        predicted_price = max(0, best_lasso.predict(input_scaled)[0])

        # Median Area Comparison
        if "Neighborhood" in df.columns:
            median_neighborhood = df.loc[
                df["Neighborhood"] == neighborhood, "SalePrice"
            ].median()
        else:
            median_neighborhood = df["SalePrice"].median()

        diff_percent = (
            (
                (predicted_price - median_neighborhood)
                / median_neighborhood
                * 100
            )
            if pd.notna(median_neighborhood) and median_neighborhood != 0
            else 0
        )

        # ============================================================
        # TAMPILAN HASIL
        # ============================================================
        st.divider()
        st.subheader("📊 Hasil Estimasi Harga Jual")

        # Menghitung Harga per sqft terlebih dahulu agar bisa dipakai di metrik dan chart
        price_per_sqft = (
            predicted_price / gr_liv_area if gr_liv_area > 0 else 0
        )

        # 1. Menampilkan Metrik Utama
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric(
            "💰 Estimasi Harga Jual",
            f"${predicted_price:,.0f}",
            delta=f"{diff_percent:+.1f}% dari Harga Lingkungan", 
        )
        col_res2.metric(
            f"📍 Median Harga {neighborhood}", 
            f"${median_neighborhood:,.0f}"
        )
        col_res3.metric(
            "📐 Harga per sqft", 
            f"${price_per_sqft:,.2f}"
        )

        st.write("") # Memberi sedikit jarak

        # ============================================================
        # FITUR TAMBAHAN: SMART BADGE & GAUGE CHART
        # ============================================================
        col_badge, col_gauge = st.columns([1.5, 1])

        # --- BAGIAN KIRI: Smart Badge (Label Pintar) ---
        with col_badge:
            st.markdown("### 💡 Insight Properti")
            
            # --- Trik CSS untuk meratakan teks (justify) di dalam st.success, st.info, st.warning ---
            st.markdown("""
                <style>
                [data-testid="stAlert"] {
                    text-align: justify;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Logika If-Else untuk menentukan sentimen harga dan kualitas
            if overall_qual >= 7 and diff_percent <= 5:
                st.success("🔥 **Hot Deal / Permata Tersembunyi!** \n\nRumah ini memiliki kualitas material yang sangat baik (Premium) namun ditawarkan dengan harga yang tergolong bersahabat atau di bawah rata-rata lingkungannya.")
            elif overall_qual >= 8 and diff_percent > 20:
                st.info("🌟 **Properti Premium.** \n\nHarga yang cukup tinggi di atas rata-rata area sangat sepadan dengan spesifikasi bangunan yang tergolong mewah.")
            elif overall_qual <= 4 and diff_percent > 0:
                st.warning("⚠️ **Perhatian.** \n\nHarga estimasi berada di atas median lingkungan, namun kualitas material berada di bawah standar. Pertimbangkan fitur lain yang mungkin mendongkrak harga.")
            else:
                st.info("✅ **Sesuai Harga Pasar.** \n\nHarga estimasi properti ini cukup wajar dan selaras dengan rata-rata tren pasar di lingkungan tersebut.")
                
        # --- BAGIAN KANAN: Gauge Chart (Speedometer Harga/sqft) ---
        with col_gauge:
            # Membuat visualisasi Speedometer menggunakan Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = price_per_sqft,
                domain = {'x': [0, 1], 'y': [0, 1]},
                number = {'prefix': "$", 'valueformat': ",.2f"},
                gauge = {
                    'axis': {'range': [None, 250], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "rgba(0,0,0,0.5)", 'thickness': 0.2}, # Jarum penunjuk
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 100], 'color': "#a8e6cf"},   # Hijau (Murah)
                        {'range': [100, 150], 'color': "#ffd3b6"}, # Kuning (Menengah)
                        {'range': [150, 250], 'color': "#ffaaa5"}  # Merah (Premium)
                    ],
                }
            ))
            
            # Atur margin agar chart muat sempurna di dalam kolom
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=200
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- TAMBAHAN: Expander Penjelasan ---
        st.write("") # Memberikan sedikit jarak vertikal
        with st.expander("ℹ️ Penjelasan Indikator & Status Properti"):
            st.markdown("""
            <div style="text-align: justify;">

            **1. Smart Badge (Label Pintar)** Label ini mengevaluasi interaksi antara skor kualitas bawaan bangunan dan persentase selisih harga rumah terhadap nilai tengah lingkungannya.
            * 🔥 **Hot Deal / Permata Tersembunyi!** Rumah ini memiliki kualitas material yang sangat baik (Premium) namun ditawarkan dengan harga yang tergolong bersahabat atau di bawah rata-rata lingkungannya.
            * 🌟 **Properti Premium.** Harga yang cukup tinggi di atas rata-rata area sangat sepadan dengan spesifikasi bangunan yang tergolong mewah.
            * ⚠️ **Perhatian.** Harga estimasi berada di atas median lingkungan, namun kualitas material berada di bawah standar. Pertimbangkan fitur lain yang mungkin mendongkrak harga.
            * ✅ **Sesuai Harga Pasar.** Harga estimasi properti ini cukup wajar dan selaras dengan rata-rata tren pasar di lingkungan tersebut.
            
            **2. Gauge Chart (Speedometer)** Grafik ini difokuskan untuk menyorot angka **Harga per sqft** dari properti yang sedang diestimasi. Desain pewarnaan pada grafik telah disesuaikan dengan rentang wajar harga rumah pada dataset Ames Housing:
            * **Hijau ( :green[0] - :green[100] ):** Mengindikasikan harga satuan yang tergolong murah.
            * **Kuning ( :green[100] - :green[150] ):** Mengindikasikan harga satuan yang berada pada rentang menengah atau wajar.
            * **Merah Muda ( :green[150] - :green[250] ):** Mengindikasikan bahwa harga satuan properti tersebut sudah masuk dalam kategori premium.
            
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ============================================================
        # RINGKASAN PROPERTI
        # ============================================================
        st.markdown("### 🏠 Ringkasan Spesifikasi Properti")
        
        # Kamus pemetaan (mapping) untuk mengubah kode mentah menjadi bahasa Indonesia yang ramah pembaca
        kategori_kualitas = {
            "Ex": "Sangat Baik (Premium)",
            "Gd": "Baik (Di Atas Rata-rata)",
            "TA": "Standar (Rata-rata)",
            "Fa": "Cukup (Di Bawah Standar)",
            "None": "Tidak Ada"
        }
        
        summary = pd.DataFrame(
            {
                "Karakteristik": [
                    "Kualitas Material Bawaan",
                    "Kondisi Perawatan Rumah",
                    "Kualitas Dapur",
                    "Kualitas Eksterior",
                    "Tinggi & Kualitas Basement",
                    "Luas Area Hunian Utama",
                    "Luas Total Basement",
                    "Kapasitas Garasi",
                    "Jumlah Kamar Mandi",
                    "Tahun Dibangun",
                    "Tahun Renovasi Terakhir",
                    "Lokasi (Lingkungan)",
                ],
                "Nilai": [
                    f"Skor {overall_qual} / 10",
                    f"Skor {overall_cond} / 10",
                    kategori_kualitas.get(kitchen_qual, kitchen_qual), # Otomatis menerjemahkan 'TA' dkk
                    kategori_kualitas.get(exter_qual, exter_qual),
                    kategori_kualitas.get(bsmt_qual, bsmt_qual),
                    f"{gr_liv_area:,} sqft",
                    f"{total_bsmt_sf:,} sqft",
                    f"{garage_cars} Mobil",
                    f"{full_bath}", # Menghilangkan kata 'Ruangan' agar lebih natural
                    year_built,
                    year_remod,
                    neighborhood,
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)