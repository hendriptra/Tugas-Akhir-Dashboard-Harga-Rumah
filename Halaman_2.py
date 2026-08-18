import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def render_eda_page(df):

    st.header("📊 House Sales Insight")

    st.caption(
        "Eksplorasi pola dan faktor-faktor yang memengaruhi harga jual rumah."
    )

    # ============================================================
    # 1. HOUSE PRICE OVERVIEW
    # ============================================================

    st.subheader("1. House Price Overview")

    st.caption(
        "Ringkasan harga jual rumah pada dataset berdasarkan jumlah observasi, "
        "rata-rata, nilai minimum, nilai maksimum, dan median."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "🏠 Total Properties",
        f"{df.shape[0]:,}"
    )

    col2.metric(
        "💰 Average Price",
        f"${df['SalePrice'].mean():,.0f}"
    )

    col3.metric(
        "📉 Lowest Price",
        f"${df['SalePrice'].min():,.0f}" 
    )

    col4.metric(
        "📈 Highest Price",
        f"${df['SalePrice'].max():,.0f}"
    )

    col5.metric(
        "🏡 Median Price",
        f"${df['SalePrice'].median():,.0f}"
    )

    st.divider()

    # ============================================================
    # 2. PRICE DISTRIBUTION
    # ============================================================

    st.subheader("2. Price Distribution")

    st.caption(
        "Melihat sebaran harga rumah dan rentang harga yang paling banyak ditemui."
    )

    col1, col2 = st.columns(2)

    # ============================================================
    # 2A. PRICE DISTRIBUTION
    # ============================================================

    with col1:

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            df["SalePrice"],
            kde=True,
            bins=30,
            color="#4C78A8",
            edgecolor="white",
            alpha=0.85,
            ax=ax
        )

        mean_price = df["SalePrice"].mean()
        median_price = df["SalePrice"].median()

        ax.axvline(
            mean_price,
            color="#F28E2B",
            linestyle="--",
            linewidth=2,
            label=f"Mean: ${mean_price:,.0f}"
        )

        ax.axvline(
            median_price,
            color="#2A9D8F",
            linestyle=":",
            linewidth=2.5,
            label=f"Median: ${median_price:,.0f}"
        )

        ax.set_title(
            "Distribution of Sale Prices",
            fontsize=13,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Sale Price ($)"
        )

        ax.set_ylabel(
            "Number of Houses"
        )

        ax.legend(
            frameon=False
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)

    # ============================================================
    # 2B. PRICE SEGMENT
    # ============================================================

    with col2:

        price_segment = pd.cut(
            df["SalePrice"],
            bins=[
                0,
                100000,
                200000,
                300000,
                np.inf
            ],
            labels=[
                "< $100K",
                "$100K - $200K",
                "$200K - $300K",
                "> $300K"
            ]
        )

        segment = (
            price_segment
            .value_counts()
            .sort_index()
            .reset_index()
        )

        segment.columns = [
            "Price Range",
            "Number of Houses"
        ]

        segment["Percentage"] = (
            segment["Number of Houses"]
            / len(df)
            * 100
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        colors = [
            "#AFC7E3",
            "#4C78A8",
            "#2A9D8F",
            "#F28E2B"
        ]

        bars = ax.bar(
            segment["Price Range"],
            segment["Percentage"],
            color=colors,
            width=0.65
        )

        ax.set_title(
            "House Price Segments",
            fontsize=13,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Price Range"
        )

        ax.set_ylabel(
            "Percentage of Houses (%)"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Label persentase
        for bar, value in zip(
            bars,
            segment["Percentage"]
        ):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

        plt.xticks(rotation=10)

        st.pyplot(fig)

    # ============================================================
    # 2C. PRICE INSIGHT
    # ============================================================

    median_price = df["SalePrice"].median()
    mean_price = df["SalePrice"].mean()

    skewness = df["SalePrice"].skew()

    largest_segment = segment.loc[
        segment["Number of Houses"].idxmax()
    ]

    premium_houses = (
        df["SalePrice"] > 300000
    ).sum()

    premium_percentage = (
        premium_houses / len(df) * 100
    )

    if skewness > 0.5:

        distribution_text = (
            "Harga rumah cenderung terkonsentrasi pada harga yang lebih rendah, "
            "dengan sebagian kecil rumah memiliki harga yang jauh lebih tinggi."
        )

    else:

        distribution_text = (
            "Harga rumah memiliki penyebaran yang relatif seimbang."
        )

    # Tambahkan .replace('$', '\\$') pada largest_segment['Price Range']
    price_range_clean = str(largest_segment['Price Range']).replace('$', '\\$')

    st.info(
        f"""
    💡 **Price Insight**

    - **Nilai Tengah & Rata-rata:** Median harga rumah berada di angka **\\${median_price/1000:.0f}K**, sementara rata-ratanya mencapai **\\${mean_price/1000:.1f}K**.
    - **Rentang Dominan:** Sebagian besar rumah terpusat pada kelompok harga **{price_range_clean}**, yaitu mencakup **{largest_segment['Percentage']:.1f}%** dari total sampel.
    - **Kategori Premium:** Rumah dengan harga di atas **\\$300K** tergolong sedikit, hanya mencakup **{premium_percentage:.1f}%** dari keseluruhan data.
    - **Bentuk Distribusi Data:** Distribusi harga rumah **menceng ke kanan (*right-skewed*)**. Artinya, mayoritas rumah berada di kisaran harga menengah ke bawah, sedangkan rumah berharga sangat tinggi jumlahnya relatif sedikit.
        """
    )

    st.divider()
    
    # ============================================================
    # 3. PREMIUM PROPERTY INSIGHT
    # ============================================================
    st.subheader("3. Premium Property Insight")

    if df is not None and "SalePrice" in df.columns:
        # 1. Menentukan Batas Kuantil 0.9 (Top 10% Termahal)
        premium_threshold = df["SalePrice"].quantile(0.9)
        premium_df = df[df["SalePrice"] >= premium_threshold]
        normal_df = df[df["SalePrice"] < premium_threshold]

        # Menggunakan &#36; agar simbol Dolar tidak pernah memicu LaTeX
        st.markdown(
            f"""
            <div style="text-align: justify; color: inherit; line-height: 1.6;">
                Keberadaan properti kategori <b>Premium</b> merupakan penyebab utama mengapa distribusi harga rumah 
                <b>menceng ke kanan (<i>right-skewed</i>)</b>. Secara statistik, segmen ini didefinisikan menggunakan nilai 
                <b>Persentil ke-90 (Kuantil 0.9)</b>, yaitu 10% rumah dengan harga jual tertinggi pada dataset 
                (harga &ge; <b>&#36;{premium_threshold:,.0f}</b>). Analisis ini membandingkan rata-rata karakteristik fisik 
                antara rumah segmen biasa (90% populasi) dengan rumah segmen premium untuk melihat variabel yang memicu ekor distribusi kanan tersebut.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Menghitung Rata-rata Variabel Kunci
        comp_data = {
            "Kategori": ["Rumah Biasa (< 90%)", "Rumah Premium (≥ 90%)"],
            "Kualitas Bangunan (1-10)": [
                normal_df["OverallQual"].mean(),
                premium_df["OverallQual"].mean()
            ],
            "Kapasitas Garasi (Mobil)": [
                normal_df["GarageCars"].mean(),
                premium_df["GarageCars"].mean()
            ],
            "Luas Area Hunian (sqft)": [
                normal_df["GrLivArea"].mean(),
                premium_df["GrLivArea"].mean()
            ],
            "Rata-rata Harga Jual": [
                normal_df["SalePrice"].mean(),
                premium_df["SalePrice"].mean()
            ]
        }
        
        comp_df = pd.DataFrame(comp_data)

        # 3. Tampilan Tabel Perbandingan
        st.markdown("##### 📊 Tabel Perbandingan Karakteristik Rata-Rata")
        
        display_df = comp_df.copy()
        display_df["Kualitas Bangunan (1-10)"] = display_df["Kualitas Bangunan (1-10)"].map("{:.2f} / 10".format)
        display_df["Kapasitas Garasi (Mobil)"] = display_df["Kapasitas Garasi (Mobil)"].map("{:.2f} mobil".format)
        display_df["Luas Area Hunian (sqft)"] = display_df["Luas Area Hunian (sqft)"].map("{:,.0f} sqft".format)
        display_df["Rata-rata Harga Jual"] = display_df["Rata-rata Harga Jual"].map("${:,.0f}".format)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # 4. Visualisasi Grafik Subplots (Memisahkan Skala)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        sns.set_theme(style="whitegrid")

        # Visualisasi 1: Quality
        sns.barplot(
            data=comp_df, x="Kategori", y="Kualitas Bangunan (1-10)", 
            ax=axes[0], palette=["#4C72B0", "#DD8452"]
        )
        axes[0].set_title("Kualitas Bangunan (Overall Quality)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Skala (1-10)")
        axes[0].set_xlabel("")
        axes[0].set_ylim(0, 10)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() - 1.2),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=10)

        # Visualisasi 2: Garage
        sns.barplot(
            data=comp_df, x="Kategori", y="Kapasitas Garasi (Mobil)", 
            ax=axes[1], palette=["#4C72B0", "#DD8452"]
        )
        axes[1].set_title("Kapasitas Garasi", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Jumlah Mobil")
        axes[1].set_xlabel("")
        axes[1].set_ylim(0, 4)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() - 0.4),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=10)

        # Visualisasi 3: Living Area
        sns.barplot(
            data=comp_df, x="Kategori", y="Luas Area Hunian (sqft)", 
            ax=axes[2], palette=["#4C72B0", "#DD8452"]
        )
        axes[2].set_title("Luas Area Hunian (GrLivArea)", fontsize=11, fontweight="bold")
        axes[2].set_ylabel("Luas (sqft)")
        axes[2].set_xlabel("")
        for p in axes[2].patches:
            axes[2].annotate(f"{p.get_height():,.0f} sqft", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)

        # 5. Insight Box (Sudah Diperbaiki & Diparafrase)
        diff_price = comp_df.loc[1, "Rata-rata Harga Jual"] / comp_df.loc[0, "Rata-rata Harga Jual"]
        diff_area = (comp_df.loc[1, "Luas Area Hunian (sqft)"] / comp_df.loc[0, "Luas Area Hunian (sqft)"] - 1) * 100

        avg_normal_price = comp_df.loc[0, 'Rata-rata Harga Jual']
        avg_premium_price = comp_df.loc[1, 'Rata-rata Harga Jual']

        st.success(
            f"""
            💡 **Insight Utama — Properti Premium:**

            - **Alasan Pemilihan Variabel:** Tiga fitur di atas (*OverallQual*, *GrLivArea*, dan *GarageCars*) dipilih karena secara statistik memiliki **korelasi paling kuat terhadap harga jual rumah (*SalePrice*)** dibandingkan seluruh variabel prediktor lainnya pada dataset.
            - **Lonjakan Harga Jual:** Rata-rata harga rumah segmen premium mencapai **&#36;{avg_premium_price:,.0f}**, atau sekitar **{diff_price:.1f} kali lipat** lebih tinggi dibandingkan rata-rata rumah segmen biasa (&#36;{avg_normal_price:,.0f}).
            - **Keunggulan Kualitas Material:** Skor *Overall Quality* rumah premium berada pada rata-rata **{comp_df.loc[1, 'Kualitas Bangunan (1-10)']:.2f} dari 10** (kategori sangat baik), jauh di atas rumah biasa yang rata-ratanya hanya **{comp_df.loc[0, 'Kualitas Bangunan (1-10)']:.2f} dari 10**.
            - **Ekspansi Luas Bangunan:** Area ruang huni (*GrLivArea*) pada segmen premium rata-rata **{diff_area:.1f}% lebih luas** dibandingkan rumah biasa.
            """
        )
    else:
        st.info("Dataset tidak ditemukan atau kolom 'SalePrice' belum tersedia.")

    st.divider()

    # ============================================================
    # 4. KEY FACTORS AFFECTING PROPERTY VALUE
    # ============================================================
    
    st.subheader("4. Key Factors Affecting Property Value")

    if df is not None and "SalePrice" in df.columns:
        numeric_df = df.select_dtypes(include=np.number)

        # 1. Kontrol Interaktif untuk User (Slider / Selectbox)
        col_select, _ = st.columns([1, 2])
        with col_select:
            top_n = st.slider(
                "Pilih jumlah fitur teratas yang ingin ditampilkan:",
                min_value=5,
                max_value=20,
                value=10,
                step=5
            )

        # 2. Hitung Korelasi Pearson terhadap SalePrice
        corr_series = (
            numeric_df
            .corr()["SalePrice"]
            .drop("SalePrice")
            .sort_values(key=abs, ascending=False)
            .head(top_n)
            # Diurutkan ascending agar saat ditampilkan di Plotly (barh), nilai terbesar berada di PALING ATAS
            .sort_values(ascending=True)
        )

        df_corr = pd.DataFrame({
            "Feature": corr_series.index,
            "Correlation": corr_series.values
        })

        # 3. Visualisasi Interaktif Menggunakan Plotly
        fig = px.bar(
            df_corr,
            x="Correlation",
            y="Feature",
            orientation="h",
            title=f"Top {top_n} Most Influential Property Features on SalePrice",
            text=df_corr["Correlation"].apply(lambda x: f"{x:.2f}"),
            color="Correlation",
            color_continuous_scale="Blues",
            labels={"Correlation": "Correlation Coefficient (r)", "Feature": "Fitur Prediktor"}
        )

        # Kustomisasi Tampilan Plotly
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Nilai Korelasi: <b>%{x:.3f}</b><extra></extra>"
        )

        fig.update_layout(
            height=max(400, top_n * 35), # Tinggi grafik otomatis menyesuaikan jumlah fitur
            xaxis=dict(range=[0, 1.0], title="Correlation Coefficient (r)"),
            yaxis=dict(title=""),
            coloraxis_showscale=False,  # Sembunyikan colorbar bawaan agar bersih
            margin=dict(l=20, r=40, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        # Tampilkan Grafik Interaktif
        st.plotly_chart(fig, use_container_width=True)

        # 4. Insight Box Dinamis
        # Mengambil 3 fitur teratas untuk narasi otomatis
        top1_feat = df_corr.iloc[-1]["Feature"]
        top1_corr = df_corr.iloc[-1]["Correlation"]
        top2_feat = df_corr.iloc[-2]["Feature"]
        top2_corr = df_corr.iloc[-2]["Correlation"]
        top3_feat = df_corr.iloc[-3]["Feature"]
        top3_corr = df_corr.iloc[-3]["Correlation"]

        st.success(
            f"""
            💡 **Insight Utama — Key Factors:**

            - **Fitur Paling Dominan:** *{top1_feat}* (penilaian kualitas material dan *finishing* bangunan) menempati posisi pertama sebagai variabel dengan korelasi linear tertinggi terhadap harga jual rumah (**r = {top1_corr:.2f}**), diikuti oleh luas area hunian di atas permukaan tanah (*{top2_feat}*, **r = {top2_corr:.2f}**) dan kapasitas mobil dalam garasi (*{top3_feat}*, **r = {top3_corr:.2f}**).
            - **Eksplorasi Interaktif:** Pengguna dapat menggeser *slider* di atas untuk mengeksplorasi hingga **20 fitur teratas** yang memiliki korelasi linear paling kuat terhadap variabel respons (*SalePrice*).
            """
        )
    else:
        st.info("Dataset belum dimuat atau kolom 'SalePrice' tidak tersedia.")

    st.divider()

    # ============================================================
    # 5. PROPERTY CHARACTERISTICS (ANALISIS HUBUNGAN)
    # ============================================================
    
    st.subheader("5. Analisis Karakteristik vs Harga Properti")

    if df is not None and "SalePrice" in df.columns:
        # --- MENGGANTI TEKS PENGANTAR YANG KAKU MENJADI LEBIH SANTAI ---
        st.markdown(
            """
            <div style="text-align: justify; color: inherit; line-height: 1.6;">
                Bagian ini akan membandingkan tiga ciri utama fisik sebuah rumah dengan <b>Harga Jualnya</b>. 
                Tujuannya adalah untuk melihat tren pasar yang wajar, serta mendeteksi jika ada rumah-rumah 
                dengan harga "aneh" (misalnya rumah sangat besar tetapi dijual sangat murah).
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Membagi layar menjadi 2 untuk menaruh grafik berdampingan
        col1, col2 = st.columns(2)

        # --------------------------------------------------------
        # GRAFIK 1: Kualitas Bawaan vs Harga (Boxplot)
        # --------------------------------------------------------
        with col1:
            # Menggambar Boxplot untuk melihat rentang harga di tiap skor kualitas [cite: 7, 8, 45]
            fig1 = px.box(
                df,
                x="OverallQual",
                y="SalePrice",
                color="OverallQual",
                color_discrete_sequence=px.colors.sequential.Blues_r,
                title="<b>Tren Harga Berdasarkan Kualitas Material</b>",
                labels={"OverallQual": "Skor Kualitas (1 - 10)", "SalePrice": "Harga Jual (USD)"}
            )
            fig1.update_layout(
                showlegend=False,
                height=420,
                xaxis=dict(tickmode="linear", tick0=1, dtick=1),
                yaxis=dict(tickprefix="$", tickformat=","),
                margin=dict(l=20, r=20, t=50, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            fig1.update_traces(marker=dict(size=4, opacity=0.6))
            st.plotly_chart(fig1, use_container_width=True)

        # --------------------------------------------------------
        # GRAFIK 2: Luas Rumah vs Harga (Titik-titik & Garis Tren)
        # --------------------------------------------------------
        with col2:
            # Menggambar sebaran titik (scatter) luas rumah terhadap harga
            fig2 = px.scatter(
                df,
                x="GrLivArea",
                y="SalePrice",
                opacity=0.6,
                title="<b>Luas Hunian vs Harga Jual</b>",
                labels={"GrLivArea": "Luas Area Hunian (sqft)", "SalePrice": "Harga Jual (USD)"},
                color_discrete_sequence=["#1f77b4"]
            )
            
            # Membuat garis lurus penunjuk arah tren pasar (Garis Regresi)
            valid_data = df[["GrLivArea", "SalePrice"]].dropna()
            m, c = np.polyfit(valid_data["GrLivArea"], valid_data["SalePrice"], 1)
            x_range = np.array([valid_data["GrLivArea"].min(), valid_data["GrLivArea"].max()])
            y_range = m * x_range + c

            # Menempelkan garis tren merah putus-putus ke dalam grafik
            fig2.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode="lines",
                    name="Garis Tren Pasar",
                    line=dict(color="#d62728", width=2, dash="dash")
                )
            )

            fig2.update_layout(
                height=420,
                yaxis=dict(tickprefix="$", tickformat=","),
                margin=dict(l=20, r=20, t=50, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            fig2.update_traces(marker=dict(size=6), selector=dict(type="scatter", mode="markers"))
            st.plotly_chart(fig2, use_container_width=True)

        # --------------------------------------------------------
        # GRAFIK 3: Usia Rumah vs Harga (Full Width ke Bawah)
        # --------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Menghitung umur rumah dengan mengurangkan tahun terjual dengan tahun dibangun
        df_age = df.copy()
        if "YearBuilt" in df_age.columns and "YrSold" in df_age.columns:
            df_age["HouseAge"] = df_age["YrSold"] - df_age["YearBuilt"]
            age_label = "Usia Rumah saat Dijual (Tahun)"
        else:
            df_age["HouseAge"] = df_age["YearBuilt"].max() - df_age["YearBuilt"]
            age_label = "Estimasi Usia Rumah (Tahun)"

        # Membuat grafik titik (scatter) untuk usia rumah
        fig3 = px.scatter(
            df_age,
            x="HouseAge",
            y="SalePrice",
            opacity=0.5,
            title="<b>Usia Rumah vs Harga Jual</b>",
            labels={"HouseAge": age_label, "SalePrice": "Harga Jual (USD)"},
            color_discrete_sequence=["#2ca02c"]
        )

        # Membuat garis lurus penunjuk arah tren pasar
        valid_age = df_age[["HouseAge", "SalePrice"]].dropna()
        m_age, c_age = np.polyfit(valid_age["HouseAge"], valid_age["SalePrice"], 1)
        x_age_range = np.array([valid_age["HouseAge"].min(), valid_age["HouseAge"].max()])
        y_age_range = m_age * x_age_range + c_age

        fig3.add_trace(
            go.Scatter(
                x=x_age_range,
                y=y_age_range,
                mode="lines",
                name="Garis Tren Pasar",
                line=dict(color="darkgreen", width=2)
            )
        )

        fig3.update_layout(
            height=420,
            yaxis=dict(tickprefix="$", tickformat=","),
            margin=dict(l=20, r=20, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        fig3.update_traces(marker=dict(size=6), selector=dict(type="scatter", mode="markers"))
        st.plotly_chart(fig3, use_container_width=True)

        # --------------------------------------------------------
        # KOTAK KESIMPULAN (Diubah bahasanya jadi sangat mudah dimengerti)
        # --------------------------------------------------------
        st.success(
            """
            💡 **Insight Utama & Implikasinya:**

            - **Kualitas Bangunan Terbukti Krusial:** Jika Anda melihat grafik pertama, harga rumah mulai melompat sangat drastis ketika skor kualitasnya menyentuh angka 8 ke atas (kelas Premium). Kotak yang semakin panjang ke atas juga menunjukkan bahwa harga rumah mewah sangat bervariasi.
            - **Anomali Luas Rumah (Pencilan/Outliers):** Pada grafik kedua, secara umum semakin luas rumah maka harganya semakin mahal (mengikuti garis tren merah). Namun, perhatikan ada beberapa titik di pojok kanan bawah: itu adalah rumah yang luar biasa besar (> 4.000 sqft) tetapi dijual sangat murah. Ini adalah data anomali yang bisa mengacaukan tebakan model prediksi.
            - **Depresiasi karena Usia:** Grafik ketiga membuktikan bahwa semakin tua usia rumah saat dijual, secara rata-rata harganya akan semakin turun (garis hijau menukik ke bawah).
            - **Kenapa kita butuh Ridge & LASSO?** Banyaknya rumah dengan harga yang "keluar jalur" (anomali luas rumah) serta harga rumah mewah yang variasi angkanya sangat liar, bisa membuat model regresi biasa (OLS) menjadi kebingungan. Regresi **Ridge dan LASSO** dirancang khusus untuk mengendalikan data-data liar ini agar hasil prediksinya tetap stabil dan akurat.
            """
        )
    else:
        st.info("Dataset belum dimuat atau kolom 'SalePrice' tidak tersedia.")

    st.divider()

    # ============================================================
    # 6. LOCATION ANALYSIS (NEIGHBORHOOD INSIGHT)
    # ============================================================
    st.subheader("6. Location & Neighborhood Impact Analysis")

    # Memastikan datanya tidak kosong dan memiliki kolom Harga (SalePrice) & Lokasi (Neighborhood)
    if df is not None and "SalePrice" in df.columns and "Neighborhood" in df.columns:
        
        # 1. MENAMPILKAN TEKS PENGANTAR
        # Menampilkan teks ke layar pengguna bahwa lokasi sangat menentukan harga rumah
        st.markdown(
            """
            <div style="text-align: justify; color: inherit; line-height: 1.6;">
                Lokasi merupakan variabel kategorikal dengan pengaruh spasial yang signifikan terhadap nilai properti. 
                Analisis ini mengevaluasi variasi harga rumah berdasarkan wilayah pemukiman (<b>Neighborhood</b>) 
                untuk membedakan kawasan premium dan kawasan terjangkau.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True) # Memberi jarak kosong (enter)

        # 2. PENGELOMPOKKAN DATA (AGREGASI)
        # Komputer menyortir rumah-rumah berdasarkan lokasi, lalu mencari nilai tengah (median) harganya [cite: 540]
        neighborhood_stats = (
            df.groupby("Neighborhood")["SalePrice"]
            .agg(["median", "mean", "count"])
            .reset_index()
            .sort_values(by="median", ascending=False) # Diurutkan dari harga yang paling mahal ke termurah [cite: 540]
        )

        # Membagi layar aplikasi menjadi dua bagian: Kiri (lebar 1.8) dan Kanan (lebar 1)
        col1, col2 = st.columns([1.75, 1])

        # --------------------------------------------------------
        # BAGIAN KIRI: Membuat Grafik 10 Kawasan Termahal
        # --------------------------------------------------------
        with col1:
            # Mengambil 10 data teratas dari hasil pengelompokkan tadi [cite: 543]
            top_10_neigh = neighborhood_stats.head(10).sort_values(by="median", ascending=True)

            # Menggambar grafik batang (bar chart) mendatar menggunakan Plotly [cite: 542]
            fig = px.bar(
                top_10_neigh,
                x="median", # Sumbu mendatar adalah Harga Median
                y="Neighborhood", # Sumbu vertikal adalah Nama Lokasi
                orientation="h",
                title="<b>Top 10 Most Expensive Neighborhoods (Median Price)</b>",
                text=top_10_neigh["median"].apply(lambda x: f"${x:,.0f}"), # Menempelkan angka harga di ujung batang grafik [cite: 545]
                color="median",
                color_continuous_scale="Purples", # Memberi warna ungu yang elegan [cite: 544]
                labels={"median": "Median Harga Jual (USD)", "Neighborhood": "Lokasi Pemukiman"}
            )

            # Mempercantik tampilan teks saat mouse diarahkan ke grafik (hover)
            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Median Harga: <b>$%{x:,.0f}</b><extra></extra>"
            )

            # Mengatur margin dan latar belakang agar grafik terlihat rapi
            fig.update_layout(
                height=420,
                xaxis=dict(title="Median Sale Price (USD)", tickprefix="$", tickformat=","),
                yaxis=dict(title=""),
                coloraxis_showscale=False,
                margin=dict(l=20, r=40, t=50, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig, use_container_width=True) # Menampilkan grafiknya ke layar Streamlit

        # --------------------------------------------------------
        # BAGIAN KANAN: Menampilkan Teks Juara dan Juru Kunci
        # --------------------------------------------------------
        with col2:
            st.markdown("#### 📍 Tertinggi vs Termurah")

            # Mencari tahu siapa yang termahal (urutan pertama/0) dan termurah (urutan terakhir/-1)
            most_expensive = neighborhood_stats.iloc[0]
            least_expensive = neighborhood_stats.iloc[-1]
            
            # Menghitung perbandingan harganya (Harga termahal dibagi harga termurah)
            price_ratio = most_expensive["median"] / least_expensive["median"]

            # Menampilkan metrik untuk juara 1 (Kawasan Termahal)
            # Label bahasa Inggrisnya dihapus agar lebih bersih
            st.metric(
                label="🏆 Kawasan Termahal",
                value=f"{most_expensive['Neighborhood']}",
                delta=f"${most_expensive['median']:,.0f}"
            )

            # Menampilkan metrik untuk posisi terbawah (Kawasan Terjangkau)
            st.metric(
                label="💰 Kawasan Terjangkau",
                value=f"{least_expensive['Neighborhood']}",
                delta=f"${least_expensive['median']:,.0f}"
            )

            # Menampilkan kotak berisi kesimpulan matematis (berapa kali lipat beda harganya)
            # Teks dibuat rata kiri-kanan (justify) dan bahasanya disederhanakan
            st.markdown(
                f"""
                <div style="background-color: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 8px; margin-top: 5px; border-left: 4px solid #9b59b6; text-align: justify;">
                    <b>Perbandingan Harga:</b> Sebagai gambaran, harga wajar rumah di kawasan elit <b>{most_expensive['Neighborhood']}</b> bisa mencapai <b>{price_ratio:.1f} kali lipat</b> lebih mahal dibandingkan jika Anda membeli rumah di kawasan yang paling terjangkau (<b>{least_expensive['Neighborhood']}</b>).
                """,
                unsafe_allow_html=True
            )

        # Memberikan baris spasi
        st.write("") 
        
        # --------------------------------------------------------
        # BAGIAN BAWAH: Kotak Kesimpulan Model Regresi (LASSO & Ridge)
        # --------------------------------------------------------
        # Mencetak "Nasihat" kenapa lokasi memengaruhi model prediksi Machine Learning kita [cite: 551, 552]
        st.success(
            f"""
            💡 **Insight Utama & Implikasi terhadap Model Regresi:**

            - **Disparitas Harga Antar-Wilayah:** Lokasi pemukiman memiliki pengaruh signifikan terhadap nilai properti. Pemukiman seperti **{most_expensive['Neighborhood']}** dan **NoRidge** secara konsisten mendominasi segmen harga tinggi, sedangkan kawasan seperti **{least_expensive['Neighborhood']}** berada di segmen terjangkau.
            - **Pembengkakan Dimensi Data:** Mengubah variabel kualitatif `Neighborhood` menjadi bentuk angka (*One-Hot Encoding*) akan menambahkan 25 kolom *dummy* baru ke dalam matriks prediktor ($X$).
            - **Urgensi Regularisasi (Ridge & LASSO):** Bertambahnya puluhan variabel *dummy* ini menyebabkan matriks data didominasi angka nol (*sparse matrix*) dan memicu masalah multikolinearitas. Metode **LASSO** sangat efektif untuk mengeliminasi *dummy* lokasi yang tidak signifikan dengan mengubah koefisiennya menjadi **tepat nol**, sedangkan **Ridge** berfungsi menstabilkan bobot koefisien antar-lokasi yang saling berkorelasi.
            """
        )
    else:
        # Pesan error jika data rumah gagal dimuat
        st.info("Dataset belum dimuat atau kolom 'SalePrice' / 'Neighborhood' tidak tersedia.")

    st.divider() # Membuat garis pembatas panjang

    # ============================================================
    # 7. BEHIND THE PREDICTION MODEL
    # ============================================================
    st.subheader("7. Behind The Prediction Model (Feature Expansion & Structure)")

    if df is not None:
        # Nilai Disesuaikan Presisi dengan Google Colab Anda
        n_samples = df.shape[0]
        n_orig_predictors = 79       # 80 kolom mentah dikurangi Id & SalePrice
        n_encoded_predictors = 259  # Total Fitur X di Colab setelah One-Hot Encoding
        added_dummy_cols = n_encoded_predictors - n_orig_predictors  # +180 kolom dummy baru
        num_cat_cols = 43           # 43 Variabel Kategorik di Colab

        st.markdown(
            """
            <div style="text-align: justify; color: inherit; line-height: 1.6;">
                Tahap pemodelan regresi tidak langsung menggunakan data mentah. Kolom identifikasi (<b>Id</b>) di-drop, 
                dan seluruh 43 variabel kualitatif ditransformasikan menggunakan <i>One-Hot Encoding</i> 
                sebelum pembagian data (<i>train-test split</i>). Proses ini memicu lonjakan dimensi matriks fitur prediktor (<b>X</b>) secara signifikan.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # --------------------------------------------------------
        # Metrics Display (4 Kolom)
        # --------------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📊 Fitur Mentah (Awal)",
                value=f"{n_orig_predictors} Fitur",
                help="79 Kolom Prediktor Mentah (setelah Id dan SalePrice dipisah)"
            )

        with col2:
            st.metric(
                label="⚡ Matriks Fitur (X)",
                value=f"{n_encoded_predictors} Fitur",
                delta=f"+{added_dummy_cols} kolom dummy",
                delta_color="normal",
                help="259 Fitur Prediktor X siap di-split dan di-fit ke model (Sama dengan Colab)"
            )

        with col3:
            st.metric(
                label="👥 Jumlah Observasi (n)",
                value=f"{n_samples:,} Sampel",
                help="Jumlah baris/observasi rumah dalam dataset"
            )

        with col4:
            st.metric(
                label="🎯 Metode Pemodelan",
                value="Ridge & LASSO",
                help="Regresi Terregularisasi Penalti L1 & L2"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --------------------------------------------------------
        # Visual Bar Bertumpuk (Fitur Asli vs Expansion)
        # --------------------------------------------------------
        fig = go.Figure()

        # Bar Fitur Asli
        fig.add_trace(go.Bar(
            y=["Dimensi Matriks X"],
            x=[n_orig_predictors],
            name="Fitur Prediktor Mentah",
            orientation="h",
            marker=dict(color="#2b5c8f"),
            text=[f"<b>{n_orig_predictors} Fitur Mentah</b>"],
            textposition="inside",
            hovertemplate="Fitur Mentah: %{x}<extra></extra>"
        ))

        # Bar Hasil Encoding
        fig.add_trace(go.Bar(
            y=["Dimensi Matriks X"],
            x=[added_dummy_cols],
            name="Hasil One-Hot Encoding",
            orientation="h",
            marker=dict(color="#e74c3c"),
            text=[f"<b>+{added_dummy_cols} Dummy</b> (Total: {n_encoded_predictors})"],
            textposition="inside",
            hovertemplate="Tambahan Dummy: %{x}<extra></extra>"
        ))

        fig.update_layout(
            barmode="stack",
            title="<b>Visualisasi Pembengkakan Dimensi Fitur Prediktor (Feature Expansion)</b>",
            height=200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(title="Jumlah Kolom Prediktor (p)", range=[0, n_encoded_predictors + 30]),
            yaxis=dict(showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------------
        # Academic Justification Callout Box
        # --------------------------------------------------------
        st.info(
            rf"""
            💡 **Urgensi Teoretis Pemodelan:**

            - **Pembengkakan Dimensi (p vs n):** Transformasi *One-Hot Encoding* pada **{num_cat_cols} variabel kualitatif** menambahkan **{added_dummy_cols} kolom *dummy* baru**. Hal ini menyebabkan dimensi matriks prediktor **X** membengkak dari **{n_orig_predictors} menjadi {n_encoded_predictors} fitur** (total 260 kolom jika memperhitungkan target *SalePrice*). Lonjakan dimensi ini meningkatkan risiko terjadinya *overfitting* pada regresi Ordinary Least Squares (OLS) biasa.
            - **Terbentuknya Matriks Jarang (*Sparse Matrix*):** Sebagian besar kolom *dummy* baru didominasi oleh nilai `0`, menciptakan struktur matriks yang rentan terhadap multikolinearitas dan variansi estimasi koefisien yang tidak stabil.
            - **Peran Regularisasi:**
              - **LASSO (L1):** Bertindak sebagai *feature selector* otomatis dengan memaksa koefisien *dummy* yang tidak signifikan menjadi **tepat nol** ($\beta_j = 0$), sehingga menyederhanakan model.
              - **Ridge (L2):** Memperkecil (*shrinkage*) nilai koefisien tanpa mengnolkan, guna menstabilkan estimasi pada fitur-fitur yang saling berkorelasi.
            """
        )
    else:
        st.info("Dataset belum dimuat.")