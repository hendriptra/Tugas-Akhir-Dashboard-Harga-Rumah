import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def apply_eda_page_style():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .eda-hero {
            background: linear-gradient(120deg, #102a43 0%, #1d4e89 100%);
            padding: 28px 32px;
            border-radius: 18px;
            margin-bottom: 28px;
            color: white;
        }

        .eda-tag {
            display: inline-block;
            padding: 5px 11px;
            margin-bottom: 12px;
            border: 1px solid rgba(255,255,255,.35);
            border-radius: 999px;
            color: #e7f0fa;
            font-size: 12px;
            letter-spacing: .4px;
        }

        .eda-hero h1 {
            color: white;
            font-size: 30px;
            margin: 0 0 8px 0;
            font-weight: 700;
        }

        .eda-hero p {
            color: #e6eef7;
            font-size: 15px;
            line-height: 1.6;
            margin: 0;
        }

        .eda-section-title {
            font-size: 21px;
            font-weight: 700;
            color: inherit;
            margin: 10px 0 4px 0;
        }

        .eda-section-caption {
            color: #718096;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 18px;
        }

        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.06);
            border: 1px solid rgba(128, 128, 128, 0.16);
            padding: 15px 17px;
            border-radius: 14px;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 13px;
        }

        div[data-testid="stMetricValue"] {
            font-size: 23px;
            font-weight: 700;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        div[data-testid="stPlotlyChart"] {
            border: 1px solid rgba(128, 128, 128, 0.12);
            border-radius: 14px;
            padding: 8px;
        }

        hr {
            margin: 28px 0;
            border-color: rgba(128, 128, 128, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def eda_section(title, caption=None):
    st.markdown(
        f'<div class="eda-section-title">{title}</div>',
        unsafe_allow_html=True
    )
    if caption:
        st.markdown(
            f'<div class="eda-section-caption">{caption}</div>',
            unsafe_allow_html=True
        )


def render_eda_page(df):

    apply_eda_page_style()

    st.markdown(
        """
        <div class="eda-hero">
            <div class="eda-tag">EXPLORATORY DATA ANALYSIS</div>
            <h1>House Sales Insight</h1>
            <p>
                Eksplorasi pola harga jual rumah dan karakteristik yang
                berkaitan dengan nilai properti.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if df is None or df.empty or "SalePrice" not in df.columns:
        st.warning("Dataset belum dimuat atau kolom 'SalePrice' tidak tersedia.")
        return

    # ============================================================
    # 1. HOUSE PRICE OVERVIEW
    # ============================================================

    eda_section(
        "1. House Price Overview",
        "Ringkasan harga jual rumah berdasarkan jumlah observasi, "
        "rata-rata, nilai minimum, maksimum, dan median."
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("🏠 Total Properties", f"{df.shape[0]:,}")
    col2.metric("💰 Average Price", f"${df['SalePrice'].mean():,.0f}")
    col3.metric("📉 Lowest Price", f"${df['SalePrice'].min():,.0f}")
    col4.metric("📈 Highest Price", f"${df['SalePrice'].max():,.0f}")
    col5.metric("🏡 Median Price", f"${df['SalePrice'].median():,.0f}")

    st.divider()

    # ============================================================
    # 2. PRICE DISTRIBUTION
    # ============================================================

    eda_section(
        "2. Price Distribution",
        "Melihat sebaran harga rumah dan rentang harga yang paling banyak ditemui."
    )

    col1, col2 = st.columns(2)

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

        ax.set_title("Distribution of Sale Prices", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sale Price ($)")
        ax.set_ylabel("Number of Houses")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        price_segment = pd.cut(
            df["SalePrice"],
            bins=[0, 100000, 200000, 300000, np.inf],
            labels=["< $100K", "$100K - $200K", "$200K - $300K", "> $300K"]
        )

        segment = (
            price_segment.value_counts()
            .sort_index()
            .reset_index()
        )
        segment.columns = ["Price Range", "Number of Houses"]
        segment["Percentage"] = segment["Number of Houses"] / len(df) * 100

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#AFC7E3", "#4C78A8", "#2A9D8F", "#F28E2B"]

        bars = ax.bar(
            segment["Price Range"].astype(str),
            segment["Percentage"],
            color=colors,
            width=0.65
        )

        ax.set_title("House Price Segments", fontsize=13, fontweight="bold")
        ax.set_xlabel("Price Range")
        ax.set_ylabel("Percentage of Houses (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, value in zip(bars, segment["Percentage"]):
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    median_price = df["SalePrice"].median()
    mean_price = df["SalePrice"].mean()
    skewness = df["SalePrice"].skew()

    largest_segment = segment.loc[segment["Number of Houses"].idxmax()]
    premium_houses = (df["SalePrice"] > 300000).sum()
    premium_percentage = premium_houses / len(df) * 100

    if skewness > 0.5:
        distribution_text = (
            "Harga rumah cenderung terkonsentrasi pada harga yang lebih rendah, "
            "dengan sebagian kecil rumah memiliki harga jauh lebih tinggi."
        )
    else:
        distribution_text = "Harga rumah memiliki penyebaran yang relatif seimbang."

    st.info(
        f"""
        **💡 Price Insight**

        - **Median dan rata-rata:** Median harga rumah sebesar **\\${median_price:,.0f}**, sedangkan rata-ratanya **\\${mean_price:,.1f}**.
        - **Rentang dominan:** Kelompok harga **{largest_segment['Price Range'].replace('$', '\\$')}** mencakup **{largest_segment['Percentage']:.1f}%** dari total sampel.
        - **Kategori premium:** Rumah dengan harga di atas **\\$300K** mencakup **{premium_percentage:.1f}%** dari data.
        - **Bentuk distribusi:** {distribution_text}
        """
    )

    st.divider()

    # ============================================================
    # 3. PREMIUM PROPERTY INSIGHT
    # ============================================================

    eda_section(
        "3. Premium Property Insight",
        "Membandingkan karakteristik rumah premium dengan rumah pada segmen lainnya."
    )

    premium_threshold = df["SalePrice"].quantile(0.9)
    premium_df = df[df["SalePrice"] >= premium_threshold]
    normal_df = df[df["SalePrice"] < premium_threshold]

    st.markdown(
        f"""
        <div class="eda-section-caption">
            Properti premium didefinisikan sebagai 10% rumah dengan harga jual tertinggi,
            yaitu rumah dengan harga setidaknya <b>&#36;{premium_threshold:,.0f}</b>.
            Perbandingan ini menunjukkan perbedaan karakteristik antara segmen premium
            dan segmen lainnya.
        </div>
        """,
        unsafe_allow_html=True
    )

    comp_df = pd.DataFrame({
        "Kategori": ["Rumah Biasa (90%)", "Rumah Premium (10%)"],
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
    })

    st.markdown("**Tabel Perbandingan Karakteristik Rata-rata**")

    display_df = comp_df.copy()
    display_df["Kualitas Bangunan (1-10)"] = display_df["Kualitas Bangunan (1-10)"].map("{:.2f} / 10".format)
    display_df["Kapasitas Garasi (Mobil)"] = display_df["Kapasitas Garasi (Mobil)"].map("{:.2f} mobil".format)
    display_df["Luas Area Hunian (sqft)"] = display_df["Luas Area Hunian (sqft)"].map("{:,.0f} sqft".format)
    display_df["Rata-rata Harga Jual"] = display_df["Rata-rata Harga Jual"].map("${:,.0f}".format)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.set_theme(style="whitegrid")

    sns.barplot(
        data=comp_df, x="Kategori", y="Kualitas Bangunan (1-10)",
        ax=axes[0], hue="Kategori", palette=["#4C72B0", "#DD8452"], legend=False
    )
    axes[0].set_title("Kualitas Bangunan", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Skala (1-10)")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 10)

    sns.barplot(
        data=comp_df, x="Kategori", y="Kapasitas Garasi (Mobil)",
        ax=axes[1], hue="Kategori", palette=["#4C72B0", "#DD8452"], legend=False
    )
    axes[1].set_title("Kapasitas Garasi", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Jumlah Mobil")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 4)

    sns.barplot(
        data=comp_df, x="Kategori", y="Luas Area Hunian (sqft)",
        ax=axes[2], hue="Kategori", palette=["#4C72B0", "#DD8452"], legend=False
    )
    axes[2].set_title("Luas Area Hunian", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Luas (sqft)")
    axes[2].set_xlabel("")

    for ax in axes:
        ax.tick_params(axis="x", rotation=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    diff_price = comp_df.loc[1, "Rata-rata Harga Jual"] / comp_df.loc[0, "Rata-rata Harga Jual"]
    diff_area = (
        comp_df.loc[1, "Luas Area Hunian (sqft)"]
        / comp_df.loc[0, "Luas Area Hunian (sqft)"] - 1
    ) * 100

    st.success(
        f"""
        **💡 Insight Utama — Properti Premium**

        - **Perbedaan harga:** Rata-rata harga rumah premium mencapai **\\${comp_df.loc[1, 'Rata-rata Harga Jual']:,.0f}**, sekitar **{diff_price:.1f} kali** rata-rata rumah biasa (**\\${comp_df.loc[0, 'Rata-rata Harga Jual']:,.0f}**).
        - **Kualitas bangunan:** Rata-rata *OverallQual* rumah premium adalah **{comp_df.loc[1, 'Kualitas Bangunan (1-10)']:.2f}/10**, dibandingkan **{comp_df.loc[0, 'Kualitas Bangunan (1-10)']:.2f}/10** pada rumah biasa.
        - **Luas hunian:** *GrLivArea* rumah premium rata-rata **{diff_area:.1f}% lebih luas** daripada rumah biasa.
        """
    )

    st.divider()

    # ============================================================
    # 4. KEY FACTORS
    # ============================================================

    eda_section(
        "4. Key Factors Affecting Property Value",
        "Menampilkan fitur numerik dengan korelasi Pearson terkuat terhadap harga jual."
    )

    numeric_df = df.select_dtypes(include=np.number)
    col_select, _ = st.columns([1, 2])

    with col_select:
        top_n = st.slider(
            "Jumlah fitur teratas:",
            min_value=5,
            max_value=20,
            value=10,
            step=5
        )

    corr_series = (
        numeric_df.corr()["SalePrice"]
        .drop("SalePrice")
        .sort_values(key=abs, ascending=False)
        .head(top_n)
        .sort_values(ascending=True)
    )

    df_corr = pd.DataFrame({
        "Feature": corr_series.index,
        "Correlation": corr_series.values
    })

    fig = px.bar(
        df_corr,
        x="Correlation",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Features Correlated with SalePrice",
        text=df_corr["Correlation"].apply(lambda x: f"{x:.2f}"),
        color="Correlation",
        color_continuous_scale="Blues",
        labels={"Correlation": "Correlation Coefficient (r)", "Feature": "Fitur"}
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Korelasi: <b>%{x:.3f}</b><extra></extra>"
    )
    fig.update_layout(
        height=max(400, top_n * 35),
        xaxis=dict(range=[0, 1.0], title="Correlation Coefficient (r)"),
        yaxis=dict(title=""),
        coloraxis_showscale=False,
        margin=dict(l=20, r=40, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    top_features = df_corr.iloc[::-1].head(3)

    st.info(
        f"""
        **💡 Insight Utama — Key Factors**

        - Fitur dengan korelasi tertinggi adalah **{top_features.iloc[0]['Feature']}** (r = **{top_features.iloc[0]['Correlation']:.2f}**).
        - Berikutnya adalah **{top_features.iloc[1]['Feature']}** (r = **{top_features.iloc[1]['Correlation']:.2f}**) dan **{top_features.iloc[2]['Feature']}** (r = **{top_features.iloc[2]['Correlation']:.2f}**).
        - Gunakan slider untuk mengeksplorasi hingga **20 fitur** dengan korelasi absolut terkuat terhadap *SalePrice*. Korelasi menunjukkan hubungan linear, bukan sebab-akibat.
        """
    )

    st.divider()

    # ============================================================
    # 5. PROPERTY CHARACTERISTICS
    # ============================================================

    eda_section(
        "5. Property Characteristics",
        "Melihat hubungan kualitas, luas hunian, dan usia rumah dengan harga jual."
    )

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.box(
            df,
            x="OverallQual",
            y="SalePrice",
            color="OverallQual",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            title="Harga Berdasarkan Kualitas Bangunan",
            labels={"OverallQual": "Skor Kualitas (1–10)", "SalePrice": "Harga Jual (USD)"}
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

    with col2:
        fig2 = px.scatter(
            df,
            x="GrLivArea",
            y="SalePrice",
            opacity=0.6,
            title="Luas Hunian vs Harga Jual",
            labels={"GrLivArea": "Luas Area Hunian (sqft)", "SalePrice": "Harga Jual (USD)"},
            color_discrete_sequence=["#1f77b4"]
        )

        valid_data = df[["GrLivArea", "SalePrice"]].dropna()
        if len(valid_data) >= 2 and valid_data["GrLivArea"].nunique() > 1:
            m, c = np.polyfit(valid_data["GrLivArea"], valid_data["SalePrice"], 1)
            x_range = np.array([valid_data["GrLivArea"].min(), valid_data["GrLivArea"].max()])
            fig2.add_trace(
                go.Scatter(
                    x=x_range,
                    y=m * x_range + c,
                    mode="lines",
                    name="Garis Tren",
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

    if "YearBuilt" in df.columns:
        df_age = df.copy()

        if "YrSold" in df_age.columns:
            df_age["HouseAge"] = df_age["YrSold"] - df_age["YearBuilt"]
            age_label = "Usia Rumah saat Dijual (Tahun)"
        else:
            df_age["HouseAge"] = df_age["YearBuilt"].max() - df_age["YearBuilt"]
            age_label = "Estimasi Usia Rumah (Tahun)"

        fig3 = px.scatter(
            df_age,
            x="HouseAge",
            y="SalePrice",
            opacity=0.5,
            title="Usia Rumah vs Harga Jual",
            labels={"HouseAge": age_label, "SalePrice": "Harga Jual (USD)"},
            color_discrete_sequence=["#2ca02c"]
        )

        valid_age = df_age[["HouseAge", "SalePrice"]].dropna()
        if len(valid_age) >= 2 and valid_age["HouseAge"].nunique() > 1:
            m_age, c_age = np.polyfit(valid_age["HouseAge"], valid_age["SalePrice"], 1)
            x_age_range = np.array([valid_age["HouseAge"].min(), valid_age["HouseAge"].max()])
            fig3.add_trace(
                go.Scatter(
                    x=x_age_range,
                    y=m_age * x_age_range + c_age,
                    mode="lines",
                    name="Garis Tren",
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

    st.info(
        """
        **💡 Insight Utama**

        - Kualitas bangunan dan luas hunian cenderung memiliki hubungan positif dengan harga jual.
        - Beberapa rumah berukuran besar memiliki harga relatif rendah. Titik-titik tersebut perlu ditinjau sebagai kemungkinan pencilan, tetapi tidak otomatis berarti datanya salah.
        - Hubungan usia rumah dengan harga dapat dipengaruhi oleh faktor lain, seperti lokasi dan kualitas bangunan.
        - Ridge dan LASSO membantu mengendalikan besarnya koefisien dan risiko *overfitting* melalui regularisasi. Keduanya bukan metode khusus untuk mengatasi pencilan.
        """
    )

    st.divider()

    # ============================================================
    # 6. LOCATION & NEIGHBORHOOD
    # ============================================================

    eda_section(
        "6. Location & Neighborhood Impact",
        "Membandingkan median harga rumah berdasarkan kawasan pemukiman."
    )

    if "Neighborhood" in df.columns:
        neighborhood_stats = (
            df.groupby("Neighborhood")["SalePrice"]
            .agg(["median", "mean", "count"])
            .reset_index()
            .sort_values(by="median", ascending=False)
        )

        col1, col2 = st.columns([1.3, 1])

        with col1:
            top_10_neigh = neighborhood_stats.head(10).sort_values(
                by="median", ascending=True
            )

            fig = px.bar(
                top_10_neigh,
                x="median",
                y="Neighborhood",
                orientation="h",
                title="Top 10 Neighborhoods by Median Price",
                text=top_10_neigh["median"].apply(lambda x: f"${x:,.0f}"),
                color="median",
                color_continuous_scale="Purples",
                labels={"median": "Median Harga Jual (USD)", "Neighborhood": "Kawasan"}
            )
            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Median Harga: <b>$%{x:,.0f}</b><extra></extra>"
            )
            fig.update_layout(
                height=420,
                xaxis=dict(title="Median Sale Price (USD)", tickprefix="$", tickformat=","),
                yaxis=dict(title=""),
                coloraxis_showscale=False,
                margin=dict(l=20, r=40, t=50, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Kawasan Termahal vs Terjangkau**")

            most_expensive = neighborhood_stats.iloc[0]
            least_expensive = neighborhood_stats.iloc[-1]
            price_ratio = (
                most_expensive["median"] / least_expensive["median"]
                if least_expensive["median"] != 0 else np.nan
            )

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    "🏆 Termahal",
                    most_expensive["Neighborhood"],
                    delta=f"${most_expensive['median']:,.0f}"
                )

            with metric_col2:
                st.metric(
                    "💰 Terjangkau",
                    least_expensive["Neighborhood"],
                    delta=f"${least_expensive['median']:,.0f}"
                )

            st.markdown(
                f"""
                <div style="
                    background: rgba(128,128,128,.07);
                    padding: 14px;
                    border-radius: 12px;
                    margin-top: 10px;
                    border-left: 4px solid #9b59b6;
                    line-height: 1.6;
                ">
                    Median harga di <b>{most_expensive['Neighborhood']}</b>
                    sekitar <b>{price_ratio:.1f} kali</b> median harga di
                    <b>{least_expensive['Neighborhood']}</b>.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("**Peta Kawasan Perumahan Kota Ames, Iowa, Amerika Serikat**")
        _, map_col, _ = st.columns([0.5, 4, 1])

        with map_col:
            st.image("neighborhood.png", use_container_width=True)

        st.info(
            f"""
            **💡 Insight Lokasi**

            - Median harga berbeda antar-kawasan; **{most_expensive['Neighborhood']}** memiliki median tertinggi pada dataset, sedangkan **{least_expensive['Neighborhood']}** terendah.
            - *Neighborhood* merupakan fitur kategorikal. Setelah *one-hot encoding*, kategori lokasi direpresentasikan sebagai sejumlah kolom dummy.
            - Fitur lokasi dapat saling berkaitan. Ridge membantu mengecilkan koefisien, sedangkan LASSO dapat membuat sebagian koefisien menjadi nol.
            """
        )
    else:
        st.info("Kolom 'Neighborhood' tidak tersedia pada dataset.")

    st.divider()

    # ============================================================
    # 7. BEHIND THE PREDICTION MODEL
    # ============================================================

    eda_section(
        "7. Behind the Prediction Model",
        "Melihat perubahan jumlah fitur dari data mentah hingga matriks prediktor."
    )

    n_samples = df.shape[0]
    n_orig_predictors = 79
    n_encoded_predictors = 259
    added_dummy_cols = n_encoded_predictors - n_orig_predictors
    num_cat_cols = 43

    st.markdown(
        """
        <div class="eda-section-caption">
            Kolom identifikasi (<b>Id</b>) dipisahkan dari prediktor.
            Variabel kategorikal kemudian diubah menggunakan <i>One-Hot Encoding</i>,
            sehingga jumlah kolom pada matriks prediktor bertambah.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📊 Fitur Mentah", f"{n_orig_predictors} fitur")
    col2.metric("⚡ Fitur Setelah Encoding", f"{n_encoded_predictors} fitur", delta=f"+{added_dummy_cols} kolom")
    col3.metric("👥 Observasi", f"{n_samples:,} sampel")
    col4.metric("🎯 Metode", "Ridge & LASSO")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=["Dimensi Matriks X"],
            x=[n_orig_predictors],
            name="Fitur Prediktor Mentah",
            orientation="h",
            marker=dict(color="#2b5c8f"),
            text=[f"{n_orig_predictors} Fitur Mentah"],
            textposition="inside",
            hovertemplate="Fitur Mentah: %{x}<extra></extra>"
        )
    )

    fig.add_trace(
        go.Bar(
            y=["Dimensi Matriks X"],
            x=[added_dummy_cols],
            name="Hasil One-Hot Encoding",
            orientation="h",
            marker=dict(color="#e74c3c"),
            text=[f"+{added_dummy_cols} Dummy"],
            textposition="inside",
            hovertemplate="Tambahan Dummy: %{x}<extra></extra>"
        )
    )

    fig.update_layout(
        barmode="stack",
        title="Perubahan Dimensi Fitur Prediktor",
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

    st.info(
        f"""
        **💡 Urgensi Pemodelan**

        - *One-Hot Encoding* pada **{num_cat_cols} variabel kategorikal** menghasilkan tambahan **{added_dummy_cols} kolom**, sehingga jumlah prediktor bertambah dari **{n_orig_predictors} menjadi {n_encoded_predictors} fitur**.
        - Penambahan fitur dapat meningkatkan kompleksitas model dan risiko *overfitting*, terutama ketika prediktor saling berkorelasi.
        - **LASSO (L1)** dapat melakukan seleksi fitur dengan membuat sebagian koefisien tepat nol.
        - **Ridge (L2)** mengecilkan koefisien tanpa menghilangkannya, sehingga membantu menstabilkan estimasi.
        """
    )