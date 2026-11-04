import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# HELPER: FORMAT DAN VISUALISASI
# ============================================================

MODEL_COLORS = {
    "Ridge": "#2F6B5F",
    "LASSO": "#D97745",
    "Elastic Net": "#8BAFA5",
}


def style_chart(ax, title, ylabel):
    """Menyeragamkan tampilan grafik agar bersih dan konsisten."""
    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        color="#24443D",
        pad=14,
    )
    ax.set_ylabel(ylabel, fontsize=10, color="#52645F")
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
        zorder=0,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.2)
    ax.spines["bottom"].set_alpha(0.2)

    ax.tick_params(axis="both", labelsize=9, colors="#52645F")


def format_cv_table(data):
    """Format tabel hasil cross-validation."""
    result = data.copy()

    result["CV R²"] = result["CV R²"].map(lambda x: f"{x:.4f}")
    result["CV MAE"] = result["CV MAE"].map(lambda x: f"${x:,.2f}")
    result["CV MSE"] = result["CV MSE"].map(
        lambda x: f"{x / 1e9:.3f} × 10⁹"
    )
    result["CV RMSE"] = result["CV RMSE"].map(lambda x: f"${x:,.2f}")

    return result


def format_final_table(data):
    """Format tabel evaluasi data testing."""
    result = data.copy()

    result["R² Test"] = result["R² Test"].map(
        lambda x: f"{x:.4f}" if pd.notnull(x) else "-"
    )
    result["MAE Test"] = result["MAE Test"].map(
        lambda x: f"${x:,.2f}" if pd.notnull(x) else "-"
    )
    result["MSE Test"] = result["MSE Test"].map(
        lambda x: f"{x / 1e9:.3f} × 10⁹" if pd.notnull(x) else "-"
    )
    result["RMSE Test"] = result["RMSE Test"].map(
        lambda x: f"${x:,.2f}" if pd.notnull(x) else "-"
    )

    return result


def render_bar_chart(
    data,
    x_column,
    y_column,
    title,
    ylabel,
    value_scale=1,
    value_format="{:.3f}",
    x_rotation=0,
    y_min=None,
    y_max_factor=1.18,
):
    """Membuat grafik batang dengan gaya yang seragam."""
    labels = data[x_column].astype(str).tolist()
    values = data[y_column].astype(float).to_numpy() / value_scale
    colors = [MODEL_COLORS.get(label.split("\n")[0], "#8A9692") for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar(
        labels,
        values,
        width=0.55,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    style_chart(ax, title, ylabel)

    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    max_value = max(abs(values)) if len(values) else 1
    if max_value > 0:
        ax.set_ylim(
            bottom=y_min if y_min is not None else 0,
            top=max(values) + max_value * (y_max_factor - 1),
        )

    for bar, value in zip(bars, values):
        offset = max_value * 0.025
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            value_format.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#24443D",
        )

    ax.tick_params(axis="x", labelrotation=x_rotation)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def make_performance_table(ridge_row, lasso_row, elastic_row):
    """Menyusun tabel CV dari baris Ridge, LASSO, dan Elastic Net."""
    rows = []

    for model_name, row in [
        ("Ridge", ridge_row),
        ("LASSO", lasso_row),
        ("Elastic Net", elastic_row),
    ]:
        rows.append(
            {
                "Model": model_name,
                "CV R²": row["CV R²"],
                "CV MAE": row["CV MAE"],
                "CV MSE": row["CV MSE"],
                "CV RMSE": row["CV RMSE"],
                "Fitur Dibuang": int(row["Dibuang"]),
                "Fitur Dipilih": int(row["Terpilih"]),
            }
        )

    return pd.DataFrame(rows)


def make_optimal_table(ridge_row, lasso_row, elastic_row):
    """Menyusun tabel lambda optimal berdasarkan CV MSE terkecil."""
    rows = []

    for model_name, row in [
        ("Ridge", ridge_row),
        ("LASSO", lasso_row),
        ("Elastic Net", elastic_row),
    ]:
        rows.append(
            {
                "Model": model_name,
                "Lambda Optimal": row["Lambda"],
                "CV MSE": row["CV MSE"],
            }
        )

    return pd.DataFrame(rows)


def make_final_table(
    ridge_row,
    lasso_row,
    elastic_row,
    test_metrics=None,
):
    """
    Menyusun tabel evaluasi akhir.
    test_metrics digunakan untuk Skenario 1, karena hasil testing
    ditetapkan pada lambda optimal yang sudah dipilih.
    """
    rows = []

    for model_name, row in [
        ("Ridge", ridge_row),
        ("LASSO", lasso_row),
        ("Elastic Net", elastic_row),
    ]:
        metrics = (
            test_metrics[model_name]
            if test_metrics is not None
            else {
                "R² Test": row["Test R²"],
                "MAE Test": row["Test MAE"],
                "MSE Test": row["Test MSE"],
                "RMSE Test": row["Test RMSE"],
            }
        )

        rows.append(
            {
                "Model": model_name,
                "Lambda Optimal": row["Lambda"],
                "R² Test": metrics["R² Test"],
                "MAE Test": metrics["MAE Test"],
                "MSE Test": metrics["MSE Test"],
                "RMSE Test": metrics["RMSE Test"],
                "Fitur Dibuang": int(row["Dibuang"]),
                "Fitur Dipilih": int(row["Terpilih"]),
            }
        )

    return pd.DataFrame(rows)


def render_metric_cards(final_data):
    """Menampilkan ringkasan metrik testing."""
    best_r2 = final_data.loc[final_data["R² Test"].idxmax()]
    best_rmse = final_data.loc[final_data["RMSE Test"].idxmin()]
    best_mae = final_data.loc[final_data["MAE Test"].idxmin()]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model Terbaik (R²)", best_r2["Model"])
    col2.metric("R² Test Tertinggi", f"{best_r2['R² Test']:.4f}")
    col3.metric("RMSE Test Terendah", f"${best_rmse['RMSE Test']:,.2f}")
    col4.metric("MAE Test Terendah", f"${best_mae['MAE Test']:,.2f}")

    return best_r2, best_rmse, best_mae


# ============================================================
# HALAMAN MODEL COMPARISON
# ============================================================

def apply_model_comparison_style():
    st.markdown(
        """
        <style>
        .mc-hero {
            background: linear-gradient(120deg, #142B4A 0%, #243F63 100%);
            padding: 1.35rem 1.6rem;
            border-radius: 16px;
            color: #FFFFFF;
            margin: 0.3rem 0 1.2rem 0;
        }
        .mc-hero h2 {
            color: #FFFFFF;
            margin: 0 0 0.35rem 0;
            font-size: 1.7rem;
        }
        .mc-hero p {
            color: #DCE6F2;
            margin: 0;
            font-size: 0.95rem;
        }
        div[data-testid="stMetric"] {
            background: #F3F6FA;
            border: 1px solid #DCE4EF;
            padding: 0.85rem 1rem;
            border-radius: 12px;
        }
        div[data-testid="stMetricLabel"] {
            color: #526783;
        }
        div[data-testid="stMetricValue"] {
            color: #183653;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_evaluation_page(df):
    apply_model_comparison_style()

    st.markdown(
        """
        <div class="mc-hero">
            <h2>Model Comparison</h2>
            <p>
                Perbandingan performa Ridge, LASSO, dan Elastic Net
                pada skenario multikolinearitas dan high-dimensional data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scenario = st.radio(
        "Pilih skenario pengujian:",
        [
            "Skenario 1: Multikolinearitas",
            "Skenario 2: High-Dimensional",
        ],
        horizontal=True,
        key="model_comparison_scenario",
    )

    # ============================================================
    # SKENARIO 1: MULTIKOLINEARITAS
    # ============================================================

    if scenario == "Skenario 1: Multikolinearitas":

        lambda_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

        ridge_s1 = pd.DataFrame(
            {
                "Lambda": lambda_list,
                "CV R²": [0.5149, 0.5651, 0.7286, 0.7207, 0.7276, 0.7757, 0.7994],
                "CV MSE": [
                    2889560612.0320, 2581293491.7869, 1537020211.4937,
                    1568425299.4137, 1575786939.4218, 1343965020.9278,
                    1235949410.7916,
                ],
                "CV MAE": [
                    20368.4753, 20198.2208, 19305.2047, 19404.6465,
                    19244.2135, 18653.8290, 19277.8444,
                ],
                "CV RMSE": [
                    51078.7317, 48754.3698, 38693.4607, 39043.6052,
                    39099.7776, 36141.9780, 34463.0027,
                ],
                "Test R²": [None, None, None, None, None, None, 0.8384],
                "Test MSE": [None, None, None, None, None, None, 1239499107.7845],
                "Test MAE": [None, None, None, None, None, None, 20096.2084],
                "Test RMSE": [None, None, None, None, None, None, 35206.5208],
                "Dibuang": [0, 0, 0, 0, 0, 0, 0],
                "Terpilih": [259, 259, 259, 259, 259, 259, 259],
            }
        )

        lasso_s1 = pd.DataFrame(
            {
                "Lambda": lambda_list,
                "CV R²": [0.4606, 0.4617, 0.4687, 0.5256, 0.7133, 0.6924, 0.7388],
                "CV MSE": [
                    3216352076.7291, 3209044896.8764, 3166020368.5367,
                    2820576747.2729, 1603711714.0687, 1758977190.4352,
                    1542616652.5397,
                ],
                "CV MAE": [
                    20799.6233, 20794.5064, 20757.2649, 20511.2354,
                    19216.6953, 18759.6227, 18703.0018,
                ],
                "CV RMSE": [
                    54989.6210, 54930.7039, 54559.3893, 51591.6624,
                    39194.5962, 40997.1444, 38534.9896,
                ],
                "Test R²": [None, None, None, None, None, None, 0.8533],
                "Test MSE": [None, None, None, None, None, None, 1125398532.0376],
                "Test MAE": [None, None, None, None, None, None, 18923.7278],
                "Test RMSE": [None, None, None, None, None, None, 33546.9601],
                "Dibuang": [3, 3, 4, 3, 12, 57, 176],
                "Terpilih": [256, 256, 255, 256, 247, 202, 83],
            }
        )

        elastic_s1 = pd.DataFrame(
            {
                "Lambda": lambda_list,
                "CV R²": [0.7380, 0.7164, 0.7586, 0.8021, 0.7182, 0.2972, 0.0384],
                "CV MSE": [
                    1453447736.2987, 1626359546.3490, 1426874983.9734,
                    1214578770.5939, 1717426263.1997, 4202766419.4446,
                    5724160854.0467,
                ],
                "CV MAE": [
                    19230.9445, 19347.1331, 18931.3119, 18534.3294,
                    24824.8111, 45511.8736, 54923.9900,
                ],
                "CV RMSE": [
                    37456.5340, 39731.9717, 37244.6899, 34187.1217,
                    40941.0803, 64551.0371, 75431.6552,
                ],
                "Test R²": [None, None, None, 0.8505, None, None, None],
                "Test MSE": [None, None, None, 1146603953.6830, None, None, None],
                "Test MAE": [None, None, None, 19883.9814, None, None, None],
                "Test RMSE": [None, None, None, 33861.5409, None, None, None],
                "Dibuang": [2, 2, 2, 2, 3, 3, 12],
                "Terpilih": [257, 257, 257, 257, 256, 256, 247],
            }
        )

        st.info(
            """
            **Skenario 1 — Multikolinearitas**

            - Data terdiri dari 1.460 observasi dan 259 fitur.
            - Lambda optimal dipilih berdasarkan CV MSE terkecil melalui
              5-Fold Cross-Validation pada data training.
            - Data testing digunakan untuk evaluasi akhir setelah lambda
              optimal ditentukan.
            """
        )

        st.markdown("### 1. Performa Model")

        selected_lambda = st.select_slider(
            "Pilih nilai lambda (α):",
            options=lambda_list,
            value=1000,
            key="lambda_s1",
            help="Pilih lambda untuk membandingkan hasil cross-validation.",
        )

        r_row = ridge_s1.loc[ridge_s1["Lambda"] == selected_lambda].iloc[0]
        l_row = lasso_s1.loc[lasso_s1["Lambda"] == selected_lambda].iloc[0]
        e_row = elastic_s1.loc[elastic_s1["Lambda"] == selected_lambda].iloc[0]

        performance_df = make_performance_table(r_row, l_row, e_row)

        st.dataframe(
            format_cv_table(performance_df),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Hasil cross-validation untuk λ = **{selected_lambda:g}**.")
        st.divider()

        st.markdown("### 2. Perbandingan CV MSE")
        st.caption(
            "Nilai CV MSE yang lebih rendah menunjukkan kesalahan prediksi "
            "rata-rata yang lebih kecil pada proses cross-validation."
        )

        chart_data = performance_df[["Model", "CV MSE"]].copy()
        render_bar_chart(
            chart_data,
            "Model",
            "CV MSE",
            f"CV MSE pada λ = {selected_lambda:g}",
            "CV MSE (× 10⁹)",
            value_scale=1e9,
        )

        st.divider()

        st.markdown("### 3. Pemilihan Lambda Optimal")
        st.caption(
            "Lambda optimal dipilih berdasarkan CV MSE terkecil pada data training."
        )

        best_ridge = ridge_s1.loc[ridge_s1["CV MSE"].idxmin()]
        best_lasso = lasso_s1.loc[lasso_s1["CV MSE"].idxmin()]
        best_elastic = elastic_s1.loc[elastic_s1["CV MSE"].idxmin()]

        optimal_df = make_optimal_table(best_ridge, best_lasso, best_elastic)
        optimal_display = optimal_df.copy()
        optimal_display["CV MSE"] = optimal_display["CV MSE"].map(
            lambda x: f"{x / 1e9:.3f} × 10⁹"
        )

        st.dataframe(
            optimal_display,
            use_container_width=True,
            hide_index=True,
        )

        optimal_chart = optimal_df.copy()
        optimal_chart["Label"] = optimal_chart.apply(
            lambda row: f"{row['Model']}\n(λ = {row['Lambda Optimal']:g})",
            axis=1,
        )

        render_bar_chart(
            optimal_chart,
            "Label",
            "CV MSE",
            "CV MSE pada Lambda Optimal",
            "CV MSE (× 10⁹)",
            value_scale=1e9,
        )

        st.divider()

        st.markdown("### 4. Evaluasi Akhir pada Data Testing")
        st.caption(
            "Model dievaluasi pada data testing setelah lambda optimal "
            "ditentukan melalui cross-validation."
        )

        test_metrics_s1 = {
            "Ridge": {
                "R² Test": 0.8384,
                "MAE Test": 20096.2084,
                "MSE Test": 1239499107.7845,
                "RMSE Test": 35206.5208,
            },
            "LASSO": {
                "R² Test": 0.8533,
                "MAE Test": 18923.7278,
                "MSE Test": 1125398532.0376,
                "RMSE Test": 33546.9601,
            },
            "Elastic Net": {
                "R² Test": 0.8505,
                "MAE Test": 19883.9814,
                "MSE Test": 1146603953.6830,
                "RMSE Test": 33861.5409,
            },
        }

        final_df = make_final_table(
            best_ridge,
            best_lasso,
            best_elastic,
            test_metrics=test_metrics_s1,
        )

        st.dataframe(
            format_final_table(final_df),
            use_container_width=True,
            hide_index=True,
        )

        best_r2, best_rmse, best_mae = render_metric_cards(final_df)

        st.divider()
        st.markdown("### 5. Perbandingan Test R²")

        final_chart = final_df.copy()
        final_chart["Label"] = final_chart.apply(
            lambda row: f"{row['Model']}\n(λ = {row['Lambda Optimal']:g})",
            axis=1,
        )

        render_bar_chart(
            final_chart,
            "Label",
            "R² Test",
            "Perbandingan Test R² Antar Model",
            "Test R²",
            y_min=0,
        )

        st.divider()
        st.markdown("### 6. Ringkasan Hasil")

        st.markdown(
            f"""
            **Model dengan Test R² tertinggi adalah {best_r2['Model']}**
            pada λ = **{best_r2['Lambda Optimal']:g}**, dengan Test R²
            **{best_r2['R² Test']:.4f}**.

            - **Lambda optimal:** Ridge = {best_ridge['Lambda']:g},
              LASSO = {best_lasso['Lambda']:g}, dan
              Elastic Net = {best_elastic['Lambda']:g}.
            - **CV MSE terendah:** Ridge = {best_ridge['CV MSE'] / 1e9:.3f} × 10⁹,
              LASSO = {best_lasso['CV MSE'] / 1e9:.3f} × 10⁹,
              dan Elastic Net = {best_elastic['CV MSE'] / 1e9:.3f} × 10⁹.
            - **Hasil testing:** {best_r2['Model']} memperoleh MAE
              \\${best_r2['MAE Test']:,.2f} dan RMSE \\${best_r2['RMSE Test']:,.2f}.
            - **Seleksi fitur:** LASSO mempertahankan
              {int(best_lasso['Terpilih'])} fitur dan membuang
              {int(best_lasso['Dibuang'])} fitur, sedangkan Ridge
              mempertahankan seluruh 259 fitur.
            """
        )

        st.caption(
            "Baseline OLS: Test R² = 0.0991 dan RMSE = $83,125.52."
        )

    # ============================================================
    # SKENARIO 2: HIGH-DIMENSIONAL
    # ============================================================

    else:

        lambda_list = [0.1, 1, 10, 100, 1000, 10000, 100000]

        ridge_s2 = pd.DataFrame(
            {
                "Lambda": lambda_list,
                "CV R²": [0.3826, 0.5366, 0.7380, 0.8029, 0.6747, 0.2128, -0.0109],
                "CV MSE": [
                    5105392691.7768, 3887634510.6369, 2383190773.6718,
                    1951264767.7975, 3288538003.7569, 7275388134.9926,
                    9140118541.0680,
                ],
                "CV MAE": [
                    52736.4135, 45083.2823, 32716.0551, 24356.4402,
                    29957.1648, 55371.3376, 64756.0536,
                ],
                "CV RMSE": [
                    71076.9302, 62042.0930, 48089.2615, 42002.1981,
                    54376.5736, 83873.3454, 94522.8271,
                ],
                "Test R²": [None, None, None, 0.8763, None, None, None],
                "Test MSE": [None, None, None, 692225208.7032, None, None, None],
                "Test MAE": [None, None, None, 20618.1399, None, None, None],
                "Test RMSE": [None, None, None, 26310.1731, None, None, None],
                "Dibuang": [0, 0, 0, 0, 0, 0, 0],
                "Terpilih": [259, 259, 259, 259, 259, 259, 259],
            }
        )

        lasso_s2 = pd.DataFrame(
            {
                "Lambda": lambda_list,
                "CV R²": [-0.1500, 0.0322, 0.4529, 0.5683, 0.7543, 0.7381, -0.0449],
                "CV MSE": [
                    9666240977.3555, 7818282236.7087, 4574939943.6852,
                    3663830658.4792, 2289356383.1194, 2517859296.8050,
                    9421912511.0589,
                ],
                "CV MAE": [
                    72906.1980, 64980.6663, 49627.1913, 41230.3121,
                    30326.6538, 27511.6081, 66108.8928,
                ],
                "CV RMSE": [
                    96348.5093, 87079.9423, 67613.4811, 60336.5598,
                    46773.0598, 48470.0046, 96025.2394,
                ],
                "Test R²": [None, None, None, None, 0.7989, None, None],
                "Test MSE": [None, None, None, None, 1125072354.8613, None, None],
                "Test MAE": [None, None, None, None, 24879.6598, None, None],
                "Test RMSE": [None, None, None, None, 33542.0982, None, None],
                "Dibuang": [42, 64, 85, 110, 176, 240, 259],
                "Terpilih": [217, 195, 174, 149, 83, 19, 0],
            }
        )

        elastic_s2 = pd.DataFrame(
            {
                "Lambda": lambda_list,
                "CV R²": [0.7057, 0.8031, 0.7247, 0.3035, 0.0064, -0.0402, -0.0448],
                "CV MSE": [
                    2626948636.3444, 1919172203.7276, 2814776121.9534,
                    6513706958.9610, 8996374778.9136, 9383374315.5137,
                    9421591796.6129,
                ],
                "CV MAE": [
                    34911.2868, 25115.3422, 26806.3534, 51197.9456,
                    64062.9390, 65924.0537, 66107.4202,
                ],
                "CV RMSE": [
                    50696.9065, 41907.8672, 49952.6218, 79092.6056,
                    93746.9429, 95821.3884, 96023.5198,
                ],
                "Test R²": [None, 0.8740, None, None, None, None, None],
                "Test MSE": [None, 705100431.6688, None, None, None, None, None],
                "Test MAE": [None, 20900.1912, None, None, None, None, None],
                "Test RMSE": [None, 26553.7273, None, None, None, None, None],
                "Dibuang": [40, 40, 40, 43, 45, 79, 245],
                "Terpilih": [219, 219, 219, 216, 214, 180, 14],
            }
        )

        st.info(
            """
            **Skenario 2 — High-Dimensional (p > n)**

            Data terdiri dari 200 observasi dan 259 fitur, sehingga jumlah
            fitur lebih banyak daripada jumlah observasi. Lambda optimal
            dipilih berdasarkan CV MSE terkecil melalui 5-Fold
            Cross-Validation pada data training.
            """
        )

        st.markdown("### 1. Performa Model")

        selected_lambda = st.select_slider(
            "Pilih nilai lambda (α):",
            options=lambda_list,
            value=100,
            key="lambda_s2",
            help="Pilih lambda untuk membandingkan hasil cross-validation.",
        )

        r_row = ridge_s2.loc[ridge_s2["Lambda"] == selected_lambda].iloc[0]
        l_row = lasso_s2.loc[lasso_s2["Lambda"] == selected_lambda].iloc[0]
        e_row = elastic_s2.loc[elastic_s2["Lambda"] == selected_lambda].iloc[0]

        performance_df = make_performance_table(r_row, l_row, e_row)

        st.dataframe(
            format_cv_table(performance_df),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(f"Hasil cross-validation untuk λ = **{selected_lambda:g}**.")
        st.divider()

        st.markdown("### 2. Perbandingan CV MSE")
        st.caption(
            "Nilai CV MSE yang lebih rendah menunjukkan kesalahan prediksi "
            "rata-rata yang lebih kecil pada proses cross-validation."
        )

        render_bar_chart(
            performance_df,
            "Model",
            "CV MSE",
            f"CV MSE pada λ = {selected_lambda:g}",
            "CV MSE (× 10⁹)",
            value_scale=1e9,
        )

        st.divider()

        st.markdown("### 3. Pemilihan Lambda Optimal")
        st.caption(
            "Lambda optimal dipilih berdasarkan CV MSE terkecil pada data training."
        )

        best_ridge = ridge_s2.loc[ridge_s2["CV MSE"].idxmin()]
        best_lasso = lasso_s2.loc[lasso_s2["CV MSE"].idxmin()]
        best_elastic = elastic_s2.loc[elastic_s2["CV MSE"].idxmin()]

        optimal_df = make_optimal_table(best_ridge, best_lasso, best_elastic)
        optimal_display = optimal_df.copy()
        optimal_display["CV MSE"] = optimal_display["CV MSE"].map(
            lambda x: f"{x / 1e9:.3f} × 10⁹"
        )

        st.dataframe(
            optimal_display,
            use_container_width=True,
            hide_index=True,
        )

        optimal_chart = optimal_df.copy()
        optimal_chart["Label"] = optimal_chart.apply(
            lambda row: f"{row['Model']}\n(λ = {row['Lambda Optimal']:g})",
            axis=1,
        )

        render_bar_chart(
            optimal_chart,
            "Label",
            "CV MSE",
            "CV MSE pada Lambda Optimal",
            "CV MSE (× 10⁹)",
            value_scale=1e9,
        )

        st.divider()

        st.markdown("### 4. Evaluasi Akhir pada Data Testing")
        st.caption(
            "Model dievaluasi pada data testing setelah lambda optimal "
            "ditentukan melalui cross-validation."
        )

        final_df = make_final_table(best_ridge, best_lasso, best_elastic)

        st.dataframe(
            format_final_table(final_df),
            use_container_width=True,
            hide_index=True,
        )

        best_r2, best_rmse, best_mae = render_metric_cards(final_df)

        st.divider()

        st.markdown("### 5. Perbandingan Test R²")
        st.caption(
            "Perbandingan kemampuan generalisasi model pada data testing."
        )

        final_chart = final_df.copy()
        final_chart["Label"] = final_chart.apply(
            lambda row: f"{row['Model']}\n(λ = {row['Lambda Optimal']:g})",
            axis=1,
        )

        render_bar_chart(
            final_chart,
            "Label",
            "R² Test",
            "Perbandingan Test R² Antar Model",
            "Test R²",
            y_min=min(0, final_df["R² Test"].min() - 0.05),
            y_max_factor=1.15,
        )

        st.divider()

        st.markdown("### 6. Ringkasan Hasil")

        st.markdown(
            f"""
            **Model dengan Test R² tertinggi adalah {best_r2['Model']}**
            pada λ = **{best_r2['Lambda Optimal']:g}**, dengan Test R²
            **{best_r2['R² Test']:.4f}**.

            - **Lambda optimal:** Ridge = {best_ridge['Lambda']:g},
              LASSO = {best_lasso['Lambda']:g}, dan
              Elastic Net = {best_elastic['Lambda']:g}.
            - **CV MSE:** Ridge = {best_ridge['CV MSE'] / 1e9:.3f} × 10⁹,
              LASSO = {best_lasso['CV MSE'] / 1e9:.3f} × 10⁹,
              dan Elastic Net = {best_elastic['CV MSE'] / 1e9:.3f} × 10⁹.
            - **Hasil testing:** {best_r2['Model']} memperoleh MAE
              \\${best_r2['MAE Test']:,.2f} dan RMSE \\${best_r2['RMSE Test']:,.2f}.
            - **Seleksi fitur:** LASSO mempertahankan
              {int(best_lasso['Terpilih'])} fitur dan membuang
              {int(best_lasso['Dibuang'])} fitur.
            """
        )

        st.caption(
            "Baseline OLS: Test R² = -0.3073 dan RMSE = $85,516.3346."
        )