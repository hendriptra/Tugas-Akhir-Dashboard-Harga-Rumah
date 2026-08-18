import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def render_evaluation_page(df):

    # ============================================================
    # ⚖️ MODEL COMPARISON
    # ============================================================

    st.subheader("⚖️ Model Comparison")

    st.caption(
        "Perbandingan performa Ridge, LASSO, dan Elastic Net "
        "pada skenario multikolinearitas dan high-dimensional data."
    )

    # ============================================================
    # SCENARIO SELECTOR
    # ============================================================

    scenario = st.radio(
        "🎯 Pilih Skenario Pengujian:",
        ["Skenario 1: Multikolinearitas", "Skenario 2: High-Dimensional"],
        horizontal=True,
    )

    # ============================================================
    # SCENARIO 1: MULTIKOLINEARITAS
    # ============================================================

    if scenario == "Skenario 1: Multikolinearitas":

        # --------------------------------------------------------
        # DATASET HASIL EKSPERIMEN
        # --------------------------------------------------------

        lambda_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

        # ========================================================
        # RIDGE
        # ========================================================

        ridge_s1 = pd.DataFrame(
            {
                "Lambda": lambda_list,

                "CV R²": [
                    0.5155,
                    0.5659,
                    0.7299,
                    0.7216,
                    0.7282,
                    0.7760,
                    0.7994,
                ],

                "CV MSE": [
                    2886501201.3964,
                    2576212435.1887,
                    1528064281.4598,
                    1563884199.9291,
                    1572788353.1742,
                    1342168253.8276,
                    1236138386.6638,
                ],

                "CV MAE": [
                    20358.5446,
                    20185.5629,
                    19271.3926,
                    19344.9052,
                    19216.6743,
                    18640.3067,
                    19261.4941,
                ],

                "CV RMSE": [
                    51041.0922,
                    48704.5008,
                    38575.9201,
                    38982.7643,
                    39054.5059,
                    36117.9295,
                    34462.5403,
                ],

                "Test R²": [
                    0.11694,
                    0.25449,
                    0.72713,
                    0.82762,
                    0.84594,
                    0.86157,
                    0.83888,
                ],

                "Test MSE": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    1235874460.4871,
                ],

                "Test MAE": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    20045.1931,
                ],

                "Test RMSE": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    35155.0062,
                ],

                "Dibuang": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],

                "Terpilih": [
                    259,
                    259,
                    259,
                    259,
                    259,
                    259,
                    259,
                ],
            }
        )

        # ========================================================
        # LASSO
        # ========================================================

        lasso_s1 = pd.DataFrame(
            {
                "Lambda": lambda_list,

                "CV R²": [
                    0.5393,
                    0.5404,
                    0.5487,
                    0.5865,
                    0.6987,
                    0.7117,
                    0.7405,
                ],

                "CV MSE": [
                    2760697780.3908,
                    2753494670.3216,
                    2698296478.6227,
                    2448190383.0134,
                    1711075903.9840,
                    1658785315.5764,
                    1533518303.0600,
                ],

                "CV MAE": [
                    20528.7153,
                    20519.1763,
                    20435.2601,
                    20080.9848,
                    19274.6227,
                    18479.9453,
                    18641.4969,
                ],

                "CV RMSE": [
                    51344.8480,
                    51270.3526,
                    50644.8414,
                    47661.5132,
                    40509.0040,
                    39798.6588,
                    38423.8645,
                ],

                "Test R²": [
                    0.38986,
                    0.39065,
                    0.39842,
                    0.47204,
                    0.82097,
                    0.84050,
                    0.85351,
                ],

                "Test MSE": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    1123591691.2032,
                ],

                "Test MAE": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    18897.5487,
                ],

                "Test RMSE": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    33520.0193,
                ],

                "Dibuang": [
                    2,
                    2,
                    4,
                    4,
                    18,
                    66,
                    175,
                ],

                "Terpilih": [
                    257,
                    257,
                    255,
                    255,
                    241,
                    193,
                    84,
                ],
            }
        )

        # ========================================================
        # ELASTIC NET
        # ========================================================

        elastic_s1 = pd.DataFrame(
            {
                "Lambda": lambda_list,

                "CV R²": [
                    0.7389,
                    0.7172,
                    0.7590,
                    0.8023,
                    0.7177,
                    0.2949,
                    0.0379,
                ],

                "CV MSE": [
                    1448546228.3300,
                    1622612427.9584,
                    1424785955.4289,
                    1213985033.7616,
                    1721070425.0307,
                    4216044944.7315,
                    5727104194.3780,
                ],

                "CV MAE": [
                    19173.7680,
                    19301.3968,
                    18912.6364,
                    18514.0350,
                    24846.4073,
                    45611.5986,
                    54944.9262,
                ],

                "CV RMSE": [
                    37387.8040,
                    39677.3583,
                    37214.4276,
                    34178.0945,
                    40982.0389,
                    64653.9148,
                    75451.2030,
                ],

                "Test R²": [
                    0.83312,
                    0.83672,
                    0.86132,
                    0.85105,
                    0.71600,
                    0.28049,
                    0.03768,
                ],

                "Test MSE": [
                    None,
                    None,
                    None,
                    1142506425.8747,
                    None,
                    None,
                    None,
                ],

                "Test MAE": [
                    None,
                    None,
                    None,
                    19822.5181,
                    None,
                    None,
                    None,
                ],

                "Test RMSE": [
                    None,
                    None,
                    None,
                    33800.9826,
                    None,
                    None,
                    None,
                ],

                "Dibuang": [
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    12,
                ],

                "Terpilih": [
                    257,
                    257,
                    257,
                    257,
                    257,
                    256,
                    247,
                ],
            }
        )

        # --------------------------------------------------------
        # INFO SCENARIO
        # --------------------------------------------------------

        st.info(
            """
            📌 **Skenario 1 — Multikolinearitas**

            - Menggunakan seluruh data dengan **n = 1.460 observasi**
            dan **p = 259 fitur**.
            - Nilai **λ optimal ditentukan berdasarkan CV MSE terkecil**
            menggunakan **5-Fold Cross-Validation** pada data training.
            - Setelah λ optimal diperoleh, model dievaluasi menggunakan
            data testing yang tidak digunakan dalam proses pemilihan
            hyperparameter.
            """
        )

        # ========================================================
        # 1. PERFORMA MODEL
        # ========================================================

        st.markdown("### 1. Performa Model")

        st.caption(
            "Perbandingan performa model berdasarkan hasil "
            "5-Fold Cross-Validation pada data training."
        )

        selected_lambda = st.select_slider(
            "🎛️ **Pilih Nilai Hyperparameter Lambda (α):**",
            options=lambda_list,
            value=1000,
            key="lambda_s1",
            help=(
                "Digunakan untuk mengeksplorasi performa "
                "cross-validation pada setiap nilai lambda."
            ),
        )

        r_row = ridge_s1[
            ridge_s1["Lambda"] == selected_lambda
        ].iloc[0]

        l_row = lasso_s1[
            lasso_s1["Lambda"] == selected_lambda
        ].iloc[0]

        e_row = elastic_s1[
            elastic_s1["Lambda"] == selected_lambda
        ].iloc[0]

        performance_df = pd.DataFrame(
            [
                {
                    "Model": "Ridge",
                    "CV R²": r_row["CV R²"],
                    "CV MAE": r_row["CV MAE"],
                    "CV MSE": r_row["CV MSE"],
                    "CV RMSE": r_row["CV RMSE"],
                    "Fitur Dibuang": r_row["Dibuang"],
                    "Fitur Dipilih": r_row["Terpilih"],
                },
                {
                    "Model": "LASSO",
                    "CV R²": l_row["CV R²"],
                    "CV MAE": l_row["CV MAE"],
                    "CV MSE": l_row["CV MSE"],
                    "CV RMSE": l_row["CV RMSE"],
                    "Fitur Dibuang": l_row["Dibuang"],
                    "Fitur Dipilih": l_row["Terpilih"],
                },
                {
                    "Model": "Elastic Net",
                    "CV R²": e_row["CV R²"],
                    "CV MAE": e_row["CV MAE"],
                    "CV MSE": e_row["CV MSE"],
                    "CV RMSE": e_row["CV RMSE"],
                    "Fitur Dibuang": e_row["Dibuang"],
                    "Fitur Dipilih": e_row["Terpilih"],
                },
            ]
        )

        disp_performance = performance_df.copy()

        disp_performance["CV R²"] = disp_performance["CV R²"].map(
            lambda x: f"{x:.4f}"
        )

        disp_performance["CV MAE"] = disp_performance["CV MAE"].map(
            lambda x: f"${x:,.2f}"
        )

        disp_performance["CV MSE"] = disp_performance["CV MSE"].map(
            lambda x: f"{x / 1e9:.3f} × 10⁹"
        )

        disp_performance["CV RMSE"] = disp_performance["CV RMSE"].map(
            lambda x: f"${x:,.2f}"
        )

        st.dataframe(
            disp_performance,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"💡 Menampilkan performa cross-validation pada "
            f"λ = **{selected_lambda:g}**."
        )

        st.divider()

        # ========================================================
        # 2. CV MSE COMPARISON
        # ========================================================

        st.markdown("### 2. CV MSE Comparison")

        st.caption(
            f"Perbandingan **CV MSE** untuk masing-masing model "
            f"pada λ = {selected_lambda}. Nilai yang lebih rendah "
            f"menunjukkan performa cross-validation yang lebih baik."
        )

        fig, ax = plt.subplots(figsize=(8, 4.5))

        bars = ax.bar(
            performance_df["Model"],
            performance_df["CV MSE"] / 1e9,
            color=["#4C78A8", "#2A9D8F", "#AFC7E3"],
            width=0.6,
        )

        ax.set_ylabel("CV MSE (× 10⁹)")
        ax.set_title(
            f"Cross-Validation MSE (λ = {selected_lambda})",
            fontsize=13,
            fontweight="bold",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, value in zip(
            bars,
            performance_df["CV MSE"] / 1e9
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (value * 0.02),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # ========================================================
        # 3. PEMILIHAN LAMBDA OPTIMAL
        # ========================================================

        st.markdown(
            "### 3. Pemilihan Lambda Optimal Berdasarkan Cross-Validation"
        )

        st.caption(
            "Lambda optimal dipilih berdasarkan **CV MSE terkecil**. "
            "Data testing tidak digunakan dalam proses pemilihan "
            "hyperparameter."
        )

        # --------------------------------------------------------
        # Lambda optimal berdasarkan CV MSE TERKECIL
        # --------------------------------------------------------

        best_ridge_s1 = ridge_s1.loc[
            ridge_s1["CV MSE"].idxmin()
        ]

        best_lasso_s1 = lasso_s1.loc[
            lasso_s1["CV MSE"].idxmin()
        ]

        best_elastic_s1 = elastic_s1.loc[
            elastic_s1["CV MSE"].idxmin()
        ]

        optimal_cv_s1 = pd.DataFrame(
            [
                {
                    "Model": "Ridge",
                    "Lambda Optimal": best_ridge_s1["Lambda"],
                    "CV MSE": best_ridge_s1["CV MSE"],
                },
                {
                    "Model": "LASSO",
                    "Lambda Optimal": best_lasso_s1["Lambda"],
                    "CV MSE": best_lasso_s1["CV MSE"],
                },
                {
                    "Model": "Elastic Net",
                    "Lambda Optimal": best_elastic_s1["Lambda"],
                    "CV MSE": best_elastic_s1["CV MSE"],
                },
            ]
        )

        disp_cv_s1 = optimal_cv_s1.copy()

        disp_cv_s1["CV MSE"] = disp_cv_s1["CV MSE"].map(
            lambda x: f"{x / 1e9:.3f} × 10⁹"
        )

        st.dataframe(
            disp_cv_s1,
            use_container_width=True,
            hide_index=True,
        )

                # --------------------------------------------------------
        # CV MSE COMPARISON PADA LAMBDA OPTIMAL
        # --------------------------------------------------------

        fig_cv, ax_cv = plt.subplots(figsize=(8, 4.5))

        bars_cv = ax_cv.bar(
            optimal_cv_s1["Model"],
            optimal_cv_s1["CV MSE"] / 1e9,
            color=["#4C78A8", "#2A9D8F", "#AFC7E3"],
            width=0.6,
        )

        ax_cv.set_xlabel("Model")
        ax_cv.set_ylabel("CV MSE (× 10⁹)")

        ax_cv.set_title(
            "CV MSE pada Lambda Optimal",
            fontsize=13,
            fontweight="bold",
        )

        ax_cv.spines["top"].set_visible(False)
        ax_cv.spines["right"].set_visible(False)

        # Menampilkan nilai CV MSE di atas bar
        for bar, value in zip(
            bars_cv,
            optimal_cv_s1["CV MSE"] / 1e9
        ):
            ax_cv.text(
                bar.get_x() + bar.get_width() / 2,
                value + (value * 0.02),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        st.pyplot(fig_cv)
        plt.close(fig_cv)

        st.divider()

        # ========================================================
        # 4. FINAL TEST EVALUATION
        # ========================================================

        st.markdown("### 4. Final Test Evaluation")

        st.caption(
            "Setelah lambda optimal ditentukan menggunakan CV MSE, "
            "model dievaluasi pada data testing untuk memperoleh "
            "estimasi performa pada data yang belum pernah digunakan."
        )

        final_test_s1 = pd.DataFrame(
            [
                {
                    "Model": "Ridge",
                    "Lambda Optimal": best_ridge_s1["Lambda"],
                    "R² Test": 0.83888,
                    "MAE Test": 20045.1931,
                    "MSE Test": 1235874460.4871,
                    "RMSE Test": 35155.0062,
                    "Fitur Dibuang": best_ridge_s1["Dibuang"],
                    "Fitur Dipilih": best_ridge_s1["Terpilih"],
                },
                {
                    "Model": "LASSO",
                    "Lambda Optimal": best_lasso_s1["Lambda"],
                    "R² Test": 0.85351,
                    "MAE Test": 18897.5487,
                    "MSE Test": 1123591691.2032,
                    "RMSE Test": 33520.0193,
                    "Fitur Dibuang": best_lasso_s1["Dibuang"],
                    "Fitur Dipilih": best_lasso_s1["Terpilih"],
                },
                {
                    "Model": "Elastic Net",
                    "Lambda Optimal": best_elastic_s1["Lambda"],
                    "R² Test": 0.85100,
                    "MAE Test": 19822.5181,
                    "MSE Test": 1142506425.8747,
                    "RMSE Test": 33800.9826,
                    "Fitur Dibuang": best_elastic_s1["Dibuang"],
                    "Fitur Dipilih": best_elastic_s1["Terpilih"],
                },
            ]
        )

        disp_final_s1 = final_test_s1.copy()

        disp_final_s1["R² Test"] = disp_final_s1["R² Test"].map(
            lambda x: f"{x:.4f}"
        )

        disp_final_s1["MAE Test"] = disp_final_s1["MAE Test"].map(
            lambda x: f"${x:,.2f}"
        )

        disp_final_s1["MSE Test"] = disp_final_s1["MSE Test"].map(
            lambda x: f"{x / 1e9:.3f} × 10⁹"
        )

        disp_final_s1["RMSE Test"] = disp_final_s1["RMSE Test"].map(
            lambda x: f"${x:,.2f}"
        )

        st.dataframe(
            disp_final_s1,
            use_container_width=True,
            hide_index=True,
        )

                # ========================================================
        # FINAL MODEL COMPARISON - KPI
        # ========================================================

        best_final_s1 = final_test_s1.loc[
            final_test_s1["R² Test"].idxmax()
        ]

        best_rmse_s1 = final_test_s1.loc[
            final_test_s1["RMSE Test"].idxmin()
        ]

        best_mae_s1 = final_test_s1.loc[
            final_test_s1["MAE Test"].idxmin()
        ]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Model Terbaik",
            best_final_s1["Model"],
            help="Model dengan Test R² tertinggi.",
        )

        col2.metric(
            "R² Test Tertinggi",
            f"{best_final_s1['R² Test']:.4f}",
        )

        col3.metric(
            "RMSE Test Terendah",
            f"${best_rmse_s1['RMSE Test']:,.2f}",
        )

        col4.metric(
            "MAE Test Terendah",
            f"${best_mae_s1['MAE Test']:,.2f}",
        )

        st.divider()

        # ========================================================
        # 5. FINAL MODEL COMPARISON
        # ========================================================

        st.markdown("### 5. Final Model Comparison")

        fig_final_r2, ax_final_r2 = plt.subplots(
            figsize=(8, 4.5)
        )

        bars_final_r2 = ax_final_r2.bar(
            final_test_s1["Model"],
            final_test_s1["R² Test"],
            color=["#4C78A8", "#2A9D8F", "#AFC7E3"],
            width=0.6,
        )

        ax_final_r2.set_ylim(0.7, 0.9)
        ax_final_r2.set_ylabel("Test R²")

        ax_final_r2.set_title(
            "Final Test R² Comparison",
            fontsize=13,
            fontweight="bold",
        )

        ax_final_r2.spines["top"].set_visible(False)
        ax_final_r2.spines["right"].set_visible(False)

        for bar, value in zip(
            bars_final_r2,
            final_test_s1["R² Test"]
        ):
            ax_final_r2.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.005,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        st.pyplot(fig_final_r2)
        plt.close(fig_final_r2)

        st.divider()

        # ========================================================
        # 6. EVALUATION INSIGHT
        # ========================================================

        st.markdown("### 6. Evaluation Insight")

        st.success(
            f"""
            🏆 **Model dengan Test R² Tertinggi: {best_final_s1['Model']}
            (λ = {best_final_s1['Lambda Optimal']:g})**

            * **Pemilihan Hyperparameter:** Lambda optimal ditentukan
              berdasarkan **CV MSE terkecil** pada data training.
              Berdasarkan hasil cross-validation, Ridge memilih
              λ = **{best_ridge_s1['Lambda']:g}**, LASSO memilih
              λ = **{best_lasso_s1['Lambda']:g}**, dan Elastic Net
              memilih λ = **{best_elastic_s1['Lambda']:g}**.

            * **Performa Cross-Validation:** Ridge menghasilkan
              CV MSE = **{best_ridge_s1['CV MSE'] / 1e9:.3f} × 10⁹**,
              LASSO = **{best_lasso_s1['CV MSE'] / 1e9:.3f} × 10⁹**,
              dan Elastic Net = **{best_elastic_s1['CV MSE'] / 1e9:.3f} × 10⁹**.

            * **Evaluasi Final:** Pada data testing, {best_final_s1['Model']}
              menghasilkan **Test R² = {best_final_s1['R² Test']:.4f}**,
              dengan **MAE = \${best_final_s1['MAE Test']:,.2f}**,
              **MSE = {best_final_s1['MSE Test'] / 1e9:.3f} × 10⁹**,
              dan **RMSE = \${best_final_s1['RMSE Test']:,.2f}**.

            * **Pemilihan Model:** Berdasarkan Test R², {best_final_s1['Model']}
              memberikan performa prediksi terbaik. Di sisi lain, LASSO
              memiliki **MAE dan RMSE terendah** serta menggunakan jumlah
              fitur yang lebih sedikit, sehingga menghasilkan model yang
              lebih **sparse dan efisien**.
            """
        )

        st.caption(
            "💡 **Baseline OLS:** Test R² = 0.0991 dengan RMSE = $83,125.52."
        )

    # ============================================================
    # SCENARIO 2: HIGH-DIMENSIONAL
    # ============================================================

    else:

        # --------------------------------------------------------
        # DATASET HASIL EKSPERIMEN
        # --------------------------------------------------------

        lambda_list_s2 = [
            0.1,
            1,
            10,
            100,
            1000,
            10000,
            100000,
        ]

        ridge_s2 = pd.DataFrame(
            {
                "Lambda": lambda_list_s2,
                "CV R²": [
                    0.37606,
                    0.53180,
                    0.73661,
                    0.80281,
                    0.67410,
                    0.21067,
                    -0.01134,
                ],
                "CV MSE": [
                    5146835809.3526,
                    3917579037.2218,
                    2392655355.8991,
                    1951175915.9518,
                    3293799598.7149,
                    7293869234.5992,
                    9143621303.8069,
                ],
                "CV MAE": [
                    52880.5015,
                    45266.3416,
                    32784.9242,
                    24348.9179,
                    30006.2688,
                    55504.0576,
                    64779.5052,
                ],
                "CV RMSE": [
                    71352.7723,
                    62281.2485,
                    48198.4493,
                    42014.5131,
                    54425.6134,
                    83984.5639,
                    94541.5387,
                ],
                "Test R²": [
                    -0.19526,
                    0.14688,
                    0.69706,
                    0.87764,
                    0.81973,
                    0.34901,
                    0.04883,
                ],
                "Test MSE": [
                    None,
                    None,
                    None,
                    684469763.3884,
                    None,
                    None,
                    None,
                ],
                "Test MAE": [
                    61260.849,
                    50927.360,
                    32932.143,
                    20589.374,
                    24246.879,
                    48347.656,
                    58597.414,
                ],
                "Test RMSE": [
                    81769.567,
                    69081.996,
                    41166.331,
                    26162.373,
                    31755.550,
                    60345.829,
                    72944.066,
                ],
                "Dibuang": [0, 0, 0, 0, 0, 0, 0],
                "Terpilih": [259, 259, 259, 259, 259, 259, 259],
            }
        )

        lasso_s2 = pd.DataFrame(
            {
                "Lambda": lambda_list_s2,
                "CV R²": [
                    -0.16052,
                    0.00601,
                    0.45388,
                    0.57136,
                    0.75441,
                    0.73815,
                    -0.04487,
                ],
                "CV MSE": [
                    9678509113.0778,
                    7993647832.7868,
                    4558979371.4262,
                    3620688044.5118,  # <-- Diperbaiki: sebelumnya tertulis 320688044.5118 (kurang angka '6')
                    2287718588.4161,  # Minimum CV MSE -> Best Lambda = 1000
                    2517859296.8050,
                    9421912511.0589,
                ],
                "CV MAE": [
                    73804.8779,
                    66831.1085,
                    50103.4658,
                    41190.1683,
                    30315.2501,
                    27511.6081,
                    66108.8928,
                ],
                "CV RMSE": [
                    95726.4802,
                    87947.4990,
                    67502.2279,
                    60003.2852,
                    46760.0101,
                    48470.0046,
                    96025.2394,
                ],
                "Test R²": [
                    -0.62891,
                    -0.50115,
                    -0.55686,
                    0.26516,
                    0.79891,
                    0.88758,
                    -0.00017,
                ],
                "Test MSE": [
                    None,
                    None,
                    None,
                    None,
                    1124879866.6893,
                    None,
                    None,
                ],
                "Test MAE": [
                    73070.387,
                    73402.736,
                    73478.504,
                    46060.216,
                    24878.695,
                    18324.845,
                    60191.139,
                ],
                "Test RMSE": [
                    95457.356,
                    91637.470,
                    93322.224,
                    64114.772,
                    33539.229,
                    25077.259,
                    74799.149,
                ],
                "Dibuang": [42, 66, 84, 110, 174, 240, 259],
                "Terpilih": [217, 193, 175, 149, 85, 19, 0],
            }
        )

        elastic_s2 = pd.DataFrame(
            {
                "Lambda": lambda_list_s2,
                "CV R²": [
                    0.70388,
                    0.80290,
                    0.72441,
                    0.30115,
                    0.00576,
                    -0.04030,
                    -0.04484,
                ],
                "CV MSE": [
                    2639657835.6721,
                    1919875448.6163,
                    2816797200.7991,
                    6534105304.5665,
                    9001579573.6760,
                    9383916604.6091,
                    9421591796.6129,
                ],
                "CV MAE": [
                    34983.7070,
                    25148.5474,
                    26837.1852,
                    51354.6477,
                    64098.2783,
                    65927.6894,
                    66107.4202,
                ],
                "CV RMSE": [
                    50831.8936,
                    41929.8724,
                    49976.1545,
                    79222.6822,
                    93774.9873,
                    95824.2561,
                    96023.5198,
                ],
                "Test R²": [
                    0.65535,
                    0.87537,
                    0.83750,
                    0.40265,
                    0.05962,
                    0.00533,
                    -0.00012,
                ],
                "Test MSE": [
                    None,
                    697203321.3465,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                "Test MAE": [
                    34616.103,
                    20888.513,
                    22912.604,
                    46281.274,
                    58238.804,
                    60013.719,
                    60189.561,
                ],
                "Test RMSE": [
                    43908.681,
                    26404.608,
                    30149.930,
                    57806.170,
                    72528.949,
                    74593.207,
                    74797.329,
                ],
                "Dibuang": [40, 40, 40, 43, 45, 79, 245],
                "Terpilih": [219, 219, 219, 216, 214, 180, 14],
            }
        )

        # --------------------------------------------------------
        # INFO SCENARIO
        # --------------------------------------------------------

        st.info(
            """
            📌 **Skenario 2 — High-Dimensional ($p > n$)**

            Data diperoleh melalui pengambilan sampel acak (*random sampling*) sebanyak 200 observasi dengan 259 fitur. 
            Kondisi ini menghasilkan data berdimensi tinggi ($p > n$), 
            di mana jumlah fitur ($p$) lebih banyak daripada jumlah sampel ($n$).

            Lambda optimal ditentukan berdasarkan **5-Fold Cross-Validation (CV MSE)**
            pada data training. Data testing kemudian digunakan untuk evaluasi akhir.
            """
        )

        # --------------------------------------------------------
        # INTERACTIVE LAMBDA EXPLORATION
        # --------------------------------------------------------

        st.markdown("### 🔎 Eksplorasi Performa Berdasarkan Lambda")

        selected_lambda_s2 = st.select_slider(
            "🎛️ **Pilih Nilai Hyperparameter Lambda (α):**",
            options=lambda_list_s2,
            value=100,
            key="lambda_s2",
            help="Digunakan untuk mengeksplorasi performa model pada setiap nilai lambda.",
        )

        r_row_s2 = ridge_s2[ridge_s2["Lambda"] == selected_lambda_s2].iloc[0]
        l_row_s2 = lasso_s2[lasso_s2["Lambda"] == selected_lambda_s2].iloc[0]
        e_row_s2 = elastic_s2[elastic_s2["Lambda"] == selected_lambda_s2].iloc[0]

        # ========================================================
        # 1. PERFORMA MODEL (CV)
        # ========================================================

        st.markdown(
            f"#### 1. Performa Model pada $\\lambda = {selected_lambda_s2}$"
        )

        eval_df_s2 = pd.DataFrame(
            [
                {
                    "Model": "Ridge",
                    "CV R²": r_row_s2["CV R²"],
                    "CV MAE": r_row_s2["CV MAE"],
                    "CV MSE": r_row_s2["CV MSE"],
                    "CV RMSE": r_row_s2["CV RMSE"],
                    "Fitur Dibuang": int(r_row_s2["Dibuang"]),
                    "Fitur Dipilih": int(r_row_s2["Terpilih"]),
                },
                {
                    "Model": "LASSO",
                    "CV R²": l_row_s2["CV R²"],
                    "CV MAE": l_row_s2["CV MAE"],
                    "CV MSE": l_row_s2["CV MSE"],
                    "CV RMSE": l_row_s2["CV RMSE"],
                    "Fitur Dibuang": int(l_row_s2["Dibuang"]),
                    "Fitur Dipilih": int(l_row_s2["Terpilih"]),
                },
                {
                    "Model": "Elastic Net",
                    "CV R²": e_row_s2["CV R²"],
                    "CV MAE": e_row_s2["CV MAE"],
                    "CV MSE": e_row_s2["CV MSE"],
                    "CV RMSE": e_row_s2["CV RMSE"],
                    "Fitur Dibuang": int(e_row_s2["Dibuang"]),
                    "Fitur Dipilih": int(e_row_s2["Terpilih"]),
                },
            ]
        )

        disp_df_s2 = eval_df_s2.copy()
        disp_df_s2["CV R²"] = disp_df_s2["CV R²"].map(lambda x: f"{x:.4f}")
        disp_df_s2["CV MAE"] = disp_df_s2["CV MAE"].map(lambda x: f"${x:,.2f}")
        disp_df_s2["CV MSE"] = disp_df_s2["CV MSE"].map(lambda x: f"{x / 1e9:.3f} × 10⁹")
        disp_df_s2["CV RMSE"] = disp_df_s2["CV RMSE"].map(lambda x: f"${x:,.2f}")

        st.dataframe(
            disp_df_s2,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "💡 Performa model pada tahap ini dievaluasi menggunakan "
            "**Cross-Validation pada data training**. "
            "Nilai CV MSE digunakan sebagai dasar pemilihan lambda optimal."
        )

        st.divider()

        # ========================================================
        # 2. CV MSE COMPARISON
        # ========================================================

        st.markdown("#### 2. CV MSE Comparison")

        st.caption(
            f"Perbandingan **CV MSE** masing-masing model pada $\\lambda = {selected_lambda_s2}$."
        )

        # Palet warna modern & profesional (Ridge: Royal Blue, LASSO: Coral/Red-Orange, Elastic Net: Teal/Emerald)
        model_colors = {
            "Ridge": "#2563EB",
            "LASSO": "#F97316",
            "Elastic Net": "#10B981",
        }
        bar_colors = [
            model_colors.get(model, "#6B7280")
            for model in eval_df_s2["Model"]
        ]

        fig_s2, ax_s2 = plt.subplots(figsize=(8, 4.5), facecolor="none")
        ax_s2.set_facecolor("none")

        mse_values = eval_df_s2["CV MSE"] / 1e9

        bars_s2 = ax_s2.bar(
            eval_df_s2["Model"],
            mse_values,
            width=0.45,
            color=bar_colors,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )

        # Grid horizontal halus di latar belakang
        ax_s2.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

        ax_s2.set_ylabel("CV MSE (× 10⁹)", fontsize=11, fontweight="medium")
        ax_s2.set_title(
            f"Model Performance Based on CV MSE (λ = {selected_lambda_s2})",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        # Bersihkan garis tepi grafik
        ax_s2.spines["top"].set_visible(False)
        ax_s2.spines["right"].set_visible(False)
        ax_s2.spines["left"].set_alpha(0.4)
        ax_s2.spines["bottom"].set_alpha(0.4)

        # Beri ruang di atas bar agar teks angka tidak bertabrakan dengan batas atas
        max_val = mse_values.max()
        ax_s2.set_ylim(0, max_val * 1.18)

        # Label nilai di atas bar
        for bar, value in zip(bars_s2, mse_values):
            ax_s2.text(
                bar.get_x() + bar.get_width() / 2,
                value + (max_val * 0.02),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color="#1F2937",
            )

        plt.tight_layout()
        st.pyplot(fig_s2)
        plt.close(fig_s2)

        st.divider()

        # ========================================================
        # 3. PEMILIHAN LAMBDA OPTIMAL
        # ========================================================

        st.markdown(
            "#### 3. Pemilihan Lambda Optimal Berdasarkan Cross-Validation"
        )

        st.caption(
            "Lambda optimal dipilih berdasarkan **CV MSE terkecil**. "
            "Data testing tidak digunakan dalam proses pemilihan hyperparameter."
        )

        best_ridge_s2 = ridge_s2.loc[ridge_s2["CV MSE"].idxmin()]
        best_lasso_s2 = lasso_s2.loc[lasso_s2["CV MSE"].idxmin()]
        best_elastic_s2 = elastic_s2.loc[elastic_s2["CV MSE"].idxmin()]

        optimal_cv_s2 = pd.DataFrame(
            [
                {
                    "Model": "Ridge",
                    "Lambda Optimal": best_ridge_s2["Lambda"],
                    "CV MSE": best_ridge_s2["CV MSE"],
                },
                {
                    "Model": "LASSO",
                    "Lambda Optimal": best_lasso_s2["Lambda"],
                    "CV MSE": best_lasso_s2["CV MSE"],
                },
                {
                    "Model": "Elastic Net",
                    "Lambda Optimal": best_elastic_s2["Lambda"],
                    "CV MSE": best_elastic_s2["CV MSE"],
                },
            ]
        )

        disp_cv_s2 = optimal_cv_s2.copy()
        disp_cv_s2["CV MSE"] = disp_cv_s2["CV MSE"].map(lambda x: f"{x / 1e9:.3f} × 10⁹")

        st.dataframe(
            disp_cv_s2,
            use_container_width=True,
            hide_index=True,
        )

        # --------------------------------------------------------
        # GRAFIK BATANG CV MSE OPTIMAL
        # --------------------------------------------------------

        # Format label sumbu X agar menampilkan nama model dan nilai lambda optimalnya
        x_labels = [
            f"{row['Model']}\n(λ = {row['Lambda Optimal']:g})"
            for _, row in optimal_cv_s2.iterrows()
        ]

        model_colors = {
            "Ridge": "#2563EB",
            "LASSO": "#F97316",
            "Elastic Net": "#10B981",
        }
        bar_colors = [
            model_colors.get(model, "#6B7280")
            for model in optimal_cv_s2["Model"]
        ]

        fig_opt, ax_opt = plt.subplots(figsize=(8, 4.5), facecolor="none")
        ax_opt.set_facecolor("none")

        opt_mse_values = optimal_cv_s2["CV MSE"] / 1e9

        bars_opt = ax_opt.bar(
            x_labels,
            opt_mse_values,
            width=0.45,
            color=bar_colors,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )

        ax_opt.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax_opt.set_ylabel("Optimal CV MSE (× 10⁹)", fontsize=11, fontweight="medium")
        ax_opt.set_title(
            "Optimal CV MSE Comparison Across Models",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        ax_opt.spines["top"].set_visible(False)
        ax_opt.spines["right"].set_visible(False)
        ax_opt.spines["left"].set_alpha(0.4)
        ax_opt.spines["bottom"].set_alpha(0.4)

        max_opt_val = opt_mse_values.max()
        ax_opt.set_ylim(0, max_opt_val * 1.18)

        # Anotasi nilai MSE di atas setiap bar
        for bar, val in zip(bars_opt, opt_mse_values):
            ax_opt.text(
                bar.get_x() + bar.get_width() / 2,
                val + (max_opt_val * 0.02),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color="#1F2937",
            )

        plt.tight_layout()
        st.pyplot(fig_opt)
        plt.close(fig_opt)

        st.divider()

        # ========================================================
        # 4. FINAL TEST EVALUATION
        # ========================================================

        st.markdown("#### 4. Final Test Evaluation")

        st.caption(
            "Setelah lambda optimal ditentukan berdasarkan **CV MSE**, "
            "model dievaluasi menggunakan data testing."
        )

        final_test_s2 = pd.DataFrame(
            [
                {
                    "Model": "Ridge",
                    "Lambda Optimal": best_ridge_s2["Lambda"],
                    "R² Test": best_ridge_s2["Test R²"],
                    "MAE Test": best_ridge_s2["Test MAE"],
                    "MSE Test": best_ridge_s2["Test MSE"],
                    "RMSE Test": best_ridge_s2["Test RMSE"],
                    "Fitur Dibuang": int(best_ridge_s2["Dibuang"]),
                    "Fitur Dipilih": int(best_ridge_s2["Terpilih"]),
                },
                {
                    "Model": "LASSO",
                    "Lambda Optimal": best_lasso_s2["Lambda"],
                    "R² Test": best_lasso_s2["Test R²"],
                    "MAE Test": best_lasso_s2["Test MAE"],
                    "MSE Test": best_lasso_s2["Test MSE"],
                    "RMSE Test": best_lasso_s2["Test RMSE"],
                    "Fitur Dibuang": int(best_lasso_s2["Dibuang"]),
                    "Fitur Dipilih": int(best_lasso_s2["Terpilih"]),
                },
                {
                    "Model": "Elastic Net",
                    "Lambda Optimal": best_elastic_s2["Lambda"],
                    "R² Test": best_elastic_s2["Test R²"],
                    "MAE Test": best_elastic_s2["Test MAE"],
                    "MSE Test": best_elastic_s2["Test MSE"],
                    "RMSE Test": best_elastic_s2["Test RMSE"],
                    "Fitur Dibuang": int(best_elastic_s2["Dibuang"]),
                    "Fitur Dipilih": int(best_elastic_s2["Terpilih"]),
                },
            ]
        )

        # Penentuan nilai optimal untuk penanda kolom kesimpulan
        best_r2_val = final_test_s2["R² Test"].max()
        best_rmse_val = final_test_s2["RMSE Test"].min()
        best_mae_val = final_test_s2["MAE Test"].min()

        disp_final_s2 = final_test_s2.copy()

        # Formatting tampilan angka & mata uang
        disp_final_s2["R² Test"] = disp_final_s2["R² Test"].map(lambda x: f"{x:.4f}")
        disp_final_s2["MAE Test"] = disp_final_s2["MAE Test"].map(lambda x: f"${x:,.2f}")
        disp_final_s2["MSE Test"] = disp_final_s2["MSE Test"].map(
            lambda x: f"{x / 1e9:.3f} × 10⁹" if pd.notnull(x) else "-"
        )
        disp_final_s2["RMSE Test"] = disp_final_s2["RMSE Test"].map(lambda x: f"${x:,.2f}")

        st.dataframe(
            disp_final_s2,
            use_container_width=True,
            hide_index=True,
        )

        best_final_s2 = final_test_s2.loc[final_test_s2["R² Test"].idxmax()]
        best_rmse_s2 = final_test_s2.loc[final_test_s2["RMSE Test"].idxmin()]
        best_mae_s2 = final_test_s2.loc[final_test_s2["MAE Test"].idxmin()]

        # --------------------------------------------------------
        # METRIC CARDS RINGKASAN
        # --------------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Model Terbaik (R²)", best_final_s2["Model"])
        col2.metric("R² Test Tertinggi", f"{best_final_s2['R² Test']:.4f}")
        col3.metric("RMSE Test Terendah", f"${best_rmse_s2['RMSE Test']:,.2f}")
        col4.metric("MAE Test Terendah", f"${best_mae_s2['MAE Test']:,.2f}")

        st.write("")

        st.divider()

        # ========================================================
        # 5. FINAL MODEL COMPARISON
        # ========================================================

        st.markdown("#### 5. Final Model Comparison")

        st.caption(
            "Perbandingan performa generalisasi model pada **Data Testing** "
            "menggunakan metrik evaluasi $R^2$, RMSE, dan MAE."
        )

        # --------------------------------------------------------
        # GRAFIK BATANG PERBANDINGAN R² TEST
        # --------------------------------------------------------
        x_labels_r2 = [
            f"{row['Model']}\n(λ = {row['Lambda Optimal']:g})"
            for _, row in final_test_s2.iterrows()
        ]

        model_colors = {
            "Ridge": "#2563EB",
            "LASSO": "#F97316",
            "Elastic Net": "#10B981",
        }
        bar_colors = [
            model_colors.get(model, "#6B7280")
            for model in final_test_s2["Model"]
        ]

        fig_r2, ax_r2 = plt.subplots(figsize=(8, 4.5), facecolor="none")
        ax_r2.set_facecolor("none")

        r2_values = final_test_s2["R² Test"].values

        bars_r2 = ax_r2.bar(
            x_labels_r2,
            r2_values,
            width=0.45,
            color=bar_colors,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )

        ax_r2.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax_r2.set_ylabel("Test R² Score", fontsize=11, fontweight="medium")
        ax_r2.set_title(
            "Perbandingan Test R² Score Antar Model Optimal",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        ax_r2.spines["top"].set_visible(False)
        ax_r2.spines["right"].set_visible(False)
        ax_r2.spines["left"].set_alpha(0.4)
        ax_r2.spines["bottom"].set_alpha(0.4)

        # Baseline y = 0
        ax_r2.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

        # Atur batas sumbu Y agar label nilai memiliki ruang cukup
        min_r2 = min(r2_values.min(), 0)
        max_r2 = r2_values.max()
        ax_r2.set_ylim(min_r2 - 0.05, max_r2 * 1.15)

        # Anotasi nilai R² di atas bar
        for bar, val in zip(bars_r2, r2_values):
            offset = 0.02 if val >= 0 else -0.05
            va_pos = "bottom" if val >= 0 else "top"
            ax_r2.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"{val:.4f}",
                ha="center",
                va=va_pos,
                fontsize=10.5,
                fontweight="bold",
                color="#1F2937",
            )

        plt.tight_layout()
        st.pyplot(fig_r2)
        plt.close(fig_r2)

        st.divider()

        # ========================================================
        # 6. EVALUATION INSIGHT
        # ========================================================

        st.markdown("#### 6. Evaluation Insight")

        st.success(
            f"""
            🏆 **Model Terbaik (Akurasi Prediksi): {best_final_s2['Model']}** 
            dengan **λ = {best_final_s2['Lambda Optimal']:g}** dan **Test R² = {best_final_s2['R² Test']:.4f}**.

            * **Pemilihan Hyperparameter:** Lambda optimal ditentukan berdasarkan **CV MSE terkecil** pada data training. Ridge memilih **λ = {best_ridge_s2['Lambda']:g}** (CV MSE = {best_ridge_s2['CV MSE'] / 1e9:.3f} × 10⁹), LASSO memilih **λ = {best_lasso_s2['Lambda']:g}** (CV MSE = {best_lasso_s2['CV MSE'] / 1e9:.3f} × 10⁹), dan Elastic Net memilih **λ = {best_elastic_s2['Lambda']:g}** (CV MSE = {best_elastic_s2['CV MSE'] / 1e9:.3f} × 10⁹).

            * **Evaluasi Final:** Setelah parameter optimal diaplikasikan ke data uji, model **{best_final_s2['Model']}** memberikan performa terbaik dengan **Test R² = {best_final_s2['R² Test']:.4f}**, **MAE = \${best_final_s2['MAE Test']:,.2f}**, dan **RMSE = \${best_final_s2['RMSE Test']:,.2f}**.

            * **Seleksi Fitur & Karakteristik Model:** 
                * **Ridge Regression** menghasilkan akurasi tertinggi dengan mempertahankan seluruh **259 fitur** (*dense representation*), membuktikan bahwa penyusutan bobot koefisien ($L_2$ regularization) sangat efektif ketika hampir semua fitur menyumbang informasi prediktif.
                * **LASSO** pada λ = {best_lasso_s2['Lambda']:g} membuang **{int(best_lasso_s2['Dibuang'])} fitur** dan hanya mempertahankan **{int(best_lasso_s2['Terpilih'])} fitur** (*sparse representation*), memberikan interpretabilitas tinggi meskipun akurasi akhirnya sedikit di bawah Ridge.
            """
        )

        st.caption(
            "💡 **Baseline OLS:** Test R² = -0.3048 dengan RMSE = \$85,434.34."
        )