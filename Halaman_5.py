import streamlit as st
import base64
from pathlib import Path


def apply_about_page_style():
    st.markdown(
        """
        <style>
        .about-hero {
            background: linear-gradient(120deg, #142B4A 0%, #243F63 100%);
            padding: 1.35rem 1.6rem;
            border-radius: 16px;
            color: #FFFFFF;
            margin: 0.3rem 0 1.2rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
        }

        .about-hero h2 {
            color: #FFFFFF;
            margin: 0 0 0.35rem 0;
            font-size: 1.7rem;
        }

        .about-hero p {
            color: #DCE6F2;
            margin: 0;
            font-size: 0.95rem;
        }

        .about-hero-logo {
            width: 125px;
            height: 125px;
            object-fit: contain;
            flex-shrink: 0;
            background: #FFFFFF;
            border-radius: 12px;
            padding: 7px;
        }

        .about-profile-name {
            color: #183653;
            font-size: 1.35rem;
            font-weight: 700;
            margin: 0.6rem 0 0.15rem 0;
        }

        .about-profile-caption {
            color: #65758B;
            margin-bottom: 0.9rem;
        }

        .about-thesis {
            background: #F1F5FA;
            border-left: 4px solid #243F63;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            color: #183653;
            font-weight: 600;
            line-height: 1.6;
        }

        @media (max-width: 640px) {
            .about-hero {
                padding: 1.1rem;
                gap: 0.8rem;
            }

            .about-hero h2 {
                font-size: 1.4rem;
            }

            .about-hero-logo {
                width: 58px;
                height: 58px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_about_page(df=None):
    apply_about_page_style()

    # Baca logo dan ubah menjadi base64 agar dapat ditampilkan di HTML
    logo_path = Path(__file__).parent / "logo_itb.png"

    with open(logo_path, "rb") as file:
        logo_base64 = base64.b64encode(file.read()).decode("utf-8")

    # Header navy dengan logo ITB di pojok kanan
    st.markdown(
        f"""
        <div class="about-hero">
            <div>
                <h2>Profil Peneliti</h2>
                <p>Informasi akademik peneliti dan tugas akhir</p>
            </div>
            <img
                class="about-hero-logo"
                src="data:image/png;base64,{logo_base64}"
                alt="Logo Institut Teknologi Bandung"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_profile, col_detail = st.columns([1, 2], gap="large")

    # ============================================================
    # PROFIL PENELITI
    # ============================================================
    with col_profile:
        with st.container(border=True):
            try:
                foto_path = Path(__file__).parent / "foto_hendri.png"
                st.image(str(foto_path), use_container_width=True)
            except Exception:
                st.info(
                    "Letakkan file `foto_hendri.png` di folder aplikasi "
                    "agar foto profil dapat ditampilkan."
                )

            st.markdown(
                """
                <div class="about-profile-name">Hendri Saputra</div>
                <div class="about-profile-caption">
                    Mahasiswa Sarjana Matematika · ITB
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**NIM:** 10122024")
            st.markdown("**Program Studi:** Sarjana Matematika")
            st.markdown("**Fakultas:** FMIPA")
            st.markdown("**Instansi:** Institut Teknologi Bandung")

            st.divider()
            st.markdown("**Kontak dan Tautan**")
            st.markdown(
                """
                ✉️ [henriptra@gmail.com](mailto:henriptra@gmail.com)

                🔗 [LinkedIn](https://linkedin.com/in/hendri-ptra/)

                💻 [GitHub](https://github.com/hendriptra)
                """
            )

    # ============================================================
    # INFORMASI TUGAS AKHIR
    # ============================================================
    with col_detail:
        with st.container(border=True):
            st.subheader("Informasi Tugas Akhir")

            st.markdown("**Judul Penelitian**")
            st.markdown(
                """
                <div class="about-thesis">
                    Analisis Perbandingan Regresi Ridge dan LASSO dalam Mengatasi
                    Masalah Multikolinearitas dan Dimensi Tinggi pada Prediksi
                    Harga Jual Rumah
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown("**Tim Akademik**")

            st.markdown("**Dosen Pembimbing**")
            st.write("Dr. Sandy Vantika, S.Si., M.Si.")

            st.write("")
            col_dosen1, col_dosen2 = st.columns(2, gap="medium")

            with col_dosen1:
                st.markdown("**Dosen Penguji I**")
                st.write("Maharani Ahsani Ummi, S.Si., M.Si., Ph.D.")

            with col_dosen2:
                st.markdown("**Dosen Penguji II**")
                st.write("Coming Soon")

            st.divider()
            st.markdown("**Fokus Penelitian**")
            st.markdown(
                """
                - Regresi terregularisasi dengan norma $L_1$ dan $L_2$
                - Penanganan multikolinearitas
                - Pemodelan pada kondisi dimensi tinggi $p > n$
                - Penerapan metode machine learning untuk prediksi harga rumah
                """
            )

            st.divider()
            st.caption(
                "Institut Teknologi Bandung · FMIPA · Sarjana Matematika"
            )
