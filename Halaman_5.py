import streamlit as st

def render_about_page(df=None):
    st.header('👤 About Me & Research Information')
    st.caption('Informasi Akademis Peneliti dan Tugas Akhir')

    st.divider()

    # Layout 2 Kolom: Profil & Detail Penelitian
    col_profile, col_detail = st.columns([1, 2])

    with col_profile:
        st.markdown('### 👨‍🎓 Profil Peneliti')
        
        # Card / Container Profil
        with st.container(border=True):
            
            # --- MENAMBAHKAN FOTO PROFIL ---
            try:
                st.image("foto_hendri.png", use_container_width=True)
            except Exception:
                st.info("💡 Letakkan file foto Anda (misal: 'foto_hendri.png') di folder yang sama agar foto tampil di sini.")
            
            st.subheader('Hendri Saputra')
            st.markdown("""
                * **NIM:** 10122024 
                * **Program Studi:** Sarjana Matematika
                * **Fakultas:** FMIPA
                * **Instansi:** Institut Teknologi Bandung (ITB)
                """)

            st.divider()
            st.markdown('**📬 Tautan Kontak:**')
            st.markdown("""
                * ✉️ **Email:** `henriptra@gmail.com`
                * 🔗 **LinkedIn:** [Profil LinkedIn](https://linkedin.com/in/hendri-ptra/)
                * 💻 **GitHub:** [Repository GitHub](https://github.com/hendriptra)
                """)

    with col_detail:
        st.markdown('### 📚 Informasi Tugas Akhir / Seminar')
        with st.container(border=True):
            st.markdown('**📌 Judul Tugas Akhir:**')
            st.info(
                '**ANALISIS PERBANDINGAN REGRESI RIDGE DAN LASSO DALAM MENGATASI'
                ' MASALAH MULTIKOLINEARITAS DAN DIMENSI TINGGI PADA PREDIKSI HARGA'
                ' JUAL RUMAH**'
            )

            col_dosen1, col_dosen2, col_dosen3 = st.columns(3)

            with col_dosen1:
                st.markdown('**👨‍🏫 Dosen Pembimbing:**')
                st.write('Dr. Sandy Vantika, S.Si., M.Si.')

            with col_dosen2:
                st.markdown('**👩‍🏫 Dosen Penguji I:**')
                st.write('Maharani Ahsani Ummi, S.Si., M.Si., Ph.D.')

            with col_dosen3:
                st.markdown('**👩‍🏫 Dosen Penguji II:**')
                st.write('Coming Soon')

            st.markdown('**🏛️ Instansi:** Institut Teknologi Bandung (ITB)')

            st.markdown('**🎯 Topik/Fokus Penelitian:**')
            st.markdown("""
            - Analisis Regresi Ter-regularisasi ($L_1$ & $L_2$ Norm)
            - Penanganan Multikolinearitas
            - Matriks Dimensi Tinggi ($p > n$)
            - Pemodelan *Machine Learning* Terapan
            """)