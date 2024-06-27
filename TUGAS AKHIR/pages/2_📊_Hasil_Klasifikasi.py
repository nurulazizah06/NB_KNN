import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image

icon = "gambar/icon4.png"
st.set_page_config(
    page_title="Hasil Analisis",
    page_icon=icon,
)

def read_data_hasil_klasifikasi():
    return pd.read_excel("dataset/smote_data_hasil_klasifikasi.xlsx")

def create_heatmap(matrix, title, vmax):
    urutan_baru_label_unik = ['berat_badan_kurang', 'normal', 'kelebihan_berat_badan', 'obesitas_I',  'obesitas_II']
    heatmap_fig, heatmap_ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, cmap="Oranges", fmt="d", ax=heatmap_ax,
                xticklabels=urutan_baru_label_unik, yticklabels=urutan_baru_label_unik, linewidths=0.4, linecolor='grey',vmin=0, vmax=vmax, cbar=False)
    heatmap_ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    heatmap_ax.set_xlabel("Kelas Klasifikasi", fontsize=10)  
    heatmap_ax.set_ylabel("Kelas Sebenarnya", fontsize=10)  
    heatmap_ax.tick_params(axis='both', labelsize=10)
    plt.xticks(rotation=40, ha='right')
    st.pyplot(heatmap_fig)
    

def show_hasil_analisis_page():
    analysis_page = st.sidebar.selectbox("Hasil Klasifikasi", ["Data Hasil Klasifikasi", "Kinerja Algoritma"])

    if analysis_page == "Data Hasil Klasifikasi":
        tab1_klasifikasi, tab2_klasifikasi = st.tabs(["Data Hasil Klasifikasi", "Distribusi Kelas Obesitas"])

        with tab1_klasifikasi:
            st.markdown("### Data Hasil Klasifikasi")
            with st.spinner("Loading Data..."):
                data_hasil_klasifikasi = read_data_hasil_klasifikasi()
                time.sleep(2)
                st.dataframe(data_hasil_klasifikasi)
            if st.download_button(
                label="Download Data Hasil Klasifikasi",
                data=data_hasil_klasifikasi.to_csv(index=False),
                file_name='data_hasil_klasifikasi.csv',
                mime='text/csv'
                ):
                st.success("File CSV berhasil diunduh")
        
        with tab2_klasifikasi:
            data_hasil_klasifikasi = read_data_hasil_klasifikasi()
            data_ringkasan = pd.DataFrame(columns=['kelas_obesitas', 'keterangan'])
            data_ringkasan = pd.concat([data_ringkasan, pd.DataFrame({'kelas_obesitas': data_hasil_klasifikasi['kelas_obesitas'], 'keterangan': 'kelas_sebenarnya'})], ignore_index=True)
            data_ringkasan = pd.concat([data_ringkasan, pd.DataFrame({'kelas_obesitas': data_hasil_klasifikasi['klasifikasi_naive_bayes'], 'keterangan': 'klasifikasi_naive_bayes'})], ignore_index=True)
            data_ringkasan = pd.concat([data_ringkasan, pd.DataFrame({'kelas_obesitas': data_hasil_klasifikasi['klasifikasi_k_nearest_neighbor'], 'keterangan': 'klasifikasi_k_nearest_neighbor'})], ignore_index=True)
            urutan_obesitas = ['berat_badan_kurang', 'normal', 'kelebihan_berat_badan', 'obesitas_I', 'obesitas_II']
            warna = {'kelas_sebenarnya': '#4b4b4b',
                    'klasifikasi_naive_bayes': '#FF9642',
                    'klasifikasi_k_nearest_neighbor': '#FFE05D'}
            plt.figure(figsize=(8, 7))
            sns.set_theme(style="darkgrid")
            plot = sns.countplot(x='kelas_obesitas', hue='keterangan', data=data_ringkasan, order=urutan_obesitas, palette=warna)
            plot.set_xlabel("Kelas Obesitas", labelpad=6)
            plot.set_ylabel("Jumlah", labelpad=5)
            # plt.xticks(rotation=18, ha='right')
            plt.suptitle('Distribusi Kelas Obesitas Berdasarkan Data Hasil Klasifikasi', fontsize=17, fontweight='bold')
            plt.tight_layout()
            st.pyplot(plt)

            img_stream = BytesIO()
            plt.savefig(img_stream, format='png')
            img_stream.seek(0)
            
            if st.download_button(
                label="Download Visualisasi",
                data=img_stream,
                file_name='Distribusi Kelas Obesitas.png',
                key='bar_chart_download'
                ):
                st.success("Gambar Visualisasi Berhasil Diunduh")

    elif analysis_page == "Kinerja Algoritma":
        # tab1_kinerja, tab2_kinerja = st.tabs(["Confusion Matrix", "Kinerja Algoritma"])

        # with tab1_kinerja:
            # Pembagian Data
            data = pd.read_excel("dataset/smote_full_data_obesitas.xlsx")
            X = data.drop(columns='kelas_obesitas', axis = 1)
            Y = data['kelas_obesitas']
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.02, stratify = Y, random_state = 0)

            # Normalisasi
            minmax = MinMaxScaler()
            minmax.fit(X_train)
            X_train_nomalisasi = minmax.transform(X_train)
            X_test_nomalisasi = minmax.transform(X_test)

            # Klasifikasi Naive Bayes
            klasifikasiNV = GaussianNB()
            start_time_NV = time.time()
            klasifikasiNV.fit(X_train_nomalisasi, Y_train)
            training_end_time_NV = time.time()
            Y_predictNV = klasifikasiNV.predict(X_test_nomalisasi)
            end_time_NV = time.time()
            prediction_end_time_NV = time.time()
            training_execution_time_NV = training_end_time_NV - start_time_NV
            prediction_execution_time_NV = prediction_end_time_NV - training_end_time_NV
            execution_time_NV1 = prediction_end_time_NV - start_time_NV
            execution_time_NV2 = end_time_NV - start_time_NV

            # Klasifikasi K-Nearest Neighbor
            klasifikasiKNN = KNeighborsClassifier(n_neighbors = 7)
            start_time_KNN = time.time()
            klasifikasiKNN.fit(X_train_nomalisasi, Y_train)
            training_end_time_KNN = time.time()
            Y_predictKNN = klasifikasiKNN.predict(X_test_nomalisasi)
            end_time_KNN = time.time()
            prediction_end_time_KNN = time.time()
            training_execution_time_KNN = training_end_time_KNN - start_time_KNN
            prediction_execution_time_KNN = prediction_end_time_KNN - training_end_time_KNN
            execution_time_KNN1 = prediction_end_time_KNN - start_time_KNN
            execution_time_KNN2 = end_time_KNN - start_time_KNN

            # confusion matrix
            label_unikNV = np.unique(np.concatenate((Y_test, Y_predictNV)))
            label_unikKNN = np.unique(np.concatenate((Y_test, Y_predictKNN)))
            matrixNV = confusion_matrix(Y_test, Y_predictNV, labels=label_unikNV)
            matrixKNN = confusion_matrix(Y_test, Y_predictKNN, labels=label_unikKNN)
            dataframe_CM_NV = pd.DataFrame(matrixNV, index=label_unikNV, columns=label_unikNV)
            dataframe_CM_KNN = pd.DataFrame(matrixKNN, index=label_unikKNN, columns=label_unikKNN)
            urutan_baru_label_unik = ['berat_badan_kurang', 'normal', 'kelebihan_berat_badan', 'obesitas_I', 'obesitas_II']
            matrixNV_baru = dataframe_CM_NV.reindex(index=urutan_baru_label_unik, columns=urutan_baru_label_unik)
            matrixKNN_baru = dataframe_CM_KNN.reindex(index=urutan_baru_label_unik, columns=urutan_baru_label_unik)

            # kinerja algoritma
            akurasiNV = accuracy_score(Y_test, Y_predictNV)
            presisiNV = precision_score(Y_test, Y_predictNV, average='macro')
            recalNV = recall_score(Y_test, Y_predictNV, average='macro')
            F1NV = f1_score(Y_test, Y_predictNV, average='macro')

            akurasiKNN = accuracy_score(Y_test, Y_predictKNN)
            presisiKNN = precision_score(Y_test, Y_predictKNN, average='macro')
            recalKNN = recall_score(Y_test, Y_predictKNN, average='macro')
            F1KNN = f1_score(Y_test, Y_predictKNN, average='macro')

            # Mencari nilai maksimum untuk cbar
            vmax = max(matrixNV.max(), matrixKNN.max())

            # Menampilkan heatmap
            st.markdown("### Perbandingan Kinerja Algoritma _Naive Bayes_ dan _K-Nearest Neighbor_")

            col1, col2 = st.columns(2)
            with col1:
                img_hm_nv = BytesIO()
                create_heatmap(matrixNV_baru, "Confusion Matrix Algoritma Naive Bayes", vmax)
                plt.savefig(img_hm_nv, format='png', bbox_inches='tight')
                img_hm_nv.seek(0)

                st.text("")

                colkiri, coltengah, colkanan = st.columns([1, 3, 1])
                with coltengah:
                    st.markdown("**_Naive Bayes_**")
                    st.markdown(f"Akurasi: {round(akurasiNV, 3)} ({round(accuracy_score(Y_test, Y_predictNV)*100)}%)")
                    st.markdown(f"Presisi: {round(presisiNV, 3)} ({round(presisiNV*100)}%)")
                    st.markdown(f"Recal: {round(recalNV, 3)} ({round(recalNV*100)}%)")
                    st.markdown(f"F1: {round(F1NV, 3)} ({round(F1NV*100)}%)")
                    st.markdown(f"Waktu Pelatihan: {round(training_execution_time_NV, 3)} detik")
                    st.markdown(f"Waktu Pengujian: {round(prediction_execution_time_NV, 3)} detik")

            with col2:
                img_hm_knn = BytesIO()
                create_heatmap(matrixKNN_baru, "Confusion Matrix Algoritma K-Nearest Neighbor", vmax)
                plt.savefig(img_hm_knn, format='png', bbox_inches='tight')
                img_hm_knn.seek(0)

                st.text("")

                colkiri, coltengah, colkanan = st.columns([1, 3, 1])
                with coltengah:
                    st.markdown("**_K-Nearest Neighbor_**")
                    st.markdown(f"Akurasi: {round(akurasiKNN, 3)}  ({round(akurasiKNN*100)}%)")
                    st.markdown(f"Presisi: {round(presisiKNN, 3)} ({round(presisiKNN*100)}%)")
                    st.markdown(f"Recall: {round(recalKNN, 3)} ({round(recalKNN*100)}%)")
                    st.markdown(f"F1: {round(F1KNN, 3)} ({round(F1KNN*100)}%)")
                    st.markdown(f"Waktu Pelatihan: {round(training_execution_time_KNN, 3)} detik")
                    st.markdown(f"Waktu Pengujian: {round(prediction_execution_time_KNN, 3)} detik")

            # Buat dokumen PDF
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            # Tambahkan judul
            doc_title = Paragraph("Perbandingan Kinerja Algoritma <i>Naive Bayes</i> dan <i>K-Nearest Neighbor</i>", styles['Title'])

            # Tambahkan konten ke dalam dokumen
            content = [doc_title, Spacer(1, 12)]

            # Membuat data untuk tabel dengan tata letak yang diinginkan
            data = [
                [
                    # Confusion Matrix Naive Bayes
                    Image(img_hm_nv, width=240, height=240), 
                    # Confusion Matrix KNN
                    Image(img_hm_knn, width=240, height=240)
                ],
                [
                    # Kinerja Naive Bayes
                    Paragraph("<b><i>Naive Bayes</i></b><br/><br/>" + f"Akurasi: {round(akurasiNV, 3)} ({round(akurasiNV*100)}%)<br/><br/>Presisi: {round(presisiNV, 3)} ({round(presisiNV*100)}%)<br/><br/>Recal: {round(recall_score(Y_test, Y_predictNV, average='weighted'), 3)} ({round(recall_score(Y_test, Y_predictNV, average='weighted')*100)}%)<br/><br/>F1: {round(f1_score(Y_test, Y_predictNV, average='weighted'), 3)} ({round(f1_score(Y_test, Y_predictNV, average='weighted')*100)}%)<br/><br/>Waktu Pelatihan: {round(training_execution_time_NV, 3)} detik<br/><br/>Waktu Pengujian: {round(prediction_execution_time_NV, 3)} detik"),

                    # Kinerja KNN
                    Paragraph("<b><i>K-Nearest Neighbor</i></b><br/><br/>" + f"Akurasi: {round(akurasiKNN, 3)} ({round(akurasiKNN*100)}%)<br/><br/>Presisi: {round(presisiKNN, 3)} ({round(presisiKNN*100)}%)<br/><br/>Recal: {round(recalKNN, 3)} ({round(recalKNN*100)}%)<br/><br/>F1: {round(F1KNN, 3)} ({round(F1KNN*100)}%)<br/><br/>Waktu Pelatihan: {round(training_execution_time_KNN, 3)} detik<br/><br/>Waktu Pengujian: {round(prediction_execution_time_KNN, 3)} detik")
                ]
            ]

            # Membuat tabel
            table = Table(data, colWidths=[250, 250], rowHeights=[250, 150])
            table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
                                    ('LEFTPADDING', (0, 1), (-1, 1), 50)]))

            # Tambahkan tabel ke dalam konten
            content.append(table)

            # Tambahkan konten ke dalam dokumen
            doc.build(content)

            # Simpan dokumen PDF ke dalam buffer
            pdf_buffer.seek(0)

            st.text("")

            # Menampilkan tombol unduhan dalam kolom tengah
            col1, col2, col3 = st.columns([1.5, 2, 1])
            with col2:
                st.download_button(
                    label="Download Perbandingan Algoritma",
                    data=pdf_buffer.getvalue(),
                    file_name="Perbandingan Kinerja Algoritma Naive Bayes dan K-Nearest Neighbor.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    show_hasil_analisis_page()
