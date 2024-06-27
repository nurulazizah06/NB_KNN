import streamlit as st

icon = "gambar/icon4.png"
st.set_page_config(
    page_title="NB-KNN",
    page_icon=icon,
)

def pengertian_klasifikasi():
    st.markdown(
        """
        Klasifikasi adalah proses mengelompokkan objek atau data baru ke dalam kategori kelas berdasarkan atribut yang dimilikinya.
        Dalam metode klasifikasi, terdapat dua jenis algoritma berdasarkan cara pelatihannya:
        - **Eager learner** melakukan
            pelatihan pada data latih, sehingga model dihasilkan sebelum mengklasifikasikan data baru, salah satu contohnya algoritma _naive bayes_.
        - **Lazy learner** tidak
            melakukan pelatihan pada data latih, maka model dihasilkan saat data baru diberikan untuk diklasifikasikan, salah satu contohnya algoritma _k-nearest neighbors_.
        """
    )

def pengertian_naive_bayes():
    st.markdown("Algoritma ini didasarkan pada Teorema Bayes, yaitu menghitung probabilitas kelas dari probabilitas fitur atau atribut. Salah satu asumsi penting dalam _naive bayes_ ialah semua atribut dianggap independen satu sama lain.")

def pengertian_knn():
    st.markdown("Algoritma ini mengklasifikasikan data baru berdasarkan mayoritas kelas dari tetangga terdekat menggunakan metrik jarak seperti _euclidean distance_.")


def main():
    st.title("Perbandingan Algoritma *Naive Bayes* dan *K-Nearest Neighbor*")
    st.markdown("Sistem ini bertujuan untuk menampilkan hasil klasifikasi dan perbandingan kinerja model klasifikasi dari algoritma _naive bayes_ dan _k-nearest neighbor_ dalam mengklasifikasikan tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik.")
    
    with st.expander("**Klasifikasi**"):
        pengertian_klasifikasi()

    with st.expander("**_Algoritma Naive Bayes_**"):
        pengertian_naive_bayes()

    with st.expander("**_Algoritma K-Nearest Neighbors_**"):
        pengertian_knn()

if __name__ == "__main__":
    main()