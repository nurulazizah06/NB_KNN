import streamlit as st
import pandas as pd
import time

icon = "gambar/icon4.png"
st.set_page_config(
    page_title="Data",
    page_icon=icon,
)

def load_data_latih():
    time.sleep(2)
    return pd.read_excel("dataset/smote_training_data_obesitas_baru.xlsx")
def load_data_uji():
    time.sleep(2)
    return pd.read_excel("dataset/smote_testing_data_obesitas_baru.xlsx")

def show_data_page():
    subpage = st.sidebar.selectbox("Data", ["Data Latih", "Data Uji"])

    if subpage == "Data Latih":
        st.markdown("### Data Latih")
        with st.spinner("Loading Data..."):
            data_latih = load_data_latih()
            st.dataframe(data_latih)

        if st.download_button(
                label="Download Data Latih",
                data=data_latih.to_csv(index=False),
                file_name='data_latih.csv',
                mime='text/csv'
            ):
            st.success("File CSV berhasil diunduh")

    elif subpage == "Data Uji":
        st.markdown("### Data Uji")
        with st.spinner("Loading Data..."):
            data_uji = load_data_uji()
        st.dataframe(data_uji)

        if st.download_button(
                label="Download Data Uji",
                data=data_uji.to_csv(index=False),
                file_name='data_uji.csv',
                mime='text/csv'
            ):
            st.success("File CSV berhasil diunduh")

if __name__ == "__main__":
    show_data_page()
