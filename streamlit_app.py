import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

# Set page config to wide layout
st.set_page_config(layout="wide")

# Load dataset if available
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

# Sidebar Navigation
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png", width=200)
    st.title("Visuallab")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

# Upload Section
if choice == "Upload":
    st.header("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

# Profiling Section
if choice == "Profiling":
    st.header("Exploratory Data Analysis")
    if 'df' in locals():
        profile_df = df.profile_report()
        st_profile_report(profile_df)
    else:
        st.warning("Please upload a dataset first.")

# Modeling Section
if choice == "Modelling":
    st.header("Modeling")
    if 'df' in locals():
        chosen_target = st.selectbox('Choose the Target Column', df.columns)

        # Convert string and categorical columns to numeric
        numeric_df = df.select_dtypes(include=['number'])
        categorical_columns = df.select_dtypes(exclude=['number']).columns

        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes

        if st.button('Run Modeling'):
            setup(df, target=chosen_target, silent=True, preprocess=False, fold_shuffle=True)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
    else:
        st.warning("Please upload a dataset first.")

# Download Section
if choice == "Download":
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    else:
        st.warning("No model available to download. Please run modeling first.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #999;'>© Visuallab</p>", unsafe_allow_html=True)
