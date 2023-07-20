import streamlit as st
import pickle
import numpy as np
import sklearn

def load_model():
    with open('sal_pred.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

dec_tree_reg = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        'United States of America',
        'Germany',
        'United Kingdom of Great Britain and Northern Ireland',
        'Canada',
        'India',
        'France',
        'Netherlands',
        'Australia',
        'Spain',
        'Brazil',
        'Sweden',
        'Italy',
        'Poland',
        'Switzerland',
        'Denmark',
        'Norway',
        'Israel',
        'Portugal',
        'Austria',
        'Finland',
        'Belgium',
        'Russian Federation',
        'New Zealand',
        'Ukraine',
        'Turkey'
    )

    education = (
        "Less than a Bachelor's",
        "Bachelor’s degree",
        "Master’s degree",
        "Post Grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = dec_tree_reg.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
show_predict_page()