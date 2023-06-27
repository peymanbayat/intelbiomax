import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

pickle_in = open('Diabetes.pkl', 'rb')
classifier = pickle.load(pickle_in)

def predict():
    st.sidebar.header('IntelBioMax Co.')
    # select = st.sidebar.selectbox('Select Form', ['Form 1'], key='1')
    # if not st.sidebar.checkbox("Hide", True, key='2'):
    st.title('IntelBioMax (Suitable Type of Antibiotic Prediction)')
    st.markdown('This trained dataset is originally collected by IntelBioMax Company.')
    st.markdown('We will be very greatful if you share your data in this regard with us.')

    name = st.text_input("Patient's Name (Optional)")
    pregnancy = st.number_input("Please connect the device to your computer, Data Reading from the Devics:")
    st.markdown('Number of time periods consideration in the plot (read from device)')
    
    glucose = st.number_input("Plasma Glucose Concentration :")
    st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

    bp =  st.number_input("Diastolic blood pressure (mm Hg):")
    st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

    skin = st.number_input("Triceps skin fold thickness (mm):")
    st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

    insulin = st.number_input("2-Hour serum insulin (mu U/ml):")
    st.markdown('Insulin: 2-Hour serum insulin (mu U/ml)')


    bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
    st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

    
    
   
   
    dpf = st.selectbox(
    'Antibiotic Types',
    ('1: "Penicillins"', '2: "Cephalosporins"', '3: "Tetracyclines"', '4:"Aminoglycosides"'))
    st.write('You selected:', dpf)

    if dpf == '1: "Penicillins"':
        dpf = 1
    if dpf == '2: "Cephalosporins"':
        dpf = 2
    if dpf == '3: "Tetracyclines"':
        dpf = 3
    if dpf == '4: "Aminoglycosides"':
        dpf =4
    

        

    age = st.number_input("Age:")
    st.markdown('Age: Age (years)')

    submit = st.button('Predict')
    st.markdown('Outcome: Class variable (0 or 1)')

    if submit:
        prediction = classifier.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])
        if prediction == 0:
            st.write('Congratulation!', name,'It seems you have selected a fit type of Antibiotic')
        else:
            st.write(name,", It seems you can choose a more fitted Antibiotic")

def main():
    new_title = '<p style="font-size: 42px;">Welcome The Congenital Disabilities Prediction App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    read_me = st.markdown("""
    The application is built using Streamlit  
    to demonstrate Fit Type of Antibiotic Prediction. It performs prediction on multiple parameters
                                  """)
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox(
        "MODE", ("About", "Predict Antibiotic Type Effects"))
    if choice == "Predict Antibiotic Type Effects":
        read_me_0.empty()
        read_me.empty()
        predict()
    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
    
