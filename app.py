import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np


# importing the model

model = pkl.load(open('model.pkl','rb'))
health_data = pd.read_pickle("health_data.pkl")

st.title("Are you a diabetic ? Let's predict !!" )

# Selecting the number of pregnancies
Pregnancies = st.selectbox("Select the number of pregnancies",health_data['Pregnancies'].unique())

# Typing in the glucose 
Glucose = st.number_input("Input your glucose level")

# Typing in the blood pressure
BloodPressure = st.number_input("Input your blood pressure")

# Typing in the skin thickness
SkinThickness = st.number_input("Input your SkinThickness")

# Typing the insulin level
Insulin = st.number_input("Input your insulin")

# Type in your BMI
BMI = st.number_input("What's your BMI")

# Type DiabetesPedigreeFunction
DiabetesPedigreeFunction = st.number_input("Type in DiabetesPedigreeFunction")

# Inputting the age
Age = st.number_input("What's your age")

# Prediction
if st.button("Predict"):
	query=np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
	query = query.reshape(1,8)
	if model.predict(query) == [0]:
		st.title("Non-Diabetic")
	else:
		st.title("Diabetic")