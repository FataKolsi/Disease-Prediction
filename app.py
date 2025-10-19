#importing all the important libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#building the sidebar of the web app which will help us navigate through the different sections of the entire application
rad=st.sidebar.radio("Navigation Menu",["Home","Diabetes","Heart Disease"])

#Home Page 

#displays all the available disease prediction options in the web app
if rad == "Home":
    st.title("Medical Predictions App")

    # 4 images side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("Doctor.jpeg", caption="Heart Specialist", use_column_width=True)

    with col2:
        st.image("Soham.jpeg", caption="Psychiatrist", use_column_width=True)

    with col3:
        st.image("Raju.jpeg", caption="Gynaecologist", use_column_width=True)

    # Text info below
    st.text("The Following Disease Predictions Are Available ->")
    st.text("1. Decubitus Ulcer Prediction")
    st.text("2. Diabetes Prediction")
    st.text("3. Heart Disease Prediction")

#Diabetes Prediction

#loading the Diabetes dataset
df2=pd.read_csv("Diabetes Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x2) & target(y2)
x2=df2.iloc[:,[1,4,5,7]].values
x2=np.array(x2)
y2=y2=df2.iloc[:,[-1]].values
y2=np.array(y2)
#performing train-test split on the data
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model2=RandomForestClassifier()
#fitting the model with train data (x2_train & y2_train)
model2.fit(x2_train,y2_train)

#Diabetes Page

#heading over to the Diabetes section
if rad=="Diabetes":
    st.header("Know If You Are Affected By Diabetes")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Glucose (glucose), Insulin (insulin), Body Mass Index-BMI (bmi), Age (age)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    glucose=st.number_input("Enter Your Glucose Level (0-200)",min_value=0,max_value=200,step=1)
    insulin=st.number_input("Enter Your Insulin Level In Body (0-850)",min_value=0,max_value=850,step=1)
    bmi=st.number_input("Enter Your Body Mass Index/BMI Value (0-70)",min_value=0,max_value=70,step=1)
    age=st.number_input("Enter Your Age (20-80)",min_value=20,max_value=80,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction2=model2.predict([[glucose,insulin,bmi,age]])[0]
    
    #prediction part predicts whether the person is affected by Diabetes or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if prediction2==1:
            st.warning("You Might Be Affected By Diabetes")
        elif prediction2==0:
            st.success("You Are Safe")


#Heart Disease Prediction

#loading the Heart Disease dataset
df3=pd.read_csv("Heart Disease Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x3) & target(y3)
x3=df3.iloc[:,[2,3,4,7]].values
x3=np.array(x3)
y3=y3=df3.iloc[:,[-1]].values
y3=np.array(y3)
#performing train-test split on the data
x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y3,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model3=RandomForestClassifier()
#fitting the model with train data (x3_train & y3_train)
model3.fit(x3_train,y3_train)

#Heart Disease Page

#heading over to the Heart Disease section
if rad=="Heart Disease":
    st.header("Know If You Are Affected By Heart Disease")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Chest Pain (chestpain), Blood Pressure-BP (bp), Cholestrol (cholestrol), Maximum HR (maxhr)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    chestpain=st.number_input("Rate Your Chest Pain (1-4)",min_value=1,max_value=4,step=1)
    bp=st.number_input("Enter Your Blood Pressure Rate (95-200)",min_value=95,max_value=200,step=1)
    cholestrol=st.number_input("Enter Your Cholestrol Level Value (125-565)",min_value=125,max_value=565,step=1)
    maxhr=st.number_input("Enter You Maximum Heart Rate (70-200)",min_value=70,max_value=200,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction3=model3.predict([[chestpain,bp,cholestrol,maxhr]])[0]
    
    #prediction part predicts whether the person is affected by Heart Disease or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if str(prediction3)=="Presence":
            st.warning("You Might Be Affected By Diabetes")
        elif str(prediction3)=="Absence":
            st.success("You Are Safe")
