import streamlit as st
import numpy as np
import pandas as pd
import numpy as np 
import pickle

file = 'DTree_model.sav'
# load the model from disk
loaded_model = pickle.load(open(file, 'rb'))


st.write("""# This is the Customer Churn Predictor to predict whether the customer is ggoing to churn or not""")

def user_input_parameters():
    Monthly_Bill=st.number_input("Enter the amount of the Monthly Bill",min_value=0.0,max_value=150.0)
    Total_Usage_GB=st.number_input("Enter the Total GB Data Used",min_value=0.0,max_value=1000.0)
    
    data={'Monthly_Bill':Monthly_Bill,
          'Total_Usage_GB':Total_Usage_GB}
    
    features=pd.DataFrame(data,index=[0])
    return features

uip=user_input_parameters().values
st.subheader("User Input Parameters")
st.write(uip)
prediction=loaded_model.predict(uip)


if prediction>0.5:
    st.write("The Customer is going to Churn")
else:
    st.write("The Customer is not going to Churn")


