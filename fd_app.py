import streamlit as st
import pickle
import pandas as pd


# set web page
st.set_page_config(page_title="WONUJIRO Credit Fraud Prediction",
                   page_icon="📇",
                   initial_sidebar_state="expanded",
                   layout= "wide")

html_temp = """
<div style=" text-align:center;">
<h4 style=" font-size: 20px;">WONUJIRO PROJECT 3</h4>
</div>
"""
st.sidebar.markdown(html_temp, unsafe_allow_html=True)


# title of the body
html_temp = """
<div style="-moz-box-shadow: 1px 1px 3px 2px #08c7b4;
  -webkit-box-shadow: 1px 1px 3px 2px #08c7b4;
  box-shadow: 1px 1px 3px 2px #08c7b4;
  padding: 10px;
  font-size: 30px;
  font-weight: bold;
  text-align: center;
  font-family: Helvetica
">
Credit Fraud Prediction
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)


# title of the sidebar
html_temp = """
<div style="background-color:#08c7b4"">
<p style="color:white;
            text-align:center;
            font-size: 17px;
            ">
Select Your Data </p>
</div>"""
st.sidebar.markdown(html_temp,unsafe_allow_html=True)



st.markdown("<div style='text-align: center; padding-top: 40px; color: black;'><h5>Select Your Model</h5></div>", unsafe_allow_html=True)
selection = st.selectbox("", ["Logistic Regression"])

st.write(selection , "model is selected.")
model = pickle.load(open ('LR_deploy_model.pkl', 'rb'))


# Collect user input
V5 = st.sidebar.slider(label="V5", min_value= -3.40, max_value= 3.40, step= 0.01)
V9 = st.sidebar.slider(label="V9", min_value= -2.30, max_value= 2.30, step= 0.01)
V10 = st.sidebar.slider(label="V10", min_value= -1.45, max_value= 1.45, step= 0.01)
V11 = st.sidebar.slider(label="V11", min_value= -3.35, max_value= 3.35, step= 0.01)
V13 = st.sidebar.slider(label="V13", min_value= -4.45, max_value= 4.45, step= 0.01)
V15 = st.sidebar.slider(label="V15", min_value= -3.10, max_value= 3.10, step= 0.01)
V18 = st.sidebar.slider(label="V18", min_value= -4.85, max_value= 4.85, step= 0.01)
V21 = st.sidebar.slider(label="V21", min_value= -1.00, max_value= 1.00, step= 0.01)
V23 = st.sidebar.slider(label="V23", min_value= -1.00, max_value= 1.00, step= 0.01)
V24 = st.sidebar.slider(label="V24", min_value= -1.65, max_value= 1.65, step= 0.01)
V28 = st.sidebar.slider(label="V28", min_value= -1.00, max_value= 1.00, step= 0.01)

# converting user inputs into dictionary format
col_dict = { "V5" : V5,            
             "V9" : V9,
             "V10" : V10,
             "V11" : V11,
             "V13" : V13,
             "V15" : V15,
             "V18" : V18,
             "V21" : V21,
             "V23" : V23,
             "V24" : V24,
             "V28" : V28
             }


df_col = pd.DataFrame.from_dict([col_dict])
user_inputs = df_col

# Decfne predictions
prediction = model.predict(user_inputs)

st.markdown("<h5 style = 'text-align: center; color: Black;'> Your Transaction Information </h5>", unsafe_allow_html=True
                            )
st.table(user_inputs)



# Make predictions
if st.button('Predict'):
    predicted_proba = model.predict_proba(user_inputs)[0]
    if prediction[0] == 0:
        st.success("✅ The Transaction is SAFE")
        st.info(f" Probability that the transaction is secure :   {predicted_proba[0]*100:.2f}% ")
    elif prediction[0] == 1:
        st.warning("🚨 The Transaction might be FRAUDULENT!")
        st.info(f"Probability that the transaction is fraudulent :  {predicted_proba[1]*100:.2f}% ")
    else:
        st.error("Fail to calculate.")
