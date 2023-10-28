import streamlit as st
import pickle
import pandas as pd


# set web page
st.set_page_config(page_title="DV Credit Card Fraud Detection",
                   page_icon="🕵🏼‍♀️", #💳
                   initial_sidebar_state="expanded",
                   layout= "wide")

# html_temp = """
# <div style=" text-align:center;">
# <h4 style=" font-size: 20px;">SRM Internship Project</h4>
# </div>
# """
# st.sidebar.markdown(html_temp, unsafe_allow_html=True)


# # title of the body
# html_temp = """
# <div style="-moz-box-shadow: 1px 1px 3px 2px red;
#   -webkit-box-shadow: 1px 1px 3px 2px red;
#   box-shadow: 1px 1px 3px 2px red;
#   padding: 10px;
#   font-size: 30px;
#   font-weight: bold;
#   text-align: center;
#   font-family: Helvetica
# ">
# SRM Internship Project <br>
# Credit Card Fraud Detection
# </div>"""
# st.markdown(html_temp, unsafe_allow_html=True)

st.title("<h1 style='text-align: center;'>SRM Internship Project</h1>")
st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)
st.header('Credit Card Fraud Detection')


# # title of the sidebar
# html_temp = """
# <div style="background-color:red">
# <p style="color:white;
#             text-align:center;
#             font-size: 17px;
#             ">
# Select Your Data </p>
# </div>"""
# st.sidebar.markdown(html_temp,unsafe_allow_html=True)

df1 = pd.read_csv('https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard1.csv')
df2 = pd.read_csv('https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard2.csv')
	
df = pd.concat([df1,df2],ignore_index=True)

st.markdown("[Click here](https://colab.research.google.com/drive/1U7xiHFdXa3zkflVrITq8MCBd6FOFbKHA?usp=sharing) to see the colab notebook.")

if st.checkbox('show dataset'):
  st.dataframe(df)
  st.write(f"Number of rows = {df.shape[0]}")
  st.write(f"Number of columns = {df.shape[1]}")
  
if st.checkbox('Show related plots'):
  st.image('distribution 1.png')#, caption = "Distribution of Legit transcations & Fraudulent transcation")
  st.image('distribution 2.png')
  st.image('% distribution.png')
  st.image('time density.png')
  st.image('total amount.png')
  st.image('tot no.of transca.png')
  st.image('avg amt of transc.png')
  st.image('max amt of transc.png')
  st.image('median amt of transc.png')
  st.image('min amt of transc.png')
  st.image('transca of amt.png')
  st.image('amt of fraud transc.png')
	

