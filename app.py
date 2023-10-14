#dataset1 link - https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard1.csv
#dataset2 link - https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard2.csv

#importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score, classification_report

#To avoid warnings
import warnings
warnings.filterwarnings('ignore')

@st.cache()
def load_data():
	# loading the dataset to a Pandas DataFrame
	df1 = pd.read_csv('https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard1.csv')
	df2 = pd.read_csv('https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard2.csv')
	
	df = pd.concat([df1,df2],ignore_index=True)
	
	#Feature Scaling - Normalize
	sc = StandardScaler()
	df['Amount']=sc.fit_transform(pd.DataFrame(df['Amount']))
	
	#Droping the time column
	data = df.drop(['Time'],axis=1)
	
	#Removing the duplicate values
	new_df = data.drop_duplicates()
	
	# separating the data for analysis
	legit = new_df[new_df.Class == 0]
	fraud = new_df[new_df.Class == 1]
	
	#handling Imbalanced Dataset
	legit_sample=legit.sample(n=473)
	
	#creating new dataframe
	new_df = pd.concat([legit_sample,fraud],ignore_index=True)
	
	#feature variable and target variable
	x = new_df.drop('Class',axis=1)  # axis=1 meansfull column will be dropped and axis = 0 will drop a row
	y = new_df['Class']
	
	#spliting train and test set after balancing
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,
	                                                 random_state=42)
	
	#Logistic Regression
	log = LogisticRegression()
	log.fit(x_train,y_train) # training the dataset using fit function ()
	
	#Decision Tree Classifier
	dt = DecisionTreeClassifier()
	dt.fit(x_train,y_train)
	
	#Random Forest Classifier
	rf = RandomForestClassifier()
	rf.fit(x_train,y_train)
	
	# Creating the SVC model 
	svc = SVC(kernel = 'linear')
	svc.fit(x_train, y_train)

load_data()

@st.cache()
def prediction(model):
	predict = model.predict(x_test)
	return predict

#st.title("Credit Card Fraud Detection App")
st.sidebar.title("Credit Card Fraud Detection App")
classifier = st.sidebar.selectbox("Classifier", ('Support Vector Machine', 
	'Logistic Regression', 'Random Forest Classifier', 
	'Decision Tree Classifier'))

if st.sidebar.button('Predict'):
	if classifier == 'Support Vector Machine':
		y_pred = prediction(svc)
		score = svc.score(x_train, y_train)
		#st.write(f"{' '*19}Support Vector Machine\n")
		#st.write(classification_report(y_test, y_pred))
	        

	elif classifier == 'Logistic Regression':
		predict = prediction(log)
		score = log.score(x_train, y_train)
		#print(f"{' '*18}Logistic Regression Report\n")
		#print(classification_report(y_test, predict))
	
	elif classifier == 'Random Forest Classifier':
		predict = prediction(rf)
		score = rf.score(x_train, y_train)
		#print(f"{' '*16}Random Forest Classifier Report\n")
		#print(classification_report(y_test, predict))

	else:
		predict = prediction(dt)
		score = dt.score(x_train, y_train)
		#print(f"{' '*19}Decision Tree Classifier\n")
		#print(classification_report(y_test, predict))

	st.write(f"accuracy score of {classifier} = {score:.4f}")
