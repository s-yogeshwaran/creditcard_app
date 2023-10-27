# import seaborn as sns
# import streamlit as st
# st.set_page_config(page_title="Graphs", page_icon="📈")
# import plotly.express as px
# import plotly.graph_objs as go
# import matplotlib.pyplot as plt
# from plotly.offline import iplot
# import plotly.figure_factory as ff
# import pandas as pd
# import numpy as np

# # from Home import load_data

# # #dataset1 link - https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard1.csv
# # #dataset2 link - https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard2.csv

# df1 = pd.read_csv('https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard1.csv')
# df2 = pd.read_csv('https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard2.csv')
		
# df = pd.concat([df1,df2],ignore_index=True)

# # st.set_page_config(page_title="Graphs", page_icon="📈")
# # st.sidebar.header("📈 Graphs")

# # df = pd.read_csv('https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcardst.csv')
# # df = load_data()
# st.dataframe(df)
# st.write(f'Number of Rows: {df.shape[0]}')
# st.write(f'Number of columns: {df.shape[1]}')

# st.set_option('deprecation.showPyplotGlobalUse', False)

# # ---------------------------------------------------------------------------------------------------------------------------

# # 1. Distribution of legit transactions & fraudulent transactions
# splot = sns.countplot(x = 'Class', data = df)

# for p in splot.patches:
#   splot.annotate(format(p.get_height(), '.0f'),
#    (p.get_x() + p.get_width() / 2., p.get_height()),
#     ha = 'center', va = 'center', xytext = (0, 5),
#     textcoords = 'offset points')

# plt.title('Distribution of Legit transactions & Fraudulent transactions')
# plt.xticks(range(2) , labels=['Legit', 'Fraud'])
# st.pyplot(plt)

# #--------------------------------------------------------------------------------------------------------------------------------

# # st.write(f"Number of duplicates = {df.duplicated().sum()}")
# # df = df.drop_duplicates()

# # # 2. Distribution of legit transactions & fraudulent transactions after removing duplicates
# # splot = sns.countplot(x = 'Class', data = df)

# # for p in splot.patches:
# #   splot.annotate(format(p.get_height(), '.0f'),
# #    (p.get_x() + p.get_width() / 2., p.get_height()),
# #     ha = 'center', va = 'center', xytext = (0, 5),
# #     textcoords = 'offset points')

# # plt.title('Distribution of Legit / Fraudulent transactions - \n After removing duplicates')
# # plt.xticks(range(2) , labels=['Legit', 'Fraud'])
# # st.pyplot(plt)

# #---------------------------------------------------------------------------------------------------------------------------------------------------------

# # 3. % distribution legit transactions & fraudulent transactions
# class_col = df['Class'].value_counts()
# label = ['Legit', 'Fraud']
# quantity = class_col.values

# fig = px.pie(df, values = quantity, names = label, hole = 0.5, title = "% Distribution of Legit / Fraudulent transactions") # - After removing duplicates")
# st.plotly_chart(fig)

# #-----------------------------------------------------------------------------------------------------------------------------------------------------

# # Transactions in time
# class_0 = df.loc[df['Class'] == 0]["Time"]
# class_1 = df.loc[df['Class'] == 1]["Time"]

# hist_data = [class_0, class_1]
# group_labels = ['Legit', 'Fraud']

# fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
# fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
# st.plotly_chart(fig)
# # iplot(fig, filename='dist_only')
# st.write("Fraudulent transactions have a distribution more even than valid transactions - are equaly distributed in time, including the low real transaction times, during night in Europe timezone.")
# st.write('\n')
# #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600))

# tmp = df.groupby(['Hour', 'Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
# df = pd.DataFrame(tmp)
# df.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']

# fig, (ax1, ax2) = plt.subplots(ncols=2)

# s = sns.lineplot(ax = ax1, x="Hour", y="Sum", data=df.loc[df.Class==0])
# s = sns.lineplot(ax = ax2, x="Hour", y="Sum", data=df.loc[df.Class==1], color="red")

# # st.header("Total Amount")
# st.markdown("<h5 style='text-align: center; color: black;'>Total Amount</h5>", unsafe_allow_html=True)

# st.plotly_chart(fig)

# #----------------------------------------------------------------------------------------------------------------------------------------------------------------

# fig, (ax1, ax2) = plt.subplots(ncols=2)

# s = sns.lineplot(ax = ax1, x="Hour", y="Transactions", data=df.loc[df.Class==0])
# s = sns.lineplot(ax = ax2, x="Hour", y="Transactions", data=df.loc[df.Class==1], color="red")

# # plt.title("Total Number of Transactions", fontsize = 18)
# st.markdown("<h5 style='text-align: center; color: black;'>Total Number of Transactions</h5>", unsafe_allow_html=True)
# st.plotly_chart(fig)

# #---------------------------------------------------------------------------------------------------------------------------------------------------------

# fig, (ax1, ax2) = plt.subplots(ncols=2)

# s = sns.lineplot(ax = ax1, x="Hour", y="Mean", data=df.loc[df.Class==0])
# s = sns.lineplot(ax = ax2, x="Hour", y="Mean", data=df.loc[df.Class==1], color="red")

# # plt.title("Average Amount of Transactions", fontsize = 18)
# st.markdown("<h5 style='text-align: center; color: black;'>Average Amount of Transactions</h5>", unsafe_allow_html=True)
# st.plotly_chart(fig)

# #-----------------------------------------------------------------------------------------------------------------------------------

# fig, (ax1, ax2) = plt.subplots(ncols=2)

# s = sns.lineplot(ax = ax1, x="Hour", y="Max", data=df.loc[df.Class==0])
# s = sns.lineplot(ax = ax2, x="Hour", y="Max", data=df.loc[df.Class==1], color="red")

# # plt.title("Maximum Amount of Transactions", fontsize = 18)
# st.markdown("<h5 style='text-align: center; color: black;'>Maximum Amount of Transactions</h5>", unsafe_allow_html=True)
# st.plotly_chart(fig)

# #----------------------------------------------------------------------------------------------------------------------------------------

# fig, (ax1, ax2) = plt.subplots(ncols=2)

# s = sns.lineplot(ax = ax1, x="Hour", y="Median", data=df.loc[df.Class==0])
# s = sns.lineplot(ax = ax2, x="Hour", y="Median", data=df.loc[df.Class==1], color="red")

# # plt.suptitle("Median Amount of Transactions", fontsize = 18)
# st.markdown("<h5 style='text-align: center; color: black;'>Median Amount of Transactions</h5>", unsafe_allow_html=True)
# plt.show()

# #-----------------------------------------------------------------------------------------------------------------------------------------------------

# fig, (ax1, ax2) = plt.subplots(ncols=2)

# s = sns.lineplot(ax = ax1, x="Hour", y="Min", data=df.loc[df.Class==0])
# s = sns.lineplot(ax = ax2, x="Hour", y="Min", data=df.loc[df.Class==1], color="red")

# # plt.suptitle("Minimum Amount of Transactions", fontsize = 18)
# st.markdown("<h5 style='text-align: center; color: black;'>Minimum Amount of Transactions</h5>", unsafe_allow_html=True)
# plt.show()

# #------------------------------------------------------------------------------------------------------------------------------------

# #Transactions amount
# fig, (ax1, ax2) = plt.subplots(ncols=2)

# s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=True)
# s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=df, palette="PRGn",showfliers=False)
# st.markdown("<h5 style='text-align: center; color: black;'>Transactions amount</h5>", unsafe_allow_html=True)
# plt.show()

# #----------------------------------------------------------------------------------------------------------------------












# # #dataset1 link - https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard1.csv
# # #dataset2 link - https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard2.csv

# # #importing the libraries
# # import numpy as np
# # import pandas as pd
# # import seaborn as sns
# # import streamlit as st
# # import matplotlib.pyplot as plt
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.svm import SVC
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score
# # from sklearn.metrics import precision_score,recall_score,f1_score, classification_report

# # #To avoid warnings
# # import warnings
# # warnings.filterwarnings('ignore')

# # @st.cache()
# # def load_data():
# # 	# loading the dataset to a Pandas DataFrame
# # 	df1 = pd.read_csv('https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard1.csv')
# # 	df2 = pd.read_csv('https://raw.githubusercontent.com/s-yogeshwaran/creditcard_app/main/creditcard2.csv')
		
# # 	df = pd.concat([df1,df2],ignore_index=True)
		
# # 	#Feature Scaling - Normalize
# # 	sc = StandardScaler()
# # 	df['Amount']=sc.fit_transform(pd.DataFrame(df['Amount']))
		
# # 	#Droping the time column
# # 	data = df.drop(['Time'],axis=1)
		
# # 	#Removing the duplicate values
# # 	new_df = data.drop_duplicates()
		
# # 	# separating the data for analysis
# # 	legit = new_df[new_df.Class == 0]
# # 	fraud = new_df[new_df.Class == 1]
		
# # 	#handling Imbalanced Dataset
# # 	legit_sample=legit.sample(n=473)
		
# # 	#creating new dataframe
# # 	new_df = pd.concat([legit_sample,fraud],ignore_index=True)
# # 	return new_df

# # new_df = load_data()

# # #feature variable and target variable
# # x = new_df.drop('Class',axis=1)  # axis=1 meansfull column will be dropped and axis = 0 will drop a row
# # y = new_df['Class']
	
# # #spliting train and test set after balancing
# # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,
# # 	                                                 random_state=42)
	
# # #Logistic Regression
# # log = LogisticRegression()
# # log.fit(x_train,y_train) # training the dataset using fit function ()
	
# # #Decision Tree Classifier
# # dt = DecisionTreeClassifier()
# # dt.fit(x_train,y_train)
	
# # #Random Forest Classifier
# # rf = RandomForestClassifier()
# # rf.fit(x_train,y_train)
	
# # # Creating the SVC model 
# # svc = SVC(kernel = 'linear')
# # svc.fit(x_train, y_train)

# # @st.cache()
# # def prediction(model):
# # 	predict = model.predict(x_test)
# # 	return predict

# # st.title("Credit Card Fraud Detection App")
# # #st.sidebar.title("Credit Card Fraud Detection App")
# # #st.sidebar.write('## Machine Learning Algorithms')
# # st.sidebar.write('Menu')
# # if st.sidebar.button('Machine Learning Algorithm'):
	
# # 	algo = st.radio(
# # 		'Machine Learning Algorithms', 
# # 		['Supervised Learning Algorithms', 'Unsupervised Learning Algorithms'], 
# # 		index = None,
# # 	)

# # 	# genre = st.radio(
# #  #    	"What's your favorite movie genre",
# #  #    	[":rainbow[Comedy]", "***Drama***", "Documentary :movie_camera:"],
# #  #    	index=None,
# # 	# )
	
# # 	if algo == 'Supervised Learning Algorithms':
# # 		classifier = st.radio('Supervised Learning Algorithms', ['Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier'])
		
# # 		if st.sidebar.button('Predict'):
# # 			if classifier == 'Support Vector Machine':
# # 				y_pred = prediction(svc)
# # 				score = svc.score(x_train, y_train)
# # 				#st.write(f"{' '*19}Support Vector Machine\n")
# # 				#st.write(classification_report(y_test, y_pred))
			        
		
# # 			elif classifier == 'Logistic Regression':
# # 				predict = prediction(log)
# # 				score = log.score(x_train, y_train)
# # 				#print(f"{' '*18}Logistic Regression Report\n")
# # 				#print(classification_report(y_test, predict))
			
# # 			elif classifier == 'Random Forest Classifier':
# # 				predict = prediction(rf)
# # 				score = rf.score(x_train, y_train)
# # 				#print(f"{' '*16}Random Forest Classifier Report\n")
# # 				#print(classification_report(y_test, predict))
		
# # 			else:
# # 				predict = prediction(dt)
# # 				score = dt.score(x_train, y_train)
# # 				#print(f"{' '*19}Decision Tree Classifier\n")
# # 				#print(classification_report(y_test, predict))
		
# # 			st.write(f"accuracy score of {classifier} = {score:.4f}")

# # if st.sidebar.button('Deep Learning'):
# # 	classifier = st.sidebar.radio('Deep Learning', ['Model 1', 'Model 2', 'Model 3'])
# # #classifier = st.sidebar.radio('Supervised Learning Algorithms', ['Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier'])
# # #classifier = st.sidebar.selectbox("Classifier", ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier'))

		
