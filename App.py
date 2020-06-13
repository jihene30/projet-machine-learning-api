from flask import Flask
import pandas as pd
import os
import pandas as pd 
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
import pickle
app = Flask(__name__)

@app.route('/predict', methods=['POST','GET'])
def predict():
	if flask.request.method =="POST":
		
	features = pd.DataFrame(request.args.get('features'))
	assert features.isna() == False
	model= pickle.load(open("model.pkl",'rb'))

	result = model.predict(features)
    return str(result)
if __name__ == '__main__':
    	app.run()   








dataset_Train = pd.read_csv(r"C:\Users\ASUS\Desktop\diabetes-data\diabetes.csv") 
import pandas as pd 
dataset_Test = pd.read_csv(r"C:\Users\ASUS\Desktop\diabetes-data\diabetes.csv") 
dataset_Test.drop('Outcome', axis=1, inplace=True)
dataset_Train['Pregnancies'] = dataset_Train['Pregnancies']/17
dataset_Train['Glucose'] = dataset_Train['Glucose']/199
dataset_Train['BloodPressure'] = dataset_Train['BloodPressure']/122
dataset_Train['SkinThickness'] = dataset_Train['SkinThickness']/99
dataset_Train['Insulin'] = dataset_Train['Insulin']/849
dataset_Train['BMI'] = dataset_Train['BMI']/67.10
dataset_Train['DiabetesPedigreeFunction'] = dataset_Train['DiabetesPedigreeFunction']/2.42
dataset_Train['Age'] = dataset_Train['Age']/81
dataset_Train['Outcome'] = dataset_Train['Outcome']/1.00



dataset_Test['Pregnancies'] = dataset_Test['Pregnancies']/17
dataset_Test['Glucose'] = dataset_Test['Glucose']/199
dataset_Test['BloodPressure'] = dataset_Test['BloodPressure']/122
dataset_Test['SkinThickness'] = dataset_Test['SkinThickness']/99
dataset_Test['Insulin'] = dataset_Test['Insulin']/849
dataset_Test['BMI'] = dataset_Test['BMI']/67.10
dataset_Test['DiabetesPedigreeFunction'] = dataset_Test['DiabetesPedigreeFunction']/2.42
dataset_Test['Age'] = dataset_Test['Age']/81

dataset_Train['Glucose']= dataset_Train['Glucose'].astype(int)
X=dataset_Train.loc[:, dataset_Train.columns != 'Outcome']
y=dataset_Train['Outcome'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

 
sc_x = StandardScaler() 
X_train = sc_x.fit_transform(X_train)  
X_test = sc_x.transform(X_test) 
  
print (X_train[0:10, :]) 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train) 

prediction = classifier.predict(X_test)


print ("Accuracy : ", accuracy_score(y_test, prediction))
prediction = classifier.predict(dataset_Test)