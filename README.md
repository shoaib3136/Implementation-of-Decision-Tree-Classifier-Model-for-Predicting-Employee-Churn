# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the packages that helps to implement Decision Tree.
2. Download and upload required csv file or dataset for predecting Employee Churn
3. Initialize variables with required features.
4. And implement Decision tree classifier to predict Employee Churn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Shaik Shoaib Nawaz
RegisterNumber:212222240094
*/
import pandas as pd
data=pd.read_csv('/content/Employee_EX6.csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
i.) Dataset:
![image](https://github.com/shoaib3136/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117919362/79db3821-3ded-4bac-a9ef-96f894e06e98)


ii.) Accuracy Score:
![image](https://github.com/shoaib3136/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117919362/3c13684e-e73a-49ab-925c-61c010198b7e)


iii.) Testing of model:
![image](https://github.com/shoaib3136/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117919362/f94faa31-8c90-45f7-bfdb-489896be15bb)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
