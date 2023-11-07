# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.. Import the standard libraries. 

2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated()
function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules
from sklearn.

7.Apply new unknown values


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Nandhini S
RegisterNumber:  212222220028


import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion 

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:

Placement Data:
![Screenshot 2023-11-07 210148](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/0343d9df-33e2-4ec2-ae38-f2bf4d95ddaf)

salary data:
![Screenshot 2023-11-07 210158](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/0117f208-f8d0-452e-97f1-03b2600397b0)

Checking the null() function:
![Screenshot 2023-11-07 210206](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/6cfd9f72-e337-4e4e-9327-6d0fd00f2eda)

Data Duplicate:
![Screenshot 2023-11-07 210213](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/1b279bb0-28a1-4ef7-a4f1-db31a509e77e)

Print data:

![Screenshot 2023-11-07 210220](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/c943d4a5-3d07-47ef-9e37-9ee37560ec96)

Data status:

![Screenshot 2023-11-07 210228](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/fdcbdd3e-a77b-4232-89ca-f8c1b3e10d78)

Y prediction array:
![Screenshot 2023-11-07 210237](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/ff9626ea-a525-4f03-bdaa-bbba6d0a58e9)

accuracy value:
![Screenshot 2023-11-07 210244](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/7e85cae4-3319-417f-a9e4-aa73f90fd0a2)

confusion array:
![Screenshot 2023-11-07 210250](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/482a0381-f5b9-4d0d-8269-c538c10cc757)

Classification report:
![Screenshot 2023-11-07 210256](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/5f1f9016-21d6-4176-9956-8ef8bc27b1ce)

Prediction of LR:

![Screenshot 2023-11-07 210305](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/011493e2-c0c7-4787-ad42-4016707b96e8)

 ## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
