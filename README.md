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
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset = pd.read_csv('Placement_Data_Full_Class.csv')

dataset
dataset.head(20)
dataset.tail(20)

#droping tha serial no salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('salary',axis=1)
dataset = dataset.drop('gender',axis=1)
dataset = dataset.drop('ssc_b',axis=1)
dataset = dataset.drop('hsc_b',axis=1)
dataset
dataset.shape
dataset.info()

#catgorising col for further labelling
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset.info()

dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
dataset.info()
dataset

#selecting the features and labels
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)
dataset.head()

x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

clf.predict([[0, 87, 0, 95, 0, 2, 78, 2, 0]])
*/
```

## Output:
DATASET:
![Screenshot 2023-09-26 205003](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/50ca3067-fa48-4ad8-9a81-bdc91f74c9b7)

dataset.head():
![Screenshot 2023-09-26 205017](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/c9940147-0947-4e63-b6f1-73f092899505)

dataset.tail():
![Screenshot 2023-09-26 205029](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/09f7655d-67c6-4b97-82c3-1c7d79c55a60)

dataset after dropping:

![Screenshot 2023-09-26 205042](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/9b043f92-b1a6-4e5e-a27a-931cee0b82b7)
![Screenshot 2023-09-26 205054](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/01dbebbe-0789-49e4-b7b7-991633391720)

datase.shape:
![Screenshot 2023-09-26 205110](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/33a03b72-8ebd-417a-97e2-391d3958e747)

dataset.info()
![Screenshot 2023-09-26 205116](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/d2c0fa5d-9f97-4dd9-be65-d3ec7c55273a)

dataset.dtypes:
![Screenshot 2023-09-26 205121](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/764b8889-41b9-4041-a6e5-e112212b2514)

dataset.info():
![Screenshot 2023-09-26 205135](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/2da8decf-3e37-46f1-85f7-f2b6681a3763)

dataset.codes:
![Screenshot 2023-09-26 205145](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/f46940e1-11c6-4381-b561-d71a1f6a5cfe)

selecting the features and labels:
![Screenshot 2023-09-26 205151](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/5718ffab-7a4a-4ec6-90bb-ae008d5b4021)

dataset.head():
![Screenshot 2023-09-26 205209](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/39f5f733-b0a7-479d-9513-ab7672db7aae)

x_train.shape:
![Screenshot 2023-09-26 205214](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/d6176757-3261-4c16-9dd6-9cc7c7944da1)

x_test.shape:
![Screenshot 2023-09-26 205218](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/3d6c6947-30d9-4b53-9769-87aba373577b)

y_train.shape:
![Screenshot 2023-09-26 205233](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/aa7e945b-ff5e-4f9f-9c49-b6c5d2479624)

y_test.shape:
![Screenshot 2023-09-26 205233](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/c02e568c-2181-4a89-9f78-7b35eb063a1f)

clf.predict:
![Screenshot 2023-09-26 205243](https://github.com/nandhu6523/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123856724/3fe141d5-a582-4a33-80e4-2b4be29abab0)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
