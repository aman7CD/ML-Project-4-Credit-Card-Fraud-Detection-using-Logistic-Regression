## Importing the Dependecies 

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score




## Data Colection and Preprocessing
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")




## This Dataset is highly unblanced

# 0 --> Normal Transaction
# 1 --> fraudulent transaction

legit = data[data.Class == 0]
fraud = data[data.Class == 1]

legit_sample = legit.sample(n=500)

data1 = pd.concat([legit_sample,fraud], axis=0)




## Splitting the Data
x = data.drop(["Class"],axis=1)
y = data["Class"]

xtn,xtt,ytn,ytt = train_test_split(x,y, test_size=0.1, random_state=2, )




## Training The Model
model = LogisticRegression(max_iter=1000,random_state=1)
model.fit(xtn,ytn)




## Model Evaluation Through r2 Score and MSE

y_pred = model.predict(xtt)

ascore = accuracy_score(ytt,y_pred)
ascore

print(f"The ascore of LogisticRegression is {ascore} ")
