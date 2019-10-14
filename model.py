import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json

dataset = pd.read_csv('sal.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 1/3, random_state=0)
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_predict = regressor.predict(X_test)
pickle.dump(regressor, open('model.pkl','wb'))