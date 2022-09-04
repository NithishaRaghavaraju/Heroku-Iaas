import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset=pd.read_csv("50_Startups.csv")
dataset["State"].unique()
encoding = {}
for i,j in enumerate(dataset["State"].unique()):
  encoding[j] = i
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

for i in range(len(x[:,3])):
  x[:,3][i] = encoding[x[:,3][i]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)

pickle.dump(regressor, open("model.pkl","wb"))
model = pickle.load(open("model.pkl","rb"))
print(model.predict([[153441.51, 101145.55, 407934.54, 2]]))

