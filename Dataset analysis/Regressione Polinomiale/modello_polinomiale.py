#Analysis on the Seoul air pollution dataset, but with a polynomial regression
#dataset credit: https://www.kaggle.com/bappekim/air-pollution-in-seoul/data#
#A simple exercise

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
dataset = pd.read_csv("Measurement_summary.csv")
dataset.head()
#dataset.shape
#dataset.info()

dataset.corr()
import seaborn as sns
sns.heatmap(dataset.corr())

#Correlazione tra NO2 e O3
cols = ["NO2","O3","SO2","CO"]
sns.heatmap(dataset[cols].corr())

#la correlazione di dimostra evidente
#SO2 e O3
sns.pairplot(dataset[cols])

x = dataset[["SO2"]]
x.shape
y = dataset["O3"]
y.shape

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
xTrain.shape
xTest.shape


pf = PolynomialFeatures(degree=2)
xTrain = pf.fit_transform(xTrain)
xTrainPoly.shape
xTest = pf.transform(xTest)


ll = LinearRegression()
ll.fit(xTrain, yTrain)
yPred = ll.predict(xTest)

mse_score = mean_squared_error(yTest, yPred)
r2 = r2_score(yTest, yPred)

print(mse_score)
print(r2)
