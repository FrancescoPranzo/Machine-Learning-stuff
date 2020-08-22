import pandas as pd
import numpy as np
#
dataset = pd.read_csv("archive/StudentsPerformance.csv")
#
dataset.head()
#one-hot-encoding of parental level of education
dataset = pd.get_dummies(dataset, columns=["parental level of education"])
# #one-hot-encoding of lunch
dataset = pd.get_dummies(dataset, columns=["lunch"])
# #one-hot-encoding of test preparation course
dataset = pd.get_dummies(dataset, columns=["test preparation course"])

#correlation between data
dataset.corr()
import seaborn as sns
sns.heatmap(dataset.corr())

#the major correlation are between math score, writing score, reading score

#using multiple linear regression to build a model
x = dataset[["reading score", "writing score"]].values
#x.shape

y = dataset["math score"].values
#y.shape

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.3)

from sklearn.linear_model import LinearRegression

ll = LinearRegression()

ll.fit(xTrain, yTrain)

yPred = ll.predict(xTest)

#valutate the model
from sklearn.metrics import r2_score

r2_score(yTest, yPred)
#r2_score = 0.6747618018275282 
#not bad
