import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


avocado = pd.read_csv("avocado.csv")

avocado.head()
#Average price single avocado

avocado.info()
avocado.corr()


sns.heatmap(avocado.corr(), xticklabels=avocado.columns, yticimport pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


avocado = pd.read_csv("avocado.csv")

avocado.head()
#Average price single avocado

avocado.info()
avocado.corr()


sns.heatmap(avocado.corr(), xticklabels=avocado.columns, yticklabels=avocado.columns)
# Viewing correlation
cols = ['AveragePrice', 'Total Volume','4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']
avocado[cols].head()

avocado[cols].corr()
sns.heatmap(avocado[cols].corr(), xticklabels=avocado[cols].columns, yticklabels=avocado[cols].columns)


x = avocado[cols].drop("Total Volume", axis=1).values
x.shape
y = avocado[cols]["Total Volume"].values
y.shape

#Dataset splitting
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.3, random_state=0)

#Standardizzazione del dataset
ss = StandardScaler()
xTrain = ss.fit_transform(xTrain)
xTest = ss.transform(xTest)

#costruzione del modello con una regressione lineare multipla

ll = LinearRegression()

ll.fit(xTrain, yTrain)
yPred = ll.predict(xTest)


mean_squared_error(yTest, yPred)
r2_score(yTest, yPred)

#Overfitting? Let's see

