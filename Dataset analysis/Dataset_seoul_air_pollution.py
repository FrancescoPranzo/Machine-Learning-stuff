import pandas as pd
import numpy as np
#Analisi compiuta sul seguente dataset: https://www.kaggle.com/bappekim/air-pollution-in-seoul/data#
#Vedere la correlazione tra SO2 e O3
dataset = pd.read_csv("Measurement_summary.csv")
dataset.head()
dataset.shape
dataset.info()

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


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

from sklearn.linear_model import LinearRegression

ll = LinearRegression()
ll.fit(xTrain, yTrain)
yPred = ll.predict(xTest)

from sklearn.metrics import r2_score

r2_score(yTest, yPred)
