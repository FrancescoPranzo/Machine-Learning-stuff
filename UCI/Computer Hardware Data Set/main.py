import pandas as pd
import numpy as np


import pandas as pd
import numpy as np


machine = pd.read_csv("machine.data", names=["vendor name", "Model Name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"])

machine.info()

#see Correlation between data
print(machine.corr())

x = machine[["MMIN", "MMAX"]].values
y = machine["ERP"].values

x.shape
y.shape


from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.3, random_state=0)

#creating the model
from sklearn.linear_model import LinearRegression

ll = LinearRegression()

ll.fit(xTrain, yTrain)

yPred = ll.predict(xTest)

#valutate the model
from sklearn.metrics import mean_squared_error, r2_score

print(mean_squared_error(yTest, yPred ))
r2_score(yTest, yPred
