import pandas as pd
import numpy as np


machine = pd.read_csv("machine.data", names=["vendor name", "Model Name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"])
machine.head()

machine.head()
machine.info()

#see Correlation between data
machine.corr()

x = machine.drop("ERP", axis=1).values
y = machine["ERP"].values

x.shape
y.shape
