import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("archive/StudentsPerformance.csv")

dataset.head()

dataset = pd.get_dummies(dataset, columns=["gender"])

dataset.corr()

dataset = pd.get_dummies(dataset, columns=["race/ethnicity"])



dataset.corr()

import seaborn as sns

sns.heatmap(dataset.corr())


#no correlation between gender and math, reading and writing score
#neither for ethnicity
