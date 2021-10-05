import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot

dataset = pd.read_csv('./processing/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])] , remainder="passthrough")
x = np.array(ct.fit_transform(x))

le = LabelEncoder()
y = le.fit_transform(y)

pyplot.plot(x,y)