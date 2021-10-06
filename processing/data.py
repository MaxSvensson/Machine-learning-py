import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot


# Read the data and split it into x witch holds the matrix and y witch holds the constand variable
dataset = pd.read_csv('./Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# Handeling missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encode country string to numbers
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])] , remainder="passthrough")
x = np.array(ct.fit_transform(x))

le = LabelEncoder()
y = le.fit_transform(y)

# Split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

# Scaling the feature set

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
