import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn import svm

# import data and preprocess
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Remove irrelevant columns
cols_to_delete = ['Embarked', 'Cabin', 'Name', 'Ticket']
train_data = train_data.drop(cols_to_delete, 1)
test_data = test_data.drop(cols_to_delete, 1)

# Use median age 
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(train_data['Age'].median())

# Make sex numeric
mask = train_data["Sex"] == "male"
train_data.loc[mask, "Sex"] = 1
mask = train_data["Sex"] == "female"
train_data.loc[mask, "Sex"] = 0

mask = test_data["Sex"] == "male"
test_data.loc[mask, "Sex"] = 1
mask = test_data["Sex"] == "female"
test_data.loc[mask, "Sex"] = 0

# Normalize
train_data

# Split into X and y
X_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
y_col = "Survived"
X_train = train_data[X_cols]
X_train = normalize(X_train)
y_train = np.array(train_data[y_col])

kf = KFold(n_splits=5)

print("SVN")
for train, test in kf.split(train_data):
    X = X_train[train]
    y = y_train[train]
    X_valid = X_train[test]
    y_valid = y_train[test]
    model = svm.SVC()
    model.fit(X, y)
    predict= model.predict(X_valid)
    print(accuracy_score(y_valid, predict))