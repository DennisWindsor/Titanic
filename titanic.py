import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle


# import data and preprocess
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Shuffle data
train_data = shuffle(train_data)

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
X_cols = ["Pclass", "Sex",  "SibSp", "Parch", "Fare"]
y_col = "Survived"
X_train = train_data[X_cols]
y_train = np.array(train_data[y_col])
X_train = SelectKBest(chi2, k=4).fit_transform(X_train, y_train)
X_train = normalize(X_train)

kf = KFold(n_splits=5)

models = [(svm.SVC(probability=True), "SVM"), (MLPClassifier(solver='lbfgs', alpha=1e-5,
          hidden_layer_sizes=(5, 5), random_state=1), "Neural Net"), 
          (SGDClassifier(loss="hinge", penalty="l2"), "SGD"),
          (KNeighborsClassifier(n_neighbors=5),"KNN"),
          (tree.DecisionTreeClassifier(), "Decision Tree"),
           (RandomForestClassifier(max_depth=7, random_state=0), "Random Forest")]
avg_acc = []

# Run k-fold cross validation on different models(classifiers)
for model in models:
    print(model[1])
    acc = []
    for train, test in kf.split(train_data):
        X = X_train[train]
        y = y_train[train]
        X_valid = X_train[test]
        y_valid = y_train[test]
        model[0].fit(X, y)
        predict= model[0].predict(X_valid)
        print(accuracy_score(y_valid, predict))
        acc.append(accuracy_score(y_valid, predict))
    print("Average: {}".format(np.mean(acc)))
    avg_acc.append(np.mean(acc))

# Test different neural network models
for i in range(1, 11):
    for j in range(1, 11):
        model = MLPClassifier(solver='lbfgs', alpha=1e-5,
          hidden_layer_sizes=(i, j), random_state=1)
        acc = []
        for train, test in kf.split(train_data):
            X = X_train[train]
            y = y_train[train]
            X_valid = X_train[test]
            y_valid = y_train[test]
            model.fit(X, y)
            predict= model.predict(X_valid)
            acc.append(accuracy_score(y_valid, predict))
        print("Hidden: {} {} Average: {}".format(i,j,np.mean(acc)))
# Optinal around 5, 5

# Test different random forest models
for i in range(1, 20):
    model = RandomForestClassifier(max_depth=i, random_state=0)
    acc = []
    for train, test in kf.split(train_data):
        X = X_train[train]
        y = y_train[train]
        X_valid = X_train[test]
        y_valid = y_train[test]
        model.fit(X, y)
        predict= model.predict(X_valid)
        acc.append(accuracy_score(y_valid, predict))
    print("max_depth: {} Average: {}".format(i,np.mean(acc)))