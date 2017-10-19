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
from sklearn.preprocessing import StandardScaler

def run_k_fold_on_model(model, X_train, y_train):
    kf = KFold(n_splits=5)
    acc = []
    for train, test in kf.split(X_train):
        X = X_train[train]
        y = y_train[train]
        X_valid = X_train[test]
        y_valid = y_train[test]
        model.fit(X, y)
        predict= model.predict(X_valid)
        print(accuracy_score(y_valid, predict))
        acc.append(accuracy_score(y_valid, predict))
    print("Average: {}".format(np.mean(acc)))


models = [(svm.SVC(probability=True), "SVM"), (MLPClassifier(solver='lbfgs', alpha=1e-5,
          hidden_layer_sizes=(5, 5), random_state=1), "Neural Net"), 
          (SGDClassifier(loss="hinge", penalty="l2"), "SGD"),
          (KNeighborsClassifier(n_neighbors=5),"KNN"),
          (tree.DecisionTreeClassifier(), "Decision Tree"),
           (RandomForestClassifier(max_depth=7, n_estimators=100, n_jobs=-1), "Random Forest")]

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
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(train_data['Age'].mean())

# Set nan age to 0
#train_data['Age'] = train_data['Age'].fillna(0)
#test_data['Age'] = test_data['Age'].fillna(0)

# Make sex numeric
mask = train_data["Sex"] == "male"
train_data.loc[mask, "Sex"] = 1
mask = train_data["Sex"] == "female"
train_data.loc[mask, "Sex"] = 0

mask = test_data["Sex"] == "male"
test_data.loc[mask, "Sex"] = 1
mask = test_data["Sex"] == "female"
test_data.loc[mask, "Sex"] = 0

# Split into X and y
X_cols = ["Pclass", "Sex",  "SibSp", "Parch", "Fare", "Age"]
y_col = "Survived"
X_train = np.array(train_data[X_cols])
y_train = np.array(train_data[y_col])
#X_train = normalize(X_train)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# Run k-fold cross validation on different models(classifiers)
for model in models:
    print(model[1])
    run_k_fold_on_model(model[0], X_train, y_train)



# Test different neural network models
#for i in range(1, 11):
#    for j in range(1, 11):
#        model = MLPClassifier(solver='lbfgs', alpha=1e-5,
#          hidden_layer_sizes=(i, j), random_state=1)
#        acc = []
#        for train, test in kf.split(train_data):
#            X = X_train[train]
#            y = y_train[train]
#            X_valid = X_train[test]
#            y_valid = y_train[test]
#            model.fit(X, y)
#            predict= model.predict(X_valid)
#            acc.append(accuracy_score(y_valid, predict))
#        print("Hidden: {} {} Average: {}".format(i,j,np.mean(acc)))
# Optinal around 5, 5

# Test different random forest models
#for i in range(1, 20):
#    model = RandomForestClassifier(max_depth=i, random_state=0)
#    acc = []
#    for train, test in kf.split(train_data):
#        X = X_train[train]
#        y = y_train[train]
#        X_valid = X_train[test]
#        y_valid = y_train[test]
#        model.fit(X, y)
#        predict= model.predict(X_valid)
#        acc.append(accuracy_score(y_valid, predict))
#    print("max_depth: {} Average: {}".format(i,np.mean(acc)))

# Test diff Random Forests
#for i in range(1, len(X_train[0])+1):
#    print("max_features={}".format(i))
#    model = RandomForestClassifier(max_depth=7, random_state=0, max_features = i)
#    run_k_fold_on_model(model, X_train, y_train)
    
#for i in range(10, 200, 10):
#    print("n_estimators={}".format(i))
#    model = RandomForestClassifier(max_depth=7, random_state=0, n_estimators=i)
#    run_k_fold_on_model(model, X_train, y_train)

#kernels = ['linear', 'poly', 'rbf', 'sigmoid']
#for kernel in kernels:
#    print(kernel)
#    model=svm.SVC(probability=True,kernel=kernel)
#    run_k_fold_on_model(model, X_train, y_train)
    