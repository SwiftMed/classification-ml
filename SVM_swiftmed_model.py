import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn import linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv("dataset.csv")
dataset = dataset[["Disease", "Symptom_1", "Symptom_2", "Symptom_3"]]
severity = pd.read_csv("Symptom-severity.csv")
severity = severity[["Symptom", "weight"]]
predict = "Disease"



le = preprocessing.LabelEncoder()
Symptom_1 = le.fit_transform(list(dataset["Symptom_1"]))
Symptom_2 = le.fit_transform(list(dataset["Symptom_2"]))
Symptom_3 = le.fit_transform(list(dataset["Symptom_3"]))
X = np.array(list(zip(Symptom_1, Symptom_2, Symptom_3)))
y = np.array(dataset[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
clf = svm.SVC(kernel="poly", degree=6)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)



