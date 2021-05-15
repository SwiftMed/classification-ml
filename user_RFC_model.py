#importing
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("dataset.csv")
dataset = dataset[["Disease", "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]]
severity = pd.read_csv("Symptom-severity.csv")
severity = severity[["Symptom", "weight"]]
predict = "Disease"

le = preprocessing.LabelEncoder()
Symptom_1 = le.fit_transform(list(dataset["Symptom_1"]))
#keys1 = le.classes_
#values1 = le.transform(le.classes_)
#dictionary1 = dict(zip(keys1, values1))
Symptom_2 = le.fit_transform(list(dataset["Symptom_2"]))
#keys2 = le.classes_
#values2 = le.transform(le.classes_)
#dictionary2 = dict(zip(keys2, values2))
Symptom_3 = le.fit_transform(list(dataset["Symptom_3"]))
#keys3 = le.classes_
#values3 = le.transform(le.classes_)
#dictionary3 = dict(zip(keys3, values3))
Symptom_4 = le.fit_transform(list(dataset["Symptom_4"]))
#keys4 = le.classes_
#values4 = le.transform(le.classes_)
#dictionary4 = dict(zip(keys4, values4))

severitylist = le.fit_transform(list(severity["Symptom"]))
def take_input():
    symptom_1 = input("What symptom are you feeling?: ").lower().replace(" ", "_")
    symptom_2 = input("What symptom are you feeling?: ").lower().replace(" ", "_")
    symptom_3 = input("What symptom are you feeling?: ").lower().replace(" ", "_")
    symptom_4 = input("What symptom are you feeling?: ").lower().replace(" ", "_")
    symptoms = symptom_1 + " " + symptom_2 + " " + symptom_3 + " " + symptom_4
    symptoms_list = symptoms.split()
    symptoms_list = le.transform(symptoms_list)
    symptoms_list = np.array(list(zip(symptoms_list)))
    return symptoms_list, type(symptoms_list), Symptom_1

X = np.array(list(zip(Symptom_1, Symptom_2, Symptom_3, Symptom_4)))
y = np.array(dataset[predict])

#X = np.array(dataset.drop([predict], 1))
#y = np.array(dataset[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=25)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("The Accuracy of this prediction is", acc)
pred = model.predict(x_test)
for i in range(len(pred[:5])):
    if pred[i][:3] == "Fun" or pred[i][:3] == "GER" or pred[i][:3] == "Dia" \
            or pred[i][:3] == "Ost" or pred[i][:3] == "Art" or pred[i][:3] == "Uni"\
            or pred[i][:3] == "Dru":
        pred[i] = "See Family Doctor"
    elif pred[i][:3] == "Pep" or pred[i][:3] == "AID" or pred[i][:3] == "Par"\
            or pred[i][:3] == "Mal" or pred[i][:3] == "Den" or pred[i][:3] == "Hea" or \
            pred[i][:3] == "(ve" or pred[i][:3] == "Pso" or pred[i][:3] == "Uri" or pred[i][:3] == "Cer":
        pred[i] = "URGENT! Go to to Emergency Room"
    elif pred[i][:3] == "Chr" or pred[i][:3] == "Gas" or pred[i][:3] == "Bro" or pred[i][:3] == "Hyp"\
            or pred[i][:3] == "Ser" or pred[i][:3] == "Jau" or pred[i][:3] == "Chic" or pred[i][:3] == "Typ":
        pred[i] = "Go to Emergency Room"
    elif pred[i][:3] == "hep" or pred[i][:3] == "Hep" or pred[i][:3] == "Alc" or pred[i][:3] =="Tub" or\
            pred[i][:3] == "Pne" or pred[i][:3] == "Dim" or pred[i][:3] == "Var" or pred[i][:3] == "Hyp"\
            or pred[i][:3] == "Imp" or pred[i][:3] == "Chi":
        pred[i] = "Go to Emergency Room"
    elif pred[i][:3] == "Mig" or pred[i][:3] == "Acn" or pred[i][:3] == "All" or pred[i][:3] == "Com":
        pred[i] = "Take Medicine"
for x in range(len(pred[:5])):
    print(pred[x], x_test[x], y_test[x])



