from sklearn.naive_bayes import GaussianNB
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score)
rng = np.random.default_rng(seed=4)

ues = {}
for i in range (1,41):
    a = 'ue'+str(i)+'.txt'
    ues[i] = np.loadtxt(a, delimiter=',')
keys = ues.keys()
elements = list(ues.items())
embb, mtc, urllc = {}, {}, {}
l1 = [2,5,8,12,15,18,22,25,28,32,35,38]
l2 = [3,6,9,13,16,19,23,26,29,33,36,39]
l3 = [1,4,7,10,11,14,17,20,21,24,27,30,31,34,37,40]
for k in l1: 
    embb[k] = ues[k]
for j in l2: 
    mtc[j] = ues[j]
for m in l3: 
    urllc[m] = ues[m]
'''a = list(rng.choice(l1, size = 4, replace=False))
b = list(rng.choice(l2, size = 4, replace=False))
c = list(rng.choice(l3, size = 5, replace=False))
a1 = list(rng.choice(l1, size = 4, replace=False))
b1 = list(rng.choice(l2, size = 4, replace=False))
c1 = list(rng.choice(l3, size = 5, replace=False))'''
a = list(rng.choice(l1, size = 8, replace=False))
b = list(rng.choice(l2, size = 8, replace=False))
c = list(rng.choice(l3, size = 10, replace=False))
rng.shuffle(a)
rng.shuffle(b)
rng.shuffle(c)
a1,a = a[:4],a[4:]
b1,b = b[:4],b[4:]
c1,c = c[:5],c[5:]

test_class,test_data,training_data,training_class = [],[],[],[]
test_UE = a1+b1+c1
training_UE = a+b+c
print (training_UE,test_UE)
for i in training_UE:
    training_data += [ues[i]]
training_data = np.array(training_data)
training_class = np.array(['embb','embb','embb','embb','mtc','mtc','mtc','mtc','urllc','urllc','urllc','urllc','urllc'])
for i in test_UE:
    test_data += [ues[i]]
test_data = np.array(test_data)
test_class = np.array(['embb','embb','embb','embb','mtc','mtc','mtc','mtc','urllc','urllc','urllc','urllc','urllc'])


classificator = GaussianNB()
classificator.fit(training_data, training_class)
predictions = classificator.predict(test_data)
for i, UE in enumerate(test_data):
    print(f"UE: {test_UE[i]} - predicted class: {predictions[i]} - real class: {test_class[i]}" )

Accuracy = accuracy_score(predictions,test_class)
f1 = f1_score (predictions,test_class, average="weighted")

print("Accuracy:", Accuracy)
print("F1 Score:", f1)



