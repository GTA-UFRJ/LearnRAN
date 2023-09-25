from sklearn.naive_bayes import GaussianNB
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score)

ues = {}
for i in range (1,41):
    a = 'ue'+str(i)+'.txt'
    with open ("//lab//users//Cruz//vivian//LearnRAN//Data//"+a) as b:
        ues[i] = list(b)
print (ues)

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

def NaiveBayes (a):
    rng = np.random.default_rng(seed=a)
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
    for i in training_UE:
        training_data += [ues[i]]
    training_data = np.array(training_data)
    training_class = np.array(['embb','embb','embb','embb','mtc','mtc','mtc','mtc','urllc','urllc','urllc','urllc','urllc'])
    for i in test_UE:
        test_data += [ues[i]]
    test_data3 = test_data
    training_data3 = training_data
    test_data = np.array(test_data)
    test_class = np.array(['embb','embb','embb','embb','mtc','mtc','mtc','mtc','urllc','urllc','urllc','urllc','urllc'])
    test_data2 = test_data3
    training_data2 = training_data3


    def classifier (training_data, training_class):
        classifier = GaussianNB()
        classifier.fit(training_data, training_class)
        predictions = classifier.predict(test_data)
        return predictions
        for i, UE in enumerate(test_data):
            yield(f"UE: {test_UE[i]} - predicted class: {predictions[i]} - real class: {test_class[i]}" )

    def Accuracy (predictions,test_class):
        Accuracy = accuracy_score(predictions,test_class)
        f1 = f1_score (predictions,test_class, average="weighted")
        return "Accuracy:" + str(Accuracy) + '\n' + "F1 Score:" + str(f1)

    for test in range (5):
        for a in range(len(test_data3)):
            print (type (test_data3[a][test]))
            test_data3.remove(test_data3[a][test])
            training_data3.remove(test_data3[a][test])
        predictions = classifier (training_data, training_class)
        Accuracy (predictions,test_class)

        test_data3 = test_data2
        training_data3 = training_data2

    for test2 in range (4):
        for a in range(len(test_data)):
            test_data.remove(a[test2])
            test_data.remove(a[test2])
            training_data.remove(a[test2])
            training_data.remove(a[test2])
        predictions = classifier (training_data, training_class)
        Accuracy (predictions,test_class)

        test_data = test_data2
        training_data = training_data2

    for test3 in range (3):
        for a in range(len(test_data)):
            test_data.remove(a[test3])
            test_data.remove(a[test3])
            test_data.remove(a[test3])
            training_data.remove(a[test3])
            training_data.remove(a[test3])
            training_data.remove(a[test3])
        predictions = classifier (training_data, training_class)
        Accuracy (predictions,test_class)

        test_data = test_data2
        training_data = training_data2
    

print(NaiveBayes (4))



