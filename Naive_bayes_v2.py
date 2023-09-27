from sklearn.naive_bayes import GaussianNB
import numpy as np
from itertools import combinations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score)
classifier = GaussianNB()
ues = {}
for i in range (1,41):
    a = 'ue'+str(i)+'.txt'
    with open ("//lab//users//Cruz//vivian//LearnRAN//Data//"+a) as b:
        ues[i]= eval('['+b.read()+']')

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

n_elements = [0,1,2,3,4]
comb_2 = list(combinations(n_elements,2))
comb_3 = list(combinations(n_elements,3))
comb_4 = list(combinations(n_elements,4))

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
    training_data2 = training_data

    for i in test_UE:
        test_data += [ues[i]]
    test_data = np.array(test_data)
    test_class = np.array(['embb','embb','embb','embb','mtc','mtc','mtc','mtc','urllc','urllc','urllc','urllc','urllc'])
    test_data2 = test_data

    print (test_data)
    classifier.fit(training_data, training_class)
    predictions = classifier.predict(test_data)
    Accuracy = accuracy_score(predictions,test_class)
    f1 = f1_score (predictions,test_class, average="weighted")
    #print("Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
    print('k,k,k,k,' + str(Accuracy) + ',' + str(f1))
    
    for k in range (5):
        training_inf = np.delete(training_data2, k, 1)
        test_inf = np.delete(test_data2, k, 1)
        classifier.fit(training_inf, training_class)
        predictions = classifier.predict(test_inf)
        Accuracy = accuracy_score(predictions,test_class)
        f1 = f1_score (predictions,test_class, average="weighted")
        #print('after ' + str(k) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
        print( str(k) +',k,k,k,' + str(Accuracy) + ',' + str(f1))
        
    for l in (comb_2):
            training_inf = np.delete(training_data2, [l[0],l[1]], 1)
            test_inf = np.delete(test_data2, [l[0],l[1]], 1)
            classifier.fit(training_inf, training_class)
            predictions = classifier.predict(test_inf)
            Accuracy = accuracy_score(predictions,test_class)
            f1 = f1_score (predictions,test_class, average="weighted")
            #print('after ' + str(l[0]) + " and " + str(l[1]) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
            print( str(l[0]) + "," + str(l[1]) + ',k,k,' + str(Accuracy) + ',' + str(f1))

    for m in (comb_3):
            training_inf = np.delete(training_data2, [m[0],m[1],m[2]], 1)
            test_inf = np.delete(test_data2, [m[0],m[1],m[2]], 1)
            classifier.fit(training_inf, training_class)
            predictions = classifier.predict(test_inf)
            Accuracy = accuracy_score(predictions,test_class)
            f1 = f1_score (predictions,test_class, average="weighted")
            #print('after ' + str(m[0]) + ", " + str(m[1]) + " and " + str(m[2]) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
            print( str(m[0]) + "," + str(m[1]) + "," + str(m[2]) +',k,' + str(Accuracy) + ',' + str(f1))

    for n in (comb_4):
            training_inf = np.delete(training_data2, [n[0],n[1],n[2],n[3]], 1)
            test_inf = np.delete(test_data2, [n[0],n[1],n[2],n[3]], 1)
            classifier.fit(training_inf, training_class)
            predictions = classifier.predict(test_inf)
            Accuracy = accuracy_score(predictions,test_class)
            f1 = f1_score (predictions,test_class, average="weighted")
            #print('after ' + str(n[0]) + ", " + str(n[1]) + ', ' + str(n[1]) + " and " + str(n[3]) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
            print( str(n[0]) + "," + str(n[1]) + "," + str(n[2]) + ',' + str(n[3]) + ',' + str(Accuracy) + ',' + str(f1))


print(NaiveBayes (39))



