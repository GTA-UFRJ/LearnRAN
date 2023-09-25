from sklearn.naive_bayes import GaussianNB
import numpy as np

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


    for k in range (5):
        training_inf = np.delete(training_data2, k, 1)
        test_inf = np.delete(test_data2, k, 1)
        classifier.fit(training_inf, training_class)
        predictions = classifier.predict(test_inf)
        Accuracy = accuracy_score(predictions,test_class)
        f1 = f1_score (predictions,test_class, average="weighted")
        #print('after ' + str(k) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
        print( str(k) +',k,k,k,' + str(Accuracy) + ',' + str(f1))
        
        for j in range (5):
            if k == j: continue 
            training_inf = np.delete(training_data2, [k,j], 1)
            test_inf = np.delete(test_data2, [k,j], 1)
            classifier.fit(training_inf, training_class)
            predictions = classifier.predict(test_inf)
            Accuracy = accuracy_score(predictions,test_class)
            f1 = f1_score (predictions,test_class, average="weighted")
            #print('after ' + str(k) + " and " + str(j) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
            print( str(k) + "," + str(j) + ',k,k,' + str(Accuracy) + ',' + str(f1))

            for l in range (5):
                if k == j or k == l or l == j: continue
                training_inf = np.delete(training_data2, [k,j,l], 1)
                test_inf = np.delete(test_data2, [k,j,l], 1)
                classifier.fit(training_inf, training_class)
                predictions = classifier.predict(test_inf)
                Accuracy = accuracy_score(predictions,test_class)
                f1 = f1_score (predictions,test_class, average="weighted")
                #print('after ' + str(k) + ", " + str(j) + " and " + str(l) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
                print( str(k) + "," + str(j) + "," + str(l) +',k,' + str(Accuracy) + ',' + str(f1))

                for m in range (5):
                    if k == j or k == l or l == j or m == j or m == k or m == l: continue
                    training_inf = np.delete(training_data2, [k,j,l,m], 1)
                    test_inf = np.delete(test_data2, [k,j,l,m], 1)
                    classifier.fit(training_inf, training_class)
                    predictions = classifier.predict(test_inf)
                    Accuracy = accuracy_score(predictions,test_class)
                    f1 = f1_score (predictions,test_class, average="weighted")
                    #print('after ' + str(k) + ", " + str(j) + ", " + str(l) + " and " + str(m) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
                    print( str(k) + "," + str(j) + "," + str(l) + "," +str(m) +  "," + str(Accuracy) + ',' + str(f1))


print(NaiveBayes (4))


