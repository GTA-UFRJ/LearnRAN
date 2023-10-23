from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score)
import numpy as np
from itertools import combinations

classifier = KNeighborsClassifier(n_neighbors=3)
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

def NaiveBayes (a,k):
    
    ue_class,ue_data = [],[]
    for i in l1+l2+l3:
        ue_data += [ues[i]]
    ue_data = np.array(ue_data)
    ue_class = np.array(['embb','embb','embb','embb','embb','embb','embb','embb','embb','embb','embb','embb','mtc','mtc','mtc','mtc','mtc','mtc','mtc','mtc','mtc','mtc','mtc','mtc','urllc','urllc','urllc','urllc','urllc','urllc','urllc','urllc','urllc','urllc','urllc','urllc','urllc','urllc','urllc','urllc'])
    ue_data2 = ue_data
    
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=a)
    
    accuracy_scores = []

    for train_index, test_index in kf.split(ue_data,ue_class):
        data_train, data_test = ue_data[train_index], ue_data[test_index]
        class_train, class_test = ue_class[train_index], ue_class[test_index]
        classifier.fit(data_train, class_train)
        predictions = classifier.predict(data_test)
        Accuracy = accuracy_score(predictions,class_test)
        accuracy_scores.append(Accuracy)

    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    print (accuracy_scores)
    print (mean_accuracy, ' ; ', std_accuracy)
    print ('k,k,k,k,'+ str(mean_accuracy) + ',' + str(std_accuracy))
    
    for k in range (5):
        data_inf = np.delete(ue_data2, k, 1)
        for train_index, test_index in kf.split(data_inf,ue_class):
            data_train, data_test = data_inf[train_index], data_inf[test_index]
            class_train, class_test = ue_class[train_index], ue_class[test_index]
            classifier.fit(data_train, class_train)
            predictions = classifier.predict(data_test)
            Accuracy = accuracy_score(predictions,class_test)
            accuracy_scores.append(Accuracy)
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        #print('after ' + str(k) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
        print( str(k) +',k,k,k,' + str(Accuracy) + ',' + str(std_accuracy))
    
    for l in comb_2:
        data_inf = np.delete(ue_data2, [l[0],l[1]], 1)
        for train_index, test_index in kf.split(data_inf,ue_class):
            data_train, data_test = data_inf[train_index], data_inf[test_index]
            class_train, class_test = ue_class[train_index], ue_class[test_index]
            classifier.fit(data_train, class_train)
            predictions = classifier.predict(data_test)
            Accuracy = accuracy_score(predictions,class_test)
            accuracy_scores.append(Accuracy)
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        #print('after ' + str(l[0]) + " and " + str(l[1]) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
        print( str(l[0]) + "," + str(l[1]) + ',k,k,' + str(mean_accuracy) + ',' + str(std_accuracy))
        
    for m in comb_3:
        data_inf = np.delete(ue_data2, [m[0],m[1],m[2]], 1)
        for train_index, test_index in kf.split(data_inf,ue_class):
            data_train, data_test = data_inf[train_index], data_inf[test_index]
            class_train, class_test = ue_class[train_index], ue_class[test_index]
            classifier.fit(data_train, class_train)
            predictions = classifier.predict(data_test)
            Accuracy = accuracy_score(predictions,class_test)
            accuracy_scores.append(Accuracy)
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        #print('after ' + str(m[0]) + ", " + str(m[1]) + " and " + str(m[2]) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
        print( str(m[0]) + "," + str(m[1]) + "," + str(m[2]) +',k,' + str(mean_accuracy) + ',' + str(std_accuracy))

    for n in comb_4:
        data_inf = np.delete(ue_data2, [n[0],n[1],n[2],n[3]], 1)
        for train_index, test_index in kf.split(data_inf,ue_class):
            data_train, data_test = data_inf[train_index], data_inf[test_index]
            class_train, class_test = ue_class[train_index], ue_class[test_index]
            classifier.fit(data_train, class_train)
            predictions = classifier.predict(data_test)
            Accuracy = accuracy_score(predictions,class_test)
            accuracy_scores.append(Accuracy)
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        #print('after ' + str(n[0]) + ", " + str(n[1]) + ', ' + str(n[1]) + " and " + str(n[3]) + ' st column deleted -> ' + "Accuracy:" + str(Accuracy) + ' ; ' + "F1 Score:" + str(f1))
        print( str(n[0]) + "," + str(n[1]) + "," + str(n[2]) + ',' + str(n[3]) + ',' + str(mean_accuracy) + ',' + str(std_accuracy))

print(NaiveBayes (51,3))



