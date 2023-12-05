"""Author: Vivian Maria da Silva e Souza 
Instutution: Coppe Del UFRJ"""
from sklearn.naive_bayes import GaussianNB
from sklearn import utils
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    f1_score)
import numpy as np
from itertools import combinations
import os
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestCentroid

configuration_file = '/lab/users/Cruz/vivian/LearnRAN/KFold Classifier/KFold_v1.ini'
 
if os.path.exists(configuration_file):
    print(f"O arquivo {configuration_file} existe.")
else:
    print(f"O arquivo {configuration_file} não foi encontrado.")

config = configparser.ConfigParser()
config.read(configuration_file)
for section in config.sections():
    print(f"[{section}]")
    for key, value in config.items(section):
        print(f"{key} = {value}")
    print()

rome_slow_close_dir = config['DEFAULT']['rome_slow_close']
rome_static_close_dir = config['DEFAULT']['rome_static_close']
rome_static_far_dir = config['DEFAULT']['rome_static_far']
rome_static_medium_dir = config['DEFAULT']['rome_static_medium']
embb_ues = config['DEFAULT']['default_embb_ues']
mtc_ues = config['DEFAULT']['default_mtc_ues']
urllc_ues = config['DEFAULT']['default_urllc_ues']

possible_cases = [rome_slow_close_dir, rome_static_close_dir, rome_static_far_dir, rome_static_medium_dir]

pd.options.display.max_rows = 9999

classifier = GaussianNB()
ues, ues_av = {},{}

def calculate_rates_classifier (data_list,n_ue):
    dl_brate, ul_brate = [],[]
    weights_times = []
    data = []
    for j in range (1,len(data_list)):
        time_interval = data_list[j][0] - data_list[j-1][0]
        if np.isinf(data_list[j-1][1]) or np.isinf(data_list[j-1][2]): continue 
        data.append(data_list[j][1:])
        weights_times.append(time_interval)
        dl_brate.append(data_list[j-1][1])
        ul_brate.append(data_list[j-1][2])
    weights_times = np.array(weights_times)
    dl_brate_av = np.average(dl_brate, weights=weights_times)
    dl_brate_std = np.std(dl_brate) 
    ul_brate_av =  np.average(ul_brate, weights=weights_times)         
    ul_brate_std = np.std(ul_brate) 
    data_av = [dl_brate_av, dl_brate_std, ul_brate_av, ul_brate_std]

    if str(n_ue) in embb_ues: 
        class_ue = 'embb'
    elif str(n_ue) in mtc_ues:
        class_ue = 'mtc'
    else: 
        class_ue = 'urllc'
    return data_av,class_ue,data,class_ue

# opening files and calling calculate_rates_classifier 

wished_cols = [0,10,15]
for n_tr in range (18):
    tr = 'tr' + str(n_tr) + '/'
    for n_exp in range (1,7): 
        exp = 'exp' + str (n_exp) + '/'
        for n_ue in range (1,41):
            if n_ue%10 != 0: 
                n_bs = n_ue//10 + 1
            else: 
                n_bs = n_ue//10 
            bs = 'bs' + str (n_bs) + '/'
            a = 'ue'+str(n_ue)+'.csv'
            for traffic_case in possible_cases:
                try:
                    inf_ue = pd.read_csv(traffic_case+tr+exp+bs+a, skiprows=1, usecols = wished_cols,dtype=np.float64, memory_map=True)
                    inf_ue = np.array(inf_ue)
                    data = calculate_rates_classifier (inf_ue,n_ue)
                    ues_av[traffic_case+tr+exp+bs+a] = data[:2]
                    ues[traffic_case+tr+exp+bs+a] = data[2:]
                except FileNotFoundError: pass

def data_used (b,ues):
    ue_data, ue_classes = [],[]
    ue_data_nu, ue_classes_nu = [],[]
    len = []
    ues_sl = same_len(ues)
    all_samples = list(ues_sl.keys())
    all_labels = [value[1] for value in ues_sl.values()]  

    used_samples, notused_samples, used_labels, notused_labels = train_test_split(
    all_samples, all_labels, test_size=1/3, stratify=all_labels, random_state=b)

    ues_used = {sample: ues_sl[sample] for sample in used_samples}
    ues_notused = {sample: ues_sl[sample] for sample in notused_samples}

    for n in ues_used:
        ue_data.append(ues_used[n][0])
        ue_classes.append(ues_used[n][1])

    for n in ues_notused:
        ue_data_nu.append(ues_notused[n][0])
        ue_classes_nu.append(ues_notused[n][1])

    return ue_data,ue_classes,ue_data_nu,ue_classes_nu

def same_len (dic):
    len_max = 0
    for i in dic: 
        if len(dic[i]) > len_max: len_max = len(dic[i])
    for n in dic:
        while len(dic[n]) < len_max:
            n = np.append(dic[n], [0,0])
    return dic

def gb_classifier_kf (a,k_fold,b):
    kf = kfold (a, k_fold, b, ues_av)
    data_train = kf[0]
    data_validation = kf[2]
    class_train = kf[1]
    class_validation = kf[3]
    accuracy_scores = []
    for i in range(3):
        classifier.fit(data_train[i], class_train[i])
        predictions = classifier.predict(data_validation[i])
        accuracy = accuracy_score(predictions,class_validation[i])
        accuracy_scores.append(accuracy)
        confusion_matrixes = confusion_matrix(class_validation[i], predictions, labels=["embb", "mtc", "urllc"])
        print (confusion_matrixes)

    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    print (accuracy_scores)
    result = str(mean_accuracy) + ' ; ' + str(std_accuracy)
    return result

def mlp_classifier_kf (a,k_fold,b,n):
    mlp_gs = MLPClassifier(max_iter=n)
    parameter_space = { 'hidden_layer_sizes': [(100,100,100,100,100,100,100,100,100,100)],
                        'activation': ['tanh', 'relu'], 'solver': ['adam'], 'alpha': [0.1, 0.01, 0.0001, 0.05], 
                        'learning_rate': ['constant','adaptive'], }
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=6)
   
    kf = kfold (a, k_fold, b, ues_av)
    data_train = kf[0]
    data_validation = kf[2]
    class_train = kf[1]
    class_validation = kf[3]
    accuracy_scores = []
    for i in range(3):
        clf.fit(data_train[i], class_train[i])
        predictions = clf.predict(data_validation[i])
        accuracy = accuracy_score(predictions,class_validation[i])
        accuracy_scores.append(accuracy)
        confusion_matrixes = confusion_matrix(class_validation[i], predictions, labels=["embb", "mtc", "urllc"])
        print (confusion_matrixes)
        print()
        print('Best parameters found:\n', clf.best_params_,'nº iterations',str(n))
        print()
        y_true, y_pred = class_validation[i] , predictions
        print('Results on the test set:')
        print(classification_report(y_true, y_pred))

    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    print (accuracy_scores)
    result = str(mean_accuracy) + ' ; ' + str(std_accuracy)

##Naive Bayes without KFold
def gb_classifier (b):
    aux = data_used(b,ues_av)
    ue_data_train = aux[0]
    ue_class_train = aux[1]
    ue_data_test = aux[2]
    ue_class_test = aux[3]
    classifier.fit(ue_data_train,ue_class_train)
    predictions = classifier.predict(ue_data_test)
    accuracy = accuracy_score(predictions,ue_class_test)
    confusion_matrixes = confusion_matrix(ue_class_test, predictions, labels=["embb", "mtc", "urllc"])
    print (confusion_matrixes)
    print()
    return accuracy

## MLP without KFold
def mlp_classifier (b,n):
    mlp_gs = MLPClassifier(max_iter=n)
    parameter_space = { 'hidden_layer_sizes': [(100,100,100,100,100,100,100,100,100,100)],
                        'activation': ['relu'], 'solver': ['adam'], 'alpha': [0.05], 
                        'learning_rate': ['adaptive'],}
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=6)
    aux = data_used(b,ues_av)
    ue_data_train = aux[0]
    ue_class_train = aux[1]
    ue_data_test = aux[2]
    ue_class_test = aux[3]
    clf.fit(ue_data_train, ue_class_train)
    predictions = clf.predict(ue_data_test)
    accuracy = accuracy_score(predictions,ue_class_test)
    confusion_matrixes = confusion_matrix(ue_class_test, predictions, labels=["embb", "mtc", "urllc"])
    print (confusion_matrixes)
    print()
    y_true, y_pred = ue_class_test, predictions
    print('Results on the test set:')
    print(classification_report(y_true, y_pred))
    return accuracy

def kfold (a, k_fold,b,ue):
    datas_train, datas_validation, classes_train, classes_validation = [],[],[],[]
    n = data_used (b,ue)
    ue_data_classifier = n[0]
    ue_classes_classifier = n[1]
    ue_data_classifier = np.array(ue_data_classifier)
    ue_classes_classifier = np.array(ue_classes_classifier)
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=a)
    for train_index, test_index in kf.split(ue_data_classifier,ue_classes_classifier):
        data_train, data_validation = ue_data_classifier[train_index], ue_data_classifier[test_index]
        class_train, class_validation = ue_classes_classifier[train_index], ue_classes_classifier[test_index]
        datas_validation.append(data_validation)
        datas_train.append(data_train)
        classes_validation.append(class_validation)
        classes_train.append(class_train)
    return datas_train,classes_train,datas_validation,classes_validation

print ('Naive bayes')
print (gb_classifier(1))
print ('MLP')
print (mlp_classifier(1,200))