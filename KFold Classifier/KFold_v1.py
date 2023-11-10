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
# import csv
from sklearn.preprocessing import StandardScaler

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
ues = {}
embb_ues_data, mtc_ues_data, urllc_ues_data = [],[],[]
n_embb_ues, n_mtc_ues, n_urllc_ues = 0,0,0
def data_processing (data_list,n_ue):
    dl_mcs, dl_turbo, dl_brate, ul_mcs, ul_brate = 0,0,0,0,0
    last_time = 0
    for j in range (1,len(data_list)):
        time = data_list[j][0] - data_list[j-1][0]
        if np.isinf(data_list[j-1][1]) or np.isinf(data_list[j-1][2]) or np.isinf(data_list[j-1][3]) or np.isinf(data_list[j-1][4]) or np.isinf(data_list[j-1][5]): continue 
        last_time += time
        dl_mcs += data_list[j-1][1] * time
        dl_turbo += data_list[j-1][2] * time
        dl_brate += data_list[j-1][3] * time
        ul_mcs += data_list[j-1][4] * time
        ul_brate += data_list[j-1][5] * time

    dl_mcs = dl_mcs/last_time
    dl_turbo = dl_turbo/last_time
    dl_brate = dl_brate/last_time
    ul_mcs = ul_mcs/last_time
    ul_brate = ul_brate/last_time                  
    data = [dl_mcs, dl_turbo, dl_brate, ul_mcs, ul_brate]

    if str(n_ue) in embb_ues: 
        class_ue = 'embb'
        global embb_ues_data 
        embb_ues_data.append(data)
    elif str(n_ue) in mtc_ues:
        class_ue = 'mtc'
        global mtc_ues_data 
        mtc_ues_data.append(data)
    else: 
        class_ue = 'urllc'
        global urllc_ues_data 
        urllc_ues_data.append(data)
    return data,class_ue

# opening files and calling data_processing
ue_data, ue_classes = [],[]
wished_cols = [0,7,9,10,13,15]
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
                    inf_ue = pd.read_csv(traffic_case+tr+exp+bs+a, skiprows=1, usecols = wished_cols,dtype=np.float64)
                    inf_ue = np.array(inf_ue)
                    data = data_processing (inf_ue,n_ue)
                    ues[traffic_case+tr+exp+bs+a] = data
                    #ue_data.append(data[0])
                    #ue_classes.append(data[1])
                except FileNotFoundError: pass

def data_used (b,c):
    rng = np.random.default_rng(seed=b)
    print (embb_ues_data)
    len_embb = len(embb_ues_data)//3
    len_mtc = len(mtc_ues_data)//3
    len_urllc = len(urllc_ues_data)//3
    rng.shuffle(embb_ues_data)
    rng.shuffle(mtc_ues_data)
    rng.shuffle(urllc_ues_data)
    rng.shuffle(embb_ues_data)
    rng.shuffle(mtc_ues_data)
    rng.shuffle(urllc_ues_data)
    embb_ues_data_used = embb_ues_data[len_embb:]
    embb_ues_data_notused = embb_ues_data[:len_embb]
    mtc_ues_data_used = mtc_ues_data[len_mtc:]
    mtc_ues_data_notused = mtc_ues_data[:len_mtc]
    urllc_ues_data_used = urllc_ues_data[len_urllc:]
    urllc_ues_data_notused = urllc_ues_data[:len_urllc]
    ue_data = embb_ues_data_used + mtc_ues_data_used + mtc_ues_data_used
    ue_classes = ['embb'] * len(embb_ues_data_used) + ['mtc'] * len(mtc_ues_data_used) + ['urllc'] * len(urllc_ues_data_used)
    #x,y =  utils.shuffle(ue_data, ue_classes,random_state = c)
    return ue_data,ue_classes


def K_Fold (a,k_fold,b,c):
    ue_data_classifier, ue_classes_classifier = data_used (b,c)
    ue_data_classifier = np.array(ue_data)
    ue_classes_classifier = np.array(ue_classes)
    print (ue_classes_classifier)
    print (ue_data_classifier)

    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=a)
    accuracy_scores = []
    for train_index, test_index in kf.split(ue_data_classifier,ue_classes_classifier):
        data_train, data_test = ue_data_classifier[train_index], ue_data_classifier[test_index]
        class_train, class_test = ue_classes_classifier[train_index], ue_classes_classifier[test_index]
        classifier.fit(data_train, class_train)
        predictions = classifier.predict(data_test)
        accuracy = accuracy_score(predictions,class_test)
        accuracy_scores.append(accuracy)
        confusion_matrixes = confusion_matrix(class_test, predictions, labels=["embb", "mtc", "urllc"])
        print (confusion_matrixes)

    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    print (accuracy_scores)
    result = str(mean_accuracy) + ' ; ' + str(std_accuracy)
    return result
    
print(K_Fold (8,3,1,5))



 