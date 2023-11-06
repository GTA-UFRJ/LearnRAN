from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score)
import numpy as np
from itertools import combinations
import os
import configparser
import csv
from sklearn.preprocessing import StandardScaler
inf = 10**308

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

classifier = KNeighborsClassifier(n_neighbors=3)
ues = {}
embb_ues_data, mtc_ues_data, urllc_ues_data = [],[],[]
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
            try:
                with open (rome_slow_close_dir+tr+exp+bs+a) as b:
                    file_reader_rswc = csv.reader(b, delimiter=',') 
                    f = []
                    data = []
                    time = 0
                    for i in file_reader_rswc:
                        time += 1
                        if time == 1: pass
                        try:
                            i = [i[0]] + [i[7]] + [i[9]] + [i[10]] + [i[13]] + [i[15]]                               
                            i = [float(k) for k in i]             
                            f.append(i)
                        except ValueError: pass
                        except IndexError: pass
                    while len(f[0]) != len(f[-1]):
                        f.remove(f[-1])
                dl_mcs, dl_turbo, dl_brate, ul_mcs, ul_brate = 0,0,0,0,0
                last_time = 0
                for j in range (1,len(f)):
                    time = f[j][0] - f[j-1][0]
                    if np.isinf(f[j-1][1]) or np.isinf(f[j-1][2]) or np.isinf(f[j-1][3]) or np.isinf(f[j-1][4]) or np.isinf(f[j-1][5]): continue 
                    last_time += time
                    dl_mcs += f[j-1][1] * time
                    dl_turbo += f[j-1][2] * time
                    dl_brate += f[j-1][3] * time
                    ul_mcs += f[j-1][4] * time
                    ul_brate += f[j-1][5] * time
                dl_mcs = round(dl_mcs/last_time,5)
                dl_turbo = round(dl_turbo/last_time,5)
                dl_brate = round(dl_brate/last_time,5)
                ul_mcs = round(ul_mcs/last_time,5)
                ul_brate = round(ul_brate/last_time,5)                   
                data.append(dl_mcs);data.append(dl_turbo)
                data.append(dl_brate);data.append(ul_mcs);data.append(ul_brate)
                ues[rome_slow_close_dir+tr+exp+bs+a] = data

                if str(n_ue) in embb_ues: 
                    embb_ues_data.append(data)
                elif str(n_ue) in mtc_ues:
                    mtc_ues_data.append(data)
                else: 
                    urllc_ues_data.append(data)
            except FileNotFoundError: pass

            try:
                with open (rome_static_close_dir+tr+exp+bs+a) as c:
                    file_reader_rscc= csv.reader(c, delimiter=',') 
                    f = []
                    data = []
                    time = 0
                    for i in file_reader_rscc:
                        time += 1
                        if time == 1: pass
                        try:
                            i = [i[0]] + [i[7]] + [i[9]] + [i[10]] + [i[13]] + [i[15]]                               
                            i = [float(k) for k in i]             
                            f.append(i)
                        except ValueError: pass
                        except IndexError: pass
                    while len(f[0]) != len(f[-1]):
                        f.remove(f[-1])
                dl_mcs, dl_turbo, dl_brate, ul_mcs, ul_brate = 0,0,0,0,0
                last_time = 0
                for j in range (1,len(f)):
                    time = f[j][0] - f[j-1][0]
                    if np.isinf(f[j-1][1]) or np.isinf(f[j-1][2]) or np.isinf(f[j-1][3]) or np.isinf(f[j-1][4]) or np.isinf(f[j-1][5]): continue 
                    last_time += time
                    dl_mcs += f[j-1][1] * time
                    dl_turbo += f[j-1][2] * time
                    dl_brate += f[j-1][3] * time
                    ul_mcs += f[j-1][4] * time
                    ul_brate += f[j-1][5] * time
                dl_mcs = round(dl_mcs/last_time,5)
                dl_turbo = round(dl_turbo/last_time,5)
                dl_brate = round(dl_brate/last_time,5)
                ul_mcs = round(ul_mcs/last_time,5)
                ul_brate = round(ul_brate/last_time,5)
                data.append(dl_mcs);data.append(dl_turbo)
                data.append(dl_brate);data.append(ul_mcs);data.append(ul_brate)
                ues[rome_static_close_dir+tr+exp+bs+a] = data
                if str(n_ue) in embb_ues: 
                    embb_ues_data.append(data)
                elif str(n_ue) in mtc_ues:
                    mtc_ues_data.append(data)
                else: 
                    urllc_ues_data.append(data)
            except FileNotFoundError: pass

            try:
                with open (rome_static_far_dir+tr+exp+bs+a) as d:
                    file_reader_rscf = csv.reader(d, delimiter=',') 
                    f = []
                    data = []
                    time = 0
                    for i in file_reader_rscf:
                        time += 1
                        if time == 1: pass
                        try:
                            i = [i[0]] + [i[7]] + [i[9]] + [i[10]] + [i[13]] + [i[15]]                               
                            i = [float(k) for k in i]             
                            f.append(i)
                        except ValueError: pass
                        except IndexError: pass
                    while len(f[0]) != len(f[-1]):
                        f.remove(f[-1])
                dl_mcs, dl_turbo, dl_brate, ul_mcs, ul_brate = 0,0,0,0,0
                last_time = 0
                for j in range (1,len(f)):
                    time = f[j][0] - f[j-1][0]
                    if np.isinf(f[j-1][1]) or np.isinf(f[j-1][2]) or np.isinf(f[j-1][3]) or np.isinf(f[j-1][4]) or np.isinf(f[j-1][5]): continue 
                    last_time += time
                    dl_mcs += f[j-1][1] * time
                    dl_turbo += f[j-1][2] * time
                    dl_brate += f[j-1][3] * time
                    ul_mcs += f[j-1][4] * time
                    ul_brate += f[j-1][5] * time
                dl_mcs = round(dl_mcs/last_time,5)
                dl_turbo = round(dl_turbo/last_time,5)
                dl_brate = round(dl_brate/last_time,5)
                ul_mcs = round(ul_mcs/last_time,5)
                ul_brate = round(ul_brate/last_time,5)         
                data.append(dl_mcs);data.append(dl_turbo)
                data.append(dl_brate);data.append(ul_mcs);data.append(ul_brate)
                ues[rome_static_far_dir+tr+exp+bs+a] = data
                if str(n_ue) in embb_ues: 
                    embb_ues_data.append(data)
                elif str(n_ue) in mtc_ues:
                    mtc_ues_data.append(data)
                else: 
                    urllc_ues_data.append(data)
            except FileNotFoundError: pass
            try:
                with open (rome_static_medium_dir+tr+exp+bs+a) as e:
                    file_reader_rscm = csv.reader(e, delimiter=',') 
                    f = []
                    data = []
                    time = 0
                    for i in file_reader_rscm:
                        time += 1
                        if time == 1: pass
                        try:
                            i = [i[0]] + [i[7]] + [i[9]] + [i[10]] + [i[13]] + [i[15]]                               
                            i = [float(k) for k in i]             
                            f.append(i)
                        except ValueError: pass
                        except IndexError: pass
                    while len(f[0]) != len(f[-1]):
                        f.remove(f[-1])
                dl_mcs, dl_turbo, dl_brate, ul_mcs, ul_brate = 0,0,0,0,0
                last_time = 0
                for j in range (1,len(f)):
                    time = f[j][0] - f[j-1][0]
                    if np.isinf(f[j-1][1]) or np.isinf(f[j-1][2]) or np.isinf(f[j-1][3]) or np.isinf(f[j-1][4]) or np.isinf(f[j-1][5]): continue 
                    last_time += time
                    dl_mcs += f[j-1][1] * time
                    dl_turbo += f[j-1][2] * time
                    dl_brate += f[j-1][3] * time
                    ul_mcs += f[j-1][4] * time
                    ul_brate += f[j-1][5] * time
                dl_mcs = round(dl_mcs/last_time,5)
                dl_turbo = round(dl_turbo/last_time,5)
                dl_brate = round(dl_brate/last_time,5)
                ul_mcs = round(ul_mcs/last_time,5)
                ul_brate = round(ul_brate/last_time,5)                   
                data.append(dl_mcs);data.append(dl_turbo)
                data.append(dl_brate);data.append(ul_mcs);data.append(ul_brate)
                ues[rome_static_medium_dir+tr+exp+bs+a] = data
                if str(n_ue) in embb_ues: 
                    embb_ues_data.append(data)
                elif str(n_ue) in mtc_ues:
                    mtc_ues_data.append(data)
                else: 
                    urllc_ues_data.append(data)
            except FileNotFoundError: pass
def K_Fold (a,k_fold,b,embb_ues_data,mtc_ues_data,urllc_ues_data):
    rng = np.random.default_rng(seed=b)
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
    
    ue_class,ue_data = [],[]
    for i in embb_ues_data_used:
        ue_data.append(i)
        ue_class.append('embb')
    for j in mtc_ues_data_used:
        ue_data.append(j)
        ue_class.append('mtc')
    for k in urllc_ues_data_used:
        ue_data.append(k)
        ue_class.append('urllc')
    ue_data = np.array(ue_data)
    ue_class = np.array(ue_class)
    ue_data2 = ue_data
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=a)
    accuracy_scores = []
    for train_index, test_index in kf.split(ue_data,ue_class):
        data_train, data_test = ue_data[train_index], ue_data[test_index]
        class_train, class_test = ue_class[train_index], ue_class[test_index]
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
    
print(K_Fold (8,3,1,embb_ues_data,mtc_ues_data,urllc_ues_data))



 