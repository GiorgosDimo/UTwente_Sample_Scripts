import numpy as np
from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import matplotlib as mpl
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
import jenkspy
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# load the data
# m --> an array with the tick_count values and all the features (101) that are scaled. This is the table that will be used for the model 
# all_observations --> is a list of lists with all the unscaled values of the tick_count and features. It also contain id, location and date of each measurment
# headers_list --> is a list with the headers for every column of array m
# combination --> all the indices of the features that are included in m  
path_train =  r"random_FS_nymphs_with_zeros_savitzky_golay.csv"
m, all_observations, headers_list, combination, descale, descale_target = load_stuff(path_train, experiment=0)
all_observations_array=np.array(all_observations)
headers = headers_list[:]
# split the all observations dataset in order to use it for the "find_sites_and_dates" function
all_obsevations_train, all_obsevations_test = train_test_split(all_observations, test_size = 0.3, random_state = 42) 
all_observations_test = all_obsevations_x_test.tolist() 

#the selected features that we will use for the classification
Features = ['ev-1', 'rh-1', 'rh-2', 'rh-3', 'sd-3', 'tmax-4', 'ev-4', 'rh-4', 'ev-5', 'sd-5', 'ev-6', 'ev-7', 
               'rh-7', 'ev-14', 'rh-14', 'tmin-30', 'prec-30', 'rh-30', 'sd-30', 'vp-30', 'tmin-90', 'prec-90', 'rh-90', 'sd-90', 
               'tmin-365', 'tmax-365', 'prec-365', 'ev-365', 'rh-365', 'sd-365', 'min_ndvi', 'range_ndvi', 'min_evi', 'range_evi', 
               'min_ndwi', 'range_ndwi', 'LandCover', 'LandCover_500m', 'LandCover_1km\r\n']

# extract the indices of the selected features
indices = []

for i in Features:
    index = headers.index(i)
    indices.append(index-1)

#Jenks natural breaks for the definition of the clusters/classes
cl_train, cl_test = train_test_split(all_observations_array, test_size = 0.3, random_state = 10) 
breaks = jenkspy.jenks_breaks(cl_train[:,3].astype(float), nb_class=3)
clusters = np.empty(2890, int)

for i in range(0,2890):
    tick_count = all_observations_array[i,3].astype(float)
    if tick_count <= breaks[1]:
        clusters[i] = 0
    elif tick_count > breaks[1] and tick_count <= breaks[2]:
        clusters[i] = 1
    elif tick_count > breaks[2] and tick_count <= breaks[3]:
        clusters[i] = 2
    '''elif tick_count > breaks[3] and tick_count <= breaks[4]:
        clusters[i] = 3
    elif tick_count > breaks[4] and tick_count <= breaks[5]:
        clusters[i] = 4
    elif tick_count > breaks[5] and tick_count <= breaks[6]:
        clusters[i] = 5
    else:
        clusters[i] = 6'''

clustered_m =  np.column_stack([clusters,m])

#perform the classification
Y = clustered_m[:,0]
X = clustered_m[:,2:]
random_st = [10]
for i in random_st:
    #for using the splited dataset
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size = 0.30, random_state = i)#, stratify = Y)
    selected_x_tr = x_tr[:, indices]
    selected_x_te = x_te[:, indices]
   
    #for using the dataset with the oob_score
    selected_X = X[:, indices]
    
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', bootstrap = 'True', oob_score = 'True', random_state = 10)
    #classifier.fit(selected_x_tr, y_tr)
    classifier.fit(selected_X, Y)

    # Predicting the Test set results
    y_pred = classifier.predict(selected_x_te)
    
    #importances = classifier.feature_importances_
    #indices = np.argsort(importances)[::-1]

    from sklearn.metrics import classification_report 
    report = classification_report(y_te, y_pred)
    #print(importances)
    #print(indices)
    print(report)
    print(classifier.oob_score_)
    

    