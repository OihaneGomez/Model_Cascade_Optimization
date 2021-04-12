import os
os.environ['PYTHONHASHSEED'] = '1'
from numpy.random import seed
seed(1)
import random as rn
rn.seed(1)

import warnings
warnings.filterwarnings("ignore")

import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier 
from scipy.stats import kurtosis 
from scipy.stats import skew
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
import timeit
import gc
from IPython import get_ipython
from sklearn.pipeline import Pipeline   
import scipy.signal as sp
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
#kernprof -l -v Time_data_processing_Inference.py >> profiler_regularSampling.txt  
#@profile

#------------------Change these variables:------------------
#Number of signal components 
#(9 for all)
number_initial_components = 9

#Best feature number 
#(Max 486 for XYZ, 54 for X)
best_features_number=486

#Features number model Mn
featM1=10
featM2=50
featMax=best_features_number

#Threshold M 1-2 (threshold 1) and M 2-3 (threshold 2)
Threshold_1 =0.15
Threshold_2=0.40

#Dataset folder relative route
root = './OHM_Dataset_Train_Test/'
#----------------------------------------------------------



#Create activities dictionary 
wrist_class = {'Other':0, 
              'Drink_glass':1, 
               'Drink_bottle':2, 
             }


#Activity label list
wrist_labels_raw = sorted(wrist_class.items(), key=lambda x: x[1])
wrist_labels = [item[0] for item in wrist_labels_raw]
wrist_labels_number = [value[1] for value in wrist_labels_raw]

#Features ranking array containing features expresion for futher computation
array_3Comp=['data_roll.std(axis=0)', 'data_split_roll[2].std(axis=0)', 'data_roll.min(axis=0)', 'data_split_roll[1].std(axis=0)', 'mad(data_y,axis=0)', 'data_y.std(axis=0)', 'mad(data_split_roll[2],axis=0)', 'data_y.var(axis=0)', 'mad(data_roll,axis=0)', 'mad(data_split_roll[1],axis=0)', 'data_pitch.var(axis=0)', 'mad(data_z,axis=0)', 'data_pitch.std(axis=0)', 'data_z.var(axis=0)', 'mad(data_pitch,axis=0)', 'data_z.std(axis=0)', 'data_roll.var(axis=0)', 'mad(data_x,axis=0)', 'data_split_roll[1].var(axis=0)', 'data_x.var(axis=0)', 'data_split_roll[2].var(axis=0)', 'data_x.std(axis=0)', 'data_pitch.max(axis=0)', 'data_split_y[1].var(axis=0)', 'data_split_pitch[2].var(axis=0)', 'data_split_pitch[1].var(axis=0)', 'data_split_roll[2].min(axis=0)', 'data_split_pitch[1].std(axis=0)', 'mad(data_split_pitch[1],axis=0)', 'mad(data_split_pitch[3],axis=0)', 'mad(data_split_y[1],axis=0)', 'data_split_pitch[3].std(axis=0)', 'data_split_y[1].std(axis=0)', 'data_split_pitch[2].std(axis=0)', 'data_split_yaw[0].std(axis=0)', 'data_split_z[1].var(axis=0)', 'data_split_y[2].var(axis=0)', 'mad(data_split_pitch[2],axis=0)', 'data_split_gyroy[3].var(axis=0)', 'data_split_x[4].std(axis=0)', 'data_split_x[4].var(axis=0)', 'np.median(data_split_y[2],axis=0)', 'data_split_y[2].mean(axis=0)', 'data_split_yaw[3].std(axis=0)', 'data_split_roll[3].std(axis=0)', 'mad(data_split_x[4],axis=0)', 'data_pitch.min(axis=0)', 'mad(data_split_y[2],axis=0)', 'data_split_roll[1].min(axis=0)', 'data_split_gyroy[4].std(axis=0)', 'data_split_gyroy[4].var(axis=0)', 'mad(data_split_yaw[0],axis=0)', 'data_split_pitch[3].var(axis=0)', 'data_gyroy.var(axis=0)', 'mad(data_split_z[4],axis=0)', 'mad(data_split_roll[3],axis=0)', 'data_split_z[1].std(axis=0)', 'mad(data_split_z[1],axis=0)', 'mad(data_split_yaw[3],axis=0)', 'mad(data_split_x[0],axis=0)', 'mad(data_split_gyroy[4],axis=0)', 'data_split_y[2].std(axis=0)', 'data_split_pitch[0].std(axis=0)', 'data_split_gyroy[0].var(axis=0)', 'data_split_yaw[1].std(axis=0)', 'data_split_gyroy[0].std(axis=0)', 'data_split_yaw[4].std(axis=0)', 'mad(data_split_pitch[0],axis=0)', 'data_yaw.min(axis=0)', 'data_split_gyroy[3].std(axis=0)', 'data_split_yaw[0].var(axis=0)', 'mad(data_split_yaw[1],axis=0)', 'mad(data_split_pitch[4],axis=0)', 'data_split_pitch[4].std(axis=0)', 'mad(data_split_z[3],axis=0)', 'mad(data_split_gyroy[3],axis=0)', 'data_roll.max(axis=0)', 'data_split_z[2].var(axis=0)', 'data_split_x[0].std(axis=0)', 'data_split_z[0].std(axis=0)', 'data_gyroy.max(axis=0)', 'mad(data_split_yaw[4],axis=0)', 'data_split_z[4].std(axis=0)', 'data_split_gyroz[4].std(axis=0)', 'data_split_gyrox[4].max(axis=0)', 'data_x.max(axis=0)', 'mad(data_split_x[3],axis=0)', 'mad(data_split_gyroy[0],axis=0)', 'mad(data_gyrox,axis=0)', 'mad(data_split_gyroz[4],axis=0)', 'data_split_gyroz[4].var(axis=0)', 'data_split_yaw[1].var(axis=0)', 'mad(data_gyroz,axis=0)', 'data_split_pitch[0].var(axis=0)', 'mad(data_split_y[3],axis=0)', 'data_split_z[0].var(axis=0)', 'mad(data_split_z[0],axis=0)', 'data_split_gyrox[4].std(axis=0)', 'data_gyrox.var(axis=0)', 'data_split_yaw[3].var(axis=0)', 'data_gyrox.std(axis=0)', 'data_split_x[0].var(axis=0)', 'mad(data_split_gyrox[4],axis=0)', 'data_z.max(axis=0)', 'data_gyrox.max(axis=0)', 'data_gyroy.std(axis=0)', 'data_split_x[3].std(axis=0)', 'data_split_roll[2].mean(axis=0)', 'data_split_z[3].std(axis=0)', 'data_split_roll[4].std(axis=0)', 'data_split_pitch[4].var(axis=0)', 'mad(data_split_y[0],axis=0)', 'np.median(data_split_roll[2],axis=0)', 'mad(data_split_z[2],axis=0)', 'data_split_z[2].std(axis=0)', 'data_gyroz.std(axis=0)', 'kurtosis(data_y,axis=0)', 'data_split_z[4].var(axis=0)', 'mad(data_split_roll[4],axis=0)', 'data_split_yaw[3].min(axis=0)', 'mad(data_split_y[4],axis=0)', 'data_split_gyrox[3].std(axis=0)', 'mad(data_split_gyrox[3],axis=0)', 'data_split_x[3].var(axis=0)', 'data_split_y[2].min(axis=0)', 'data_split_gyrox[4].var(axis=0)', 'data_yaw.std(axis=0)', 'data_gyroz.var(axis=0)', 'data_split_y[0].std(axis=0)', 'data_split_x[4].max(axis=0)', 'data_split_roll[3].var(axis=0)', 'data_split_y[4].std(axis=0)', 'data_split_gyroy[0].max(axis=0)', 'data_split_gyroz[0].std(axis=0)', 'data_split_pitch[0].max(axis=0)', 'mad(data_yaw,axis=0)', 'skew(data_roll,axis=0)', 'data_split_y[3].std(axis=0)', 'data_split_gyrox[2].var(axis=0)', 'data_yaw.var(axis=0)', 'mad(data_split_gyrox[2],axis=0)', 'data_split_gyroy[3].max(axis=0)', 'mad(data_split_gyroz[0],axis=0)', 'mad(data_split_x[1],axis=0)', 'data_split_gyroy[4].max(axis=0)', 'data_split_yaw[4].var(axis=0)', 'data_split_roll[0].std(axis=0)', 'data_split_gyrox[3].max(axis=0)', 'data_split_gyroz[4].max(axis=0)', 'kurtosis(data_pitch,axis=0)', 'data_split_gyrox[0].std(axis=0)', 'mad(data_split_gyroz[3],axis=0)', 'data_split_x[1].std(axis=0)', 'data_split_yaw[2].std(axis=0)', 'data_gyroz.max(axis=0)', 'data_split_gyrox[2].std(axis=0)', 'data_split_y[1].min(axis=0)', 'mad(data_gyroy,axis=0)', 'data_split_yaw[0].max(axis=0)', 'data_split_gyroy[1].var(axis=0)', 'mad(data_split_yaw[2],axis=0)', 'mad(data_split_gyroy[2],axis=0)', 'data_split_z[3].var(axis=0)', 'data_split_yaw[0].min(axis=0)', 'data_split_gyroz[3].std(axis=0)', 'data_split_yaw[1].min(axis=0)', 'data_split_z[4].max(axis=0)', 'data_gyroy.min(axis=0)', 'kurtosis(data_split_x[0],axis=0)', 'data_split_gyrox[0].max(axis=0)', 'mad(data_split_gyrox[0],axis=0)', 'data_split_roll[3].min(axis=0)', 'data_split_gyroy[2].std(axis=0)', 'data_split_gyroz[0].max(axis=0)', 'mad(data_split_gyroz[2],axis=0)', 'np.median(data_split_y[1],axis=0)', 'data_split_y[4].var(axis=0)', 'data_split_yaw[4].min(axis=0)', 'data_split_roll[4].var(axis=0)', 'data_split_y[1].mean(axis=0)', 'data_split_yaw[3].max(axis=0)', 'data_split_gyrox[3].var(axis=0)', 'data_split_y[2].max(axis=0)', 'mad(data_split_roll[0],axis=0)', 'data_split_pitch[4].max(axis=0)', 'data_y.mean(axis=0)', 'data_split_x[1].var(axis=0)', 'data_split_gyroy[1].std(axis=0)', 'mad(data_split_gyrox[1],axis=0)', 'data_split_roll[1].mean(axis=0)', 'data_split_gyroz[0].var(axis=0)', 'np.median(data_split_z[2],axis=0)', 'data_split_gyrox[1].std(axis=0)', 'data_split_z[0].max(axis=0)', 'data_split_pitch[3].max(axis=0)', 'data_split_z[2].max(axis=0)', 'data_split_gyroz[2].std(axis=0)', 'kurtosis(data_gyroy,axis=0)', 'data_split_yaw[4].max(axis=0)', 'data_split_gyrox[2].max(axis=0)', 'data_yaw.max(axis=0)', 'np.median(data_split_roll[1],axis=0)', 'data_split_z[2].mean(axis=0)', 'data_roll.mean(axis=0)', 'data_split_y[0].var(axis=0)', 'data_split_yaw[2].min(axis=0)', 'data_split_y[3].var(axis=0)', 'kurtosis(data_x,axis=0)', 'data_split_gyroz[3].max(axis=0)', 'np.median(data_split_y[3],axis=0)', 'data_split_x[2].std(axis=0)', 'mad(data_split_gyroy[1],axis=0)', 'kurtosis(data_split_y[1],axis=0)', 'mad(data_split_x[2],axis=0)', 'mad(data_split_gyroz[1],axis=0)', 'data_split_gyroz[4].min(axis=0)', 'data_split_z[1].max(axis=0)', 'data_split_gyrox[1].var(axis=0)', 'data_split_pitch[1].min(axis=0)', 'data_split_yaw[2].var(axis=0)', 'data_split_x[0].max(axis=0)', 'data_split_y[3].mean(axis=0)', 'data_split_gyrox[0].var(axis=0)', 'data_split_gyroz[3].var(axis=0)', 'data_split_gyroz[2].var(axis=0)', 'data_split_gyroy[0].min(axis=0)', 'data_split_gyroz[1].std(axis=0)', 'data_split_yaw[1].max(axis=0)', 'data_split_pitch[3].min(axis=0)', 'data_split_y[3].min(axis=0)', 'data_split_pitch[2].min(axis=0)', 'kurtosis(data_z,axis=0)', 'kurtosis(data_split_y[3],axis=0)', 'kurtosis(data_split_x[4],axis=0)', 'data_y.min(axis=0)', 'kurtosis(data_split_gyroy[0],axis=0)', 'kurtosis(data_split_y[4],axis=0)', 'np.median(data_y,axis=0)', 'data_split_gyroy[2].var(axis=0)', 'kurtosis(data_split_z[0],axis=0)', 'data_split_x[3].max(axis=0)', 'data_split_x[2].var(axis=0)', 'kurtosis(data_gyrox,axis=0)', 'data_split_gyroz[1].max(axis=0)', 'kurtosis(data_split_gyroz[1],axis=0)', 'kurtosis(data_split_z[4],axis=0)', 'data_split_roll[0].var(axis=0)', 'data_split_gyroz[2].max(axis=0)', 'kurtosis(data_roll,axis=0)', 'data_split_gyrox[0].min(axis=0)', 'data_split_z[3].max(axis=0)', 'data_gyrox.min(axis=0)', 'data_split_roll[0].min(axis=0)', 'data_split_gyroy[4].min(axis=0)', 'kurtosis(data_split_y[0],axis=0)', 'kurtosis(data_split_x[1],axis=0)', 'data_gyroz.min(axis=0)', 'data_split_gyroz[0].min(axis=0)', 'kurtosis(data_split_gyroz[0],axis=0)', 'data_split_pitch[2].max(axis=0)', 'data_split_gyroy[3].min(axis=0)', 'data_split_gyrox[1].max(axis=0)', 'data_split_x[3].min(axis=0)', 'data_z.mean(axis=0)', 'data_split_x[2].max(axis=0)', 'kurtosis(data_split_x[3],axis=0)', 'data_x.min(axis=0)', 'kurtosis(data_split_gyroz[3],axis=0)', 'kurtosis(data_split_yaw[4],axis=0)', 'kurtosis(data_split_gyroy[2],axis=0)', 'data_split_x[1].max(axis=0)', 'kurtosis(data_split_gyroz[4],axis=0)', 'data_split_gyrox[1].min(axis=0)', 'np.median(data_split_z[1],axis=0)', 'data_split_pitch[1].max(axis=0)', 'data_split_pitch[4].min(axis=0)', 'data_split_gyrox[4].min(axis=0)', 'data_split_z[1].mean(axis=0)', 'data_split_gyroy[1].max(axis=0)', 'kurtosis(data_split_gyrox[4],axis=0)', 'data_split_gyroz[1].var(axis=0)', 'kurtosis(data_gyroz,axis=0)', 'kurtosis(data_split_gyrox[1],axis=0)', 'kurtosis(data_split_z[3],axis=0)', 'data_y.max(axis=0)', 'data_split_pitch[0].min(axis=0)', 'data_split_gyroy[1].min(axis=0)', 'data_split_z[0].min(axis=0)', 'kurtosis(data_split_gyrox[0],axis=0)', 'kurtosis(data_split_gyroy[1],axis=0)', 'kurtosis(data_split_y[2],axis=0)', 'data_split_roll[4].min(axis=0)', 'data_split_x[4].min(axis=0)', 'data_split_x[0].min(axis=0)', 'data_split_gyrox[3].min(axis=0)', 'data_split_roll[3].mean(axis=0)', 'data_z.min(axis=0)', 'skew(data_split_y[2],axis=0)', 'skew(data_split_gyroy[1],axis=0)', 'skew(data_y,axis=0)', 'np.median(data_roll,axis=0)', 'data_split_gyroz[3].min(axis=0)', 'np.median(data_split_y[4],axis=0)', 'kurtosis(data_split_roll[1],axis=0)', 'np.median(data_split_y[0],axis=0)', 'np.median(data_z,axis=0)', 'kurtosis(data_split_pitch[0],axis=0)', 'data_split_y[0].mean(axis=0)', 'data_split_yaw[2].max(axis=0)', 'data_split_gyroz[2].min(axis=0)', 'data_split_y[0].min(axis=0)', 'data_split_z[3].mean(axis=0)', 'kurtosis(data_split_gyrox[3],axis=0)', 'kurtosis(data_split_x[2],axis=0)', 'data_split_z[2].min(axis=0)', 'kurtosis(data_split_z[1],axis=0)', 'data_split_gyroy[2].max(axis=0)', 'np.median(data_split_yaw[0],axis=0)', 'np.median(data_split_roll[3],axis=0)', 'data_split_gyroy[2].min(axis=0)', 'np.median(data_split_x[2],axis=0)', 'np.median(data_split_z[3],axis=0)', 'data_split_gyrox[1].mean(axis=0)', 'data_split_y[4].mean(axis=0)', 'data_split_y[4].min(axis=0)', 'skew(data_split_y[0],axis=0)', 'data_split_x[2].mean(axis=0)', 'kurtosis(data_split_gyroz[2],axis=0)', 'data_split_gyrox[2].min(axis=0)', 'data_split_roll[0].mean(axis=0)', 'data_split_y[1].max(axis=0)', 'data_split_z[4].min(axis=0)', 'data_split_gyroz[1].min(axis=0)', 'data_split_x[1].min(axis=0)', 'kurtosis(data_split_roll[2],axis=0)', 'kurtosis(data_split_z[2],axis=0)', 'skew(data_split_y[3],axis=0)', 'np.median(data_split_gyrox[1],axis=0)', 'data_split_roll[2].max(axis=0)', 'kurtosis(data_split_roll[0],axis=0)', 'data_split_z[3].min(axis=0)', 'kurtosis(data_split_roll[4],axis=0)', 'np.median(data_split_pitch[2],axis=0)', 'data_split_roll[3].max(axis=0)', 'kurtosis(data_split_pitch[2],axis=0)', 'data_split_yaw[0].mean(axis=0)', 'data_split_gyrox[3].mean(axis=0)', 'np.median(data_split_roll[0],axis=0)', 'data_split_y[4].max(axis=0)', 'kurtosis(data_split_yaw[0],axis=0)', 'data_split_pitch[2].mean(axis=0)', 'np.median(data_split_pitch[1],axis=0)', 'np.median(data_split_roll[4],axis=0)', 'data_split_gyroz[2].mean(axis=0)', 'data_split_pitch[1].mean(axis=0)', 'data_split_roll[4].mean(axis=0)', 'kurtosis(data_split_yaw[3],axis=0)', 'kurtosis(data_split_gyroy[3],axis=0)', 'np.median(data_gyrox,axis=0)', 'np.median(data_split_gyrox[3],axis=0)', 'data_split_gyroy[1].mean(axis=0)', 'kurtosis(data_split_pitch[1],axis=0)', 'skew(data_split_gyroz[0],axis=0)', 'data_split_gyroy[0].mean(axis=0)', 'data_gyrox.mean(axis=0)', 'data_split_y[3].max(axis=0)', 'kurtosis(data_split_roll[3],axis=0)', 'data_split_roll[1].max(axis=0)', 'kurtosis(data_split_yaw[1],axis=0)', 'kurtosis(data_split_yaw[2],axis=0)', 'data_split_y[0].max(axis=0)', 'kurtosis(data_split_pitch[4],axis=0)', 'np.median(data_split_gyroz[1],axis=0)', 'np.median(data_split_pitch[3],axis=0)', 'data_split_gyroz[1].mean(axis=0)', 'skew(data_split_z[2],axis=0)', 'np.median(data_split_x[1],axis=0)', 'kurtosis(data_yaw,axis=0)', 'data_split_x[1].mean(axis=0)', 'skew(data_pitch,axis=0)', 'data_split_yaw[1].mean(axis=0)', 'skew(data_split_yaw[0],axis=0)', 'np.median(data_split_gyroz[2],axis=0)', 'skew(data_split_y[1],axis=0)', 'data_split_gyrox[4].mean(axis=0)', 'kurtosis(data_split_gyroy[4],axis=0)', 'skew(data_gyroy,axis=0)', 'np.median(data_yaw,axis=0)', 'data_split_pitch[3].mean(axis=0)', 'np.median(data_split_gyroy[0],axis=0)', 'data_split_yaw[2].mean(axis=0)', 'data_split_roll[4].max(axis=0)', 'np.median(data_split_gyrox[2],axis=0)', 'data_split_gyroz[3].mean(axis=0)', 'skew(data_split_gyroy[4],axis=0)', 'data_split_z[1].min(axis=0)', 'skew(data_split_yaw[2],axis=0)', 'np.median(data_split_z[0],axis=0)', 'skew(data_split_roll[1],axis=0)', 'np.median(data_split_x[0],axis=0)', 'data_split_z[0].mean(axis=0)', 'data_split_gyroy[3].mean(axis=0)', 'skew(data_split_gyroy[0],axis=0)', 'skew(data_split_gyrox[3],axis=0)', 'data_split_z[4].mean(axis=0)', 'skew(data_gyrox,axis=0)', 'kurtosis(data_split_gyrox[2],axis=0)', 'data_split_gyroy[4].mean(axis=0)', 'np.median(data_split_yaw[4],axis=0)', 'np.median(data_split_z[4],axis=0)', 'skew(data_split_gyroz[1],axis=0)', 'data_gyroz.mean(axis=0)', 'data_x.mean(axis=0)', 'np.median(data_split_yaw[1],axis=0)', 'skew(data_split_roll[3],axis=0)', 'data_pitch.mean(axis=0)', 'data_split_gyrox[2].mean(axis=0)', 'np.median(data_split_gyroy[4],axis=0)', 'skew(data_split_x[4],axis=0)', 'data_split_pitch[4].mean(axis=0)', 'np.median(data_split_gyroy[1],axis=0)', 'data_split_x[0].mean(axis=0)', 'skew(data_split_gyroy[2],axis=0)', 'skew(data_split_gyrox[1],axis=0)', 'data_split_yaw[4].mean(axis=0)', 'np.median(data_split_gyrox[4],axis=0)', 'skew(data_split_pitch[3],axis=0)', 'kurtosis(data_split_pitch[3],axis=0)', 'data_split_x[2].min(axis=0)', 'np.median(data_gyroz,axis=0)', 'data_split_pitch[0].mean(axis=0)', 'skew(data_split_z[0],axis=0)', 'np.median(data_split_x[3],axis=0)', 'np.median(data_split_gyroy[3],axis=0)', 'skew(data_split_yaw[3],axis=0)', 'np.median(data_split_gyroz[3],axis=0)', 'skew(data_split_pitch[0],axis=0)', 'data_split_x[3].mean(axis=0)', 'skew(data_split_roll[0],axis=0)', 'skew(data_gyroz,axis=0)', 'np.median(data_split_pitch[0],axis=0)', 'data_split_yaw[3].mean(axis=0)', 'skew(data_split_gyroz[2],axis=0)', 'np.median(data_split_yaw[3],axis=0)', 'skew(data_split_pitch[2],axis=0)', 'skew(data_split_roll[2],axis=0)', 'skew(data_split_yaw[1],axis=0)', 'skew(data_x,axis=0)', 'skew(data_split_yaw[4],axis=0)', 'np.median(data_gyroy,axis=0)', 'skew(data_split_z[4],axis=0)', 'data_split_x[4].mean(axis=0)', 'skew(data_split_pitch[4],axis=0)', 'data_split_gyrox[0].mean(axis=0)', 'skew(data_split_z[3],axis=0)', 'np.median(data_split_pitch[4],axis=0)', 'data_split_roll[0].max(axis=0)', 'np.median(data_pitch,axis=0)', 'data_yaw.mean(axis=0)', 'skew(data_split_x[1],axis=0)', 'np.median(data_split_gyroz[0],axis=0)', 'np.median(data_split_gyrox[0],axis=0)', 'data_split_gyroz[0].mean(axis=0)', 'skew(data_split_y[4],axis=0)', 'np.median(data_split_x[4],axis=0)', 'data_gyroy.mean(axis=0)', 'skew(data_split_z[1],axis=0)', 'skew(data_split_roll[4],axis=0)', 'data_split_gyroy[2].mean(axis=0)', 'skew(data_split_gyrox[0],axis=0)', 'skew(data_split_pitch[1],axis=0)', 'skew(data_split_x[2],axis=0)', 'skew(data_split_gyrox[4],axis=0)', 'skew(data_split_gyrox[2],axis=0)', 'np.median(data_split_yaw[2],axis=0)', 'skew(data_z,axis=0)', 'np.median(data_x,axis=0)', 'skew(data_yaw,axis=0)', 'skew(data_split_gyroz[4],axis=0)', 'skew(data_split_x[3],axis=0)', 'skew(data_split_gyroy[3],axis=0)', 'np.median(data_split_gyroy[2],axis=0)', 'np.median(data_split_gyroz[4],axis=0)', 'skew(data_split_gyroz[3],axis=0)', 'skew(data_split_x[0],axis=0)', 'data_split_gyroz[4].mean(axis=0)']
#array_1Comp=['data_roll.std(axis=0)', 'mad(data_roll,axis=0)', 'mad(data_y,axis=0)', 'data_y.std(axis=0)', 'data_pitch.std(axis=0)', 'data_pitch.var(axis=0)', 'data_split_roll[2].std(axis=0)', 'mad(data_pitch,axis=0)', 'data_roll.min(axis=0)', 'data_roll.var(axis=0)', 'mad(data_split_roll[2],axis=0)', 'data_y.var(axis=0)', 'data_split_roll[1].std(axis=0)', 'data_pitch.max(axis=0)', 'data_split_pitch[3].std(axis=0)', 'mad(data_split_roll[1],axis=0)', 'mad(data_split_pitch[3],axis=0)', 'data_split_roll[2].var(axis=0)', 'data_split_roll[2].min(axis=0)', 'data_split_roll[3].std(axis=0)', 'data_split_pitch[1].std(axis=0)', 'mad(data_split_pitch[1],axis=0)', 'data_split_roll[1].var(axis=0)', 'data_split_y[3].std(axis=0)', 'mad(data_split_y[3],axis=0)', 'data_split_pitch[1].var(axis=0)', 'mad(data_split_roll[3],axis=0)', 'data_split_y[2].var(axis=0)', 'np.median(data_split_y[2],axis=0)', 'mad(data_split_y[2],axis=0)', 'data_split_pitch[3].var(axis=0)', 'data_split_y[2].mean(axis=0)', 'mad(data_split_y[1],axis=0)', 'data_split_y[3].var(axis=0)', 'data_split_y[2].min(axis=0)', 'data_split_y[1].std(axis=0)', 'data_pitch.min(axis=0)', 'data_split_y[2].std(axis=0)', 'data_split_y[1].var(axis=0)', 'data_split_y[0].std(axis=0)', 'data_split_pitch[2].var(axis=0)', 'data_split_pitch[4].std(axis=0)', 'data_split_pitch[3].max(axis=0)', 'mad(data_split_pitch[4],axis=0)', 'data_split_pitch[2].std(axis=0)', 'data_split_pitch[0].std(axis=0)', 'data_split_roll[1].min(axis=0)', 'data_split_roll[3].var(axis=0)', 'mad(data_split_pitch[0],axis=0)', 'mad(data_split_pitch[2],axis=0)', 'mad(data_split_y[0],axis=0)', 'kurtosis(data_y,axis=0)', 'data_split_y[4].std(axis=0)', 'kurtosis(data_pitch,axis=0)', 'skew(data_roll,axis=0)', 'data_split_y[0].var(axis=0)', 'data_roll.max(axis=0)', 'data_split_pitch[0].var(axis=0)', 'data_split_roll[4].std(axis=0)', 'mad(data_split_y[4],axis=0)', 'data_split_roll[2].mean(axis=0)', 'data_split_pitch[4].var(axis=0)', 'mad(data_split_roll[4],axis=0)', 'np.median(data_split_roll[2],axis=0)', 'data_y.min(axis=0)', 'kurtosis(data_split_y[4],axis=0)', 'data_split_y[1].min(axis=0)', 'mad(data_split_roll[0],axis=0)', 'data_split_y[4].var(axis=0)', 'data_split_pitch[3].min(axis=0)', 'data_split_roll[3].min(axis=0)', 'data_split_pitch[0].max(axis=0)', 'data_split_y[2].max(axis=0)', 'data_split_roll[0].std(axis=0)', 'kurtosis(data_split_pitch[2],axis=0)', 'kurtosis(data_split_y[1],axis=0)', 'np.median(data_split_y[1],axis=0)', 'data_split_y[1].mean(axis=0)', 'data_y.mean(axis=0)', 'kurtosis(data_split_y[3],axis=0)', 'data_split_roll[4].var(axis=0)', 'kurtosis(data_split_y[0],axis=0)', 'data_split_pitch[4].max(axis=0)', 'skew(data_y,axis=0)', 'kurtosis(data_split_pitch[1],axis=0)', 'data_split_y[3].min(axis=0)', 'data_roll.mean(axis=0)', 'kurtosis(data_split_pitch[0],axis=0)', 'data_split_pitch[1].max(axis=0)', 'data_split_pitch[2].max(axis=0)', 'kurtosis(data_split_pitch[4],axis=0)', 'kurtosis(data_split_y[2],axis=0)', 'data_split_pitch[1].min(axis=0)', 'data_split_roll[1].mean(axis=0)', 'kurtosis(data_split_roll[0],axis=0)', 'data_split_y[3].mean(axis=0)', 'kurtosis(data_split_roll[2],axis=0)', 'np.median(data_split_y[3],axis=0)', 'np.median(data_split_roll[1],axis=0)', 'data_split_y[4].min(axis=0)', 'data_split_pitch[2].min(axis=0)', 'kurtosis(data_split_roll[4],axis=0)', 'data_split_pitch[4].min(axis=0)', 'np.median(data_y,axis=0)', 'skew(data_split_y[2],axis=0)', 'data_split_y[0].min(axis=0)', 'skew(data_split_y[0],axis=0)', 'data_split_roll[3].max(axis=0)', 'data_split_roll[4].min(axis=0)', 'kurtosis(data_roll,axis=0)', 'kurtosis(data_split_roll[1],axis=0)', 'data_split_roll[3].mean(axis=0)', 'data_split_roll[0].min(axis=0)', 'np.median(data_split_y[4],axis=0)', 'data_split_y[4].mean(axis=0)', 'data_y.max(axis=0)', 'kurtosis(data_split_pitch[3],axis=0)', 'np.median(data_split_roll[0],axis=0)', 'np.median(data_split_y[0],axis=0)', 'data_split_pitch[0].min(axis=0)', 'data_split_roll[0].mean(axis=0)', 'data_split_y[0].mean(axis=0)', 'np.median(data_split_roll[3],axis=0)', 'np.median(data_roll,axis=0)', 'skew(data_split_roll[2],axis=0)', 'data_split_y[0].max(axis=0)', 'data_pitch.mean(axis=0)', 'skew(data_split_roll[1],axis=0)', 'data_split_roll[1].max(axis=0)', 'np.median(data_split_roll[4],axis=0)', 'skew(data_split_roll[3],axis=0)', 'data_split_roll[4].mean(axis=0)', 'data_split_pitch[2].mean(axis=0)', 'data_split_pitch[3].mean(axis=0)', 'np.median(data_split_pitch[2],axis=0)', 'skew(data_split_roll[4],axis=0)', 'data_split_roll[0].var(axis=0)', 'skew(data_split_pitch[2],axis=0)', 'data_split_y[1].max(axis=0)', 'skew(data_split_y[1],axis=0)', 'skew(data_split_y[3],axis=0)', 'np.median(data_split_pitch[3],axis=0)', 'data_split_pitch[1].mean(axis=0)', 'np.median(data_split_pitch[1],axis=0)', 'data_split_y[4].max(axis=0)', 'data_split_pitch[0].mean(axis=0)', 'data_split_pitch[4].mean(axis=0)', 'np.median(data_split_pitch[4],axis=0)', 'np.median(data_pitch,axis=0)', 'skew(data_pitch,axis=0)', 'skew(data_split_y[4],axis=0)', 'np.median(data_split_pitch[0],axis=0)', 'skew(data_split_pitch[4],axis=0)', 'data_split_y[3].max(axis=0)', 'skew(data_split_pitch[3],axis=0)', 'skew(data_split_pitch[0],axis=0)', 'skew(data_split_roll[0],axis=0)', 'skew(data_split_pitch[1],axis=0)', 'data_split_roll[0].max(axis=0)', 'kurtosis(data_split_roll[3],axis=0)', 'data_split_roll[4].max(axis=0)', 'data_split_roll[2].max(axis=0)']

#Binarize pred output to have a binary array
class LabelBinarizer2:

    def __init__(self):
        self.lb = LabelBinarizer()

    def fit(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit X using the LabelBinarizer object
        self.lb.fit(X)
        # Save the classes
        self.classes_ = self.lb.classes_

    def fit_transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit + transform X using the LabelBinarizer object
        Xlb = self.lb.fit_transform(X)
        # Save the classes
        self.classes_ = self.lb.classes_
        if len(self.classes_) == 2:
            Xlb = np.hstack((Xlb, 1 - Xlb))
        return Xlb

    def transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Transform X using the LabelBinarizer object
        Xlb = self.lb.transform(X)
        if len(self.classes_) == 2:
            Xlb = np.hstack((Xlb, 1 - Xlb))
        return Xlb

    def inverse_transform(self, Xlb):
        # Convert Xlb to array
        Xlb = np.array(Xlb)
        if len(self.classes_) == 2:
            X = self.lb.inverse_transform(Xlb[:, 0])
        else:
            X = self.lb.inverse_transform(Xlb)
        return X


#Median Absolute Deviation
def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


#Meadian Filter
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))       



#Import all the txt for every class, calculate its features and append it in a matrix for Train
def gatherTrain(class_dict,features):
    df = []
    for c in class_dict.keys():
        f = glob.glob(root +"TRAIN/" + c + '/*') #Get all files 
        d = pd.DataFrame(reformat(f, cls=c,features=features)) #Pandas dataframe for reformat funtion features
        df.append(d)
    return pd.concat(df)

#Import all the txt for every class, calculate its features and append it in a matrix for Test
def gatherTest(class_dict,features):
    df = []
    for c in class_dict.keys():
        f = glob.glob(root +"TEST/" + c + '/*') #Get all files 
        d = pd.DataFrame(reformat(f, cls=c,features=features)) #Pandas dataframe for reformat funtion features
        df.append(d)
    return pd.concat(df)


#Import all file names for Test
def gatherTest_names(class_dict):
    files=[]
    for c in class_dict.keys():
        f = glob.glob(root +"TEST/"+ c + '/*') #Get all files 
        files.append(f)
    return files



def features_calculation_pred(data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw,features):

    
	#Features Calculation

    appended_before=array_3Comp

    #Create initial_features_matrix
    appended_features_split=[]
    appended_features=[]

    features_ini=features[0]
    features_fin=features[-1]

    for i in range (features_ini, features_fin):
        appended_features_before = eval(appended_before[i])
        appended_features.append(appended_features_before[0])
        appended_features_before=[] 

    return appended_features



#Obtain Data values for every secuence/instance
def reformat(files, cls, features):
    appended_features_all=[]
    appended_features_df = pd.DataFrame()
    for f in files:
       #Read every txt file (Number of row acc_x, acc_Y, acc_Z, gyro_X, gyro_Y, gyro_Z, pitch, roll, yaw)
        data = pd.read_csv(f, sep=',', header=None, names=['x', 'y', 'z', 'gyrox' , 'gyroy', 'gyroz', 'pitch', 'roll', 'yaw']) 

        
        df_x = data.iloc[:,0:1]
        df_y = data.iloc[:,1:2]
        df_z = data.iloc[:,2:3]
    
        df_gyrox = data.iloc[:,3:4]
        df_gyroy = data.iloc[:,4:5]
        df_gyroz = data.iloc[:,5:6]
        
        df_pitch = data.iloc[:,6:7]
        df_roll = data.iloc[:,7:8]
        df_yaw = data.iloc[:,8:9]
        
        
        
        #Median filtering
        x = np.median(strided_app(df_x.values.flatten(), 3,1),axis=1)
        y = np.median(strided_app(df_y.values.flatten(), 3,1),axis=1)
        z = np.median(strided_app(df_z.values.flatten(), 3,1),axis=1)
        
        gyrox = np.median(strided_app(df_gyrox.values.flatten(), 3,1),axis=1)
        gyroy = np.median(strided_app(df_gyroy.values.flatten(), 3,1),axis=1)
        gyroz = np.median(strided_app(df_gyroz.values.flatten(), 3,1),axis=1)
        
        pitch = np.median(strided_app(df_pitch.values.flatten(), 3,1),axis=1)
        roll = np.median(strided_app(df_roll.values.flatten(), 3,1),axis=1)
        yaw = np.median(strided_app(df_yaw.values.flatten(), 3,1),axis=1)
        
        df_x = pd.DataFrame(x, columns=['x'])
        df_y = pd.DataFrame(y, columns=['y'])
        df_z = pd.DataFrame(z, columns=['z'])

        df_gyrox = pd.DataFrame(gyrox, columns=['gyrox'])
        df_gyroy = pd.DataFrame(gyroy, columns=['gyroy'])
        df_gyroz = pd.DataFrame(gyroz, columns=['gyroz'])

        df_pitch = pd.DataFrame(pitch, columns=['pitch'])
        df_roll = pd.DataFrame(roll, columns=['roll'])
        df_yaw = pd.DataFrame(yaw, columns=['yaw'])


        data_x = df_x.values
        data_y = df_y.values
        data_z = df_z.values
        
        data_gyrox = df_gyrox.values
        data_gyroy = df_gyroy.values
        data_gyroz = df_gyroz.values

        data_pitch = df_pitch.values
        data_roll = df_roll.values
        data_yaw = df_yaw.values
        
        
        
        #Split selected component of every secuence in X segments
        split_index=5#Number of segments
        data_split_x=np.array_split(data_x, split_index)
        data_split_y=np.array_split(data_y, split_index)
        data_split_z=np.array_split(data_z, split_index)
        
        data_split_gyrox=np.array_split(data_gyrox, split_index)
        data_split_gyroy=np.array_split(data_gyroy, split_index)
        data_split_gyroz=np.array_split(data_gyroz, split_index)

        data_split_pitch=np.array_split(data_pitch, split_index)
        data_split_roll=np.array_split(data_roll, split_index)
        data_split_yaw=np.array_split(data_yaw, split_index)
        appended_features=features_calculation(data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw,features)

        appended_features_all.append(appended_features)

    appended_features_df = pd.DataFrame(appended_features_all) 

    appended_features_df[-1]= wrist_class[cls]

    return appended_features_df


#Calculate features
def features_calculation(data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw,features):

    appended_before= array_3Comp

    #Create initial_features_matrix
    appended_features_split=[]
    appended_features=[]

    features_ini=features[0]
    features_fin=features[-1]

    for i in range (features_ini, features_fin):
        appended_features_before = eval(appended_before[i])
        appended_features.append(appended_features_before[0])
        appended_features_before=[] 

    return appended_features





#Obtain Data values for every secuence/instance
def reformat_OnebyOne(files):
    
    data = pd.read_csv(files, sep=',', header=None, names=['x', 'y', 'z', 'gyrox' , 'gyroy', 'gyroz', 'pitch', 'roll', 'yaw']) 
    
    df_x = data.iloc[:,0:1]
    df_y = data.iloc[:,1:2]
    df_z = data.iloc[:,2:3]

    df_gyrox = data.iloc[:,3:4]
    df_gyroy = data.iloc[:,4:5]
    df_gyroz = data.iloc[:,5:6]
    
    df_pitch = data.iloc[:,6:7]
    df_roll = data.iloc[:,7:8]
    df_yaw = data.iloc[:,8:9]
    
    
    
    #Median filtering
    x = np.median(strided_app(df_x.values.flatten(), 3,1),axis=1)
    y = np.median(strided_app(df_y.values.flatten(), 3,1),axis=1)
    z = np.median(strided_app(df_z.values.flatten(), 3,1),axis=1)
    
    gyrox = np.median(strided_app(df_gyrox.values.flatten(), 3,1),axis=1)
    gyroy = np.median(strided_app(df_gyroy.values.flatten(), 3,1),axis=1)
    gyroz = np.median(strided_app(df_gyroz.values.flatten(), 3,1),axis=1)
    
    pitch = np.median(strided_app(df_pitch.values.flatten(), 3,1),axis=1)
    roll = np.median(strided_app(df_roll.values.flatten(), 3,1),axis=1)
    yaw = np.median(strided_app(df_yaw.values.flatten(), 3,1),axis=1)
    
    df_x = pd.DataFrame(x, columns=['x'])
    df_y = pd.DataFrame(y, columns=['y'])
    df_z = pd.DataFrame(z, columns=['z'])

    df_gyrox = pd.DataFrame(gyrox, columns=['gyrox'])
    df_gyroy = pd.DataFrame(gyroy, columns=['gyroy'])
    df_gyroz = pd.DataFrame(gyroz, columns=['gyroz'])

    df_pitch = pd.DataFrame(pitch, columns=['pitch'])
    df_roll = pd.DataFrame(roll, columns=['roll'])
    df_yaw = pd.DataFrame(yaw, columns=['yaw'])


    data_x = df_x.values
    data_y = df_y.values
    data_z = df_z.values
    
    data_gyrox = df_gyrox.values
    data_gyroy = df_gyroy.values
    data_gyroz = df_gyroz.values

    data_pitch = df_pitch.values
    data_roll = df_roll.values
    data_yaw = df_yaw.values
    
    
    
    #Split selected component of every secuence in X segments
    split_index=5#Number of segments
    data_split_x=np.array_split(data_x, split_index)
    data_split_y=np.array_split(data_y, split_index)
    data_split_z=np.array_split(data_z, split_index)
    
    data_split_gyrox=np.array_split(data_gyrox, split_index)
    data_split_gyroy=np.array_split(data_gyroy, split_index)
    data_split_gyroz=np.array_split(data_gyroz, split_index)

    data_split_pitch=np.array_split(data_pitch, split_index)
    data_split_roll=np.array_split(data_roll, split_index)
    data_split_yaw=np.array_split(data_yaw, split_index)

    return data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw


#Train data preprocessing steps

def train_process (class_, features,model=1): 
    #Measure data processing times for inference
    wrist_df = gatherTrain(class_, features=features)
    wrist_Y = np.asarray(wrist_df.iloc[:,-1])
    num_features=len(wrist_df.columns)-1
    wrist_X = np.asarray(wrist_df.iloc[:,:num_features])
    wrist_X_df = pd.DataFrame(wrist_X, columns=(wrist_df.iloc[:,:num_features]).columns)
    return wrist_X_df,wrist_Y


def prediction_process (class_,features,model=1): 
	#Measure data processing times for inference
    wrist_df_test = gatherTest(class_, features=features)
    wrist_Y_test = np.asarray(wrist_df_test.iloc[:,-1])
    num_features=len(wrist_df_test.columns)-1
    wrist_X_test = np.asarray(wrist_df_test.iloc[:,:num_features])
    wrist_X_df_test = pd.DataFrame(wrist_X_test, columns=(wrist_df_test.iloc[:,:num_features]).columns)
    return  wrist_X_df_test,wrist_Y_test

#Multiclass log loss 
def multiclass_log_loss(y_true, y_pred,eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred)).sum(axis=1)


i=best_features_number-1
print ("Number of features: "+str(i+1))


time_avg_array=[]
time_std_array=[]
n=0

i=best_features_number-1




n=0

f1_score_array_LG_pre, f1_score_array_RF_pre, f1_score_array_KNN_pre, f1_score_array_NB_pre, f1_score_array_SVM_pre,  f1_score_array_MLP_pre = [], [], [], [], [], []
f1_score_array_LG, f1_score_array_RF, f1_score_array_KNN, f1_score_array_NB, f1_score_array_SVM, f1_score_array_MLP = [], [], [], [], [], []

f1_score_array_LG_pre_std, f1_score_array_RF_pre_std, f1_score_array_KNN_pre_std, f1_score_array_NB_pre_std, f1_score_array_SVM_pre_std, f1_score_array_MLP_pre_std = [], [], [], [], [], []
f1_score_array_LG_std, f1_score_array_RF_std, f1_score_array_KNN_std, f1_score_array_NB_std, f1_score_array_SVM_std, f1_score_array_MLP_std = [], [], [], [], [], []



#Classifiers array (repeated to avoid models being overwritten)
alg_array0 = [LogisticRegression(), 
            RandomForestClassifier(verbose= 0,n_estimators= 100,random_state= n),
            KNeighborsClassifier(n_neighbors=3),
            GaussianNB(), 
            svm.SVC(kernel='linear', C=64.0, probability=True, random_state = n),
            MLPClassifier(hidden_layer_sizes=(16, 16, 16), max_iter=1000,random_state = n)]

alg_array1 = [LogisticRegression(), 
            RandomForestClassifier(verbose= 0,n_estimators= 100,random_state= n),
            KNeighborsClassifier(n_neighbors=3),
            GaussianNB(), 
            svm.SVC(kernel='linear', C=64.0, probability=True, random_state = n),
            MLPClassifier(hidden_layer_sizes=(16, 16, 16), max_iter=1000,random_state = n)]

alg_array2 = [LogisticRegression(), 
            RandomForestClassifier(verbose= 0,n_estimators= 100,random_state= n),
            KNeighborsClassifier(n_neighbors=3),
            GaussianNB(), 
            svm.SVC(kernel='linear', C=64.0, probability=True, random_state = n),
            MLPClassifier(hidden_layer_sizes=(16, 16, 16), max_iter=1000,random_state = n)]
alg_array3 = [LogisticRegression(), 
            RandomForestClassifier(verbose= 0,n_estimators= 100,random_state= n),
            KNeighborsClassifier(n_neighbors=3),
            GaussianNB(), 
            svm.SVC(kernel='linear', C=64.0, probability=True, random_state = n),
            MLPClassifier(hidden_layer_sizes=(16, 16, 16), max_iter=1000,random_state = n)]

#Classifiers Names
alg_array_names  = ['LG', 'RF', 'KNN', 'NB','SVM', 'MLP']

#F1 score 
scoring = {'F1_score': 'f1_macro'}

#To save mean and std timming resulst 
time_dict = {"LG": f1_score_array_LG, 
                "RF": f1_score_array_RF,
                "KNN": f1_score_array_KNN,
                "NB": f1_score_array_NB,
                "SVM": f1_score_array_SVM,
                "MLP": f1_score_array_MLP,
                }


array_dict_std = {"LG": f1_score_array_LG_std, 
        "RF": f1_score_array_RF_std,
        "KNN": f1_score_array_KNN_std,
        "NB": f1_score_array_NB_std,
        "SVM": f1_score_array_SVM_std,
        "MLP": f1_score_array_MLP_std,
        }






#Perform experiments for all the included algorithms
for item0, item1 ,item2, item3, item_name in zip(alg_array0,alg_array1, alg_array2,alg_array3,alg_array_names): 

    print("--------------"+str(item_name))

    #Train models
    #Reference model model
    features_all_array=[0,featMax]
    model=item0
    X_train,Y_train = train_process(wrist_class,features_all_array,model=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train=min_max_scaler.fit_transform(X_train)

    # ------------------------------------MODEL 1 
    n_features_1=[0,featM1]
    model_1=item1
    X_train_1,Y_train_1 = train_process(wrist_class,n_features_1,model=1)
    min_max_scaler_1 = preprocessing.MinMaxScaler()
    X_train_1=min_max_scaler_1.fit_transform(X_train_1)
    model_1.fit(X_train_1,Y_train_1)

    # --------------------------------------MODEL 2
    n_features_2=[featM1,featM2]
    features_array_2=[10,50] #needed features are computed
    model_2=item2
    X_train_2,Y_train_2 = train_process(wrist_class,features_array_2,model=1)
    min_max_scaler_2 = preprocessing.MinMaxScaler()
    X_train_2=min_max_scaler_2.fit_transform(X_train_2)  #Adding features 
    X_train_2=np.append(X_train_1,X_train_2,axis=1)
    model_2.fit(X_train_2,Y_train)
    
    # ---------------------------------------MODEL 3
    n_features_3=[featM2,featMax]
    features_ini=n_features_3[0]
    features_fin=n_features_3[-1]
    model_3=item3
    X_train_3,Y_train_3 = train_process(wrist_class,n_features_3,model=1)
    min_max_scaler_3 = preprocessing.MinMaxScaler()
    X_train_3=min_max_scaler_3.fit_transform(X_train_3)
    X_train_3=np.append(X_train_2,X_train_3,axis=1)
    model_3.fit(X_train_3,Y_train_3)
    model.fit(X_train,Y_train)



    #Obtain test data
    X_test,Y_test=prediction_process(wrist_class,features_all_array,model=1)
    files_test=gatherTest_names(wrist_class)


    i=0
    z=0
    j=0
    count_classif_m1=0
    count_classif_m2=0
    count_classif_m3=0
    count_errors_m1=0
    count_errors_m2=0
    count_errors_m3=0

    #INFERENCE: perform classification process file by file to see the performance of the model 
    for file_ in files_test:
        for f_ in file_:
            #process data and calculate features for model 1
            data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw=reformat_OnebyOne(f_) 
            X_test1=features_calculation_pred(data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,
                                              data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw,n_features_1)
            X_test1= min_max_scaler_1.transform([X_test1])
            #obtain proba model 1
            proba1=model_1.predict_proba(X_test1)
            # Obtain pred model 1
            pred1=np.argmax(proba1, axis=1)
            pred_binarized1=label_binarize(pred1, wrist_labels_number)
            #Obtain log loss model 1
            ll_pred1 = multiclass_log_loss(pred_binarized1, proba1)
            #Check log loss (confidence requirements)
            if ll_pred1 < Threshold_1: #If confidence passes the threshold the instance is classified with this model
                count_classif_m1=count_classif_m1+1
                if pred1[0]!=Y_test[z]:
                    count_errors_m1=count_errors_m1+1
            
            else: #If confidence does not pass the threshold the instance continues the cascade
                #process data and calculate remaining features for model 2
                data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw=reformat_OnebyOne(f_) 
                X_test2=features_calculation_pred(data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,
                                                 data_split_z,data_split_gyrox,data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw,n_features_2)
                X_test2= min_max_scaler_2.transform([X_test2])
                # append features with calculated in model 1
                X_test2=np.hstack((X_test1,X_test2)) 
                #obtain proba model 2
                proba2=model_2.predict_proba(X_test2)
                #Obtain log loss model 2
                pred2=np.argmax(proba2, axis=1)
                pred_binarized2=label_binarize(pred2, wrist_labels_number)
                #Obtain log loss model 2
                ll_pred2 = multiclass_log_loss(pred_binarized2, proba2)
                if ll_pred2 < Threshold_2:#If confidence passes the threshold the instance is classified with this model
                    count_classif_m2=count_classif_m2+1
                    if pred2!=Y_test[z]:
                        count_errors_m2=count_errors_m2+1

                else:#If confidence does not pass the threshold the instance continues the cascade
                    count_classif_m3=count_classif_m3+1
                    #process data and calculate remaining features for model 3
                    X_test3=features_calculation_pred(data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,data_split_gyroy,
                                                      data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw,n_features_3)
                    X_test3= min_max_scaler_3.transform([X_test3])
                    # append features with calculated in model 2
                    X_test3=np.hstack((X_test2,X_test3))
                    pred3=model_3.predict(X_test3)
                    if pred3[0]!=Y_test[z]:
                        count_errors_m3=count_errors_m3+1
            
            z=z+1
    #print report
    print("N_errores 1= "+str(count_errors_m1))
    print("N_clasificadas 1= "+str(count_classif_m1))

    print("N_errores 2= "+str(count_errors_m2))
    print("N_clasificadas 2= "+str(count_classif_m2))
    
    print("N_classif 3= "+str(count_classif_m3))
    print("N_errores 3= "+str(count_errors_m3))
    
    print("N_classif total= "+str(count_classif_m1+count_classif_m2+count_classif_m3))
    print("N_errores total= "+str(count_errors_m1+count_errors_m2+count_errors_m3))

    #Perform timming loop  3 repetitions of 10 times loop
    timer = get_ipython().run_cell_magic('timeit', '-r 3 -n 10 -o    ', 'for file_ in files_test:\n    for f_ in file_:\n\n        data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw=reformat_OnebyOne(f_) \n        X_test1=features_calculation_pred(data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,\n                                          data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw,n_features_1)\n        X_test1= min_max_scaler_1.transform([X_test1])\n        proba1=model_1.predict_proba(X_test1)\n        pred1=np.argmax(proba1, axis=1)\n        pred_binarized1=label_binarize(pred1, wrist_labels_number)\n        ll_pred1 = multiclass_log_loss(pred_binarized1, proba1)\n        if ll_pred1 > Threshold_1:\n            data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw=reformat_OnebyOne(f_) \n            X_test2=features_calculation_pred(data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,\n                                             data_split_z,data_split_gyrox,data_split_gyroy,data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw,n_features_2)\n            X_test2= min_max_scaler_2.transform([X_test2])\n            X_test2=np.hstack((X_test1,X_test2))\n            proba2=model_2.predict_proba(X_test2)\n            pred2=np.argmax(proba2, axis=1)\n            pred_binarized2=label_binarize(pred2, wrist_labels_number)\n            ll_pred2 = multiclass_log_loss(pred_binarized2, proba2)\n            if ll_pred2 > Threshold_2:\n                X_test3=features_calculation_pred(data_x,data_y,data_z,data_gyrox,data_gyroy,data_gyroz,data_pitch,data_roll,data_yaw,data_split_x,data_split_y,data_split_z,data_split_gyrox,data_split_gyroy,\n                                                  data_split_gyroz,data_split_pitch,data_split_roll,data_split_yaw,n_features_3)\n                X_test3= min_max_scaler_3.transform([X_test3])\n                X_test3=np.hstack((X_test2,X_test3))\n                pred3=model_3.predict(X_test3)')

    #Obtain average and standard deviation values
    time_std=(np.std(timer.timings)*1000)
    timer_avg = (np.mean(timer.timings)*1000)
    time_dict[item_name].append(timer_avg)
    array_dict_std[item_name].append(time_std)

    
    
    #Save data in .txt file, mean values
    get_ipython().run_cell_magic('capture', 'cap --no-stderr', 'print (time_dict)')
    with open('Prediction_task_times_3Model_MEAN.txt', 'w') as f:
        f.write(cap.stdout)    

    #Save data in .txt file, std values
    get_ipython().run_cell_magic('capture', 'cap ', 'print(array_dict_std)')
    with open('Prediction_task_times_3Model_STD.txt', 'w') as f:
        f.write(cap.stdout) 



  


    

 


