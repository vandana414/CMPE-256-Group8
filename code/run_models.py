import sqlite3
import pandas as pd
from pandas_ml import ConfusionMatrix
import numpy as np
from sklearn import preprocessing
from collections import Counter
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
import sys
sys.path.append("../")
from preprocessing_scripts.readdata import read_sql
from preprocessing_scripts.process_data import process_df
# from preprocessing_scripts.labels import labels
# from NeuralNetwork import NNModel
# from LogisticRegression import LRModel
from NaiveBayes import NBModel 
from DecisionTree import DecTreeModel
from RandomForests import RandomFModel
from SVM import SVMModel
from Bagging import BaggingModel
from MergeLabels import MergeLabel
import sys
sys.stdout = open('../data/output.txt','wt')

def main_p():
    print("Running ......")
    
    # DecisionTree and NaiveBayes Algorithms
    firedata = read_sql.read_data()
    fire = process_df.preprocess(firedata)

    X = fire[['FIRE_YEAR','DAY_OF_WEEK','MONTH','STATE', 'LATITUDE','LONGITUDE','FIRE_SIZE']]
    y = fire['STAT_CAUSE_DESCR']
    print("Fire Cause Model and Prediction")
    NBModel.run_NB(X, y, 0.2)
    DecTreeModel.run_tree(X,y,0.2)

    X = firedata[['FIRE_YEAR','DAY_OF_WEEK','MONTH','STATE', 'LATITUDE','LONGITUDE','STAT_CAUSE_DESCR']]
    y = firedata['FIRE_SIZE_CLASS']
    print("Running ......")
    print("Fire Class Prediction")
    NBModel.run_NB(X, y, 0.2)
    DecTreeModel.run_tree(X,y,0.2)
    
    
    #SVM and Bagging Classifier
    firedata = read_sql.read_data()
    fire = process_df.preprocess(firedata)

    X = fire[['FIRE_YEAR','DAY_OF_WEEK','MONTH','STATE', 'LATITUDE','LONGITUDE','FIRE_SIZE']]
    y = fire['STAT_CAUSE_DESCR']
    print("Fire Cause Model and Prediction")
    SVMModel.run_SVM(X, y, 0.20)
    BaggingModel.run_Bagging(X, y, 0.20)

    X = firedata[['FIRE_YEAR','DAY_OF_WEEK','MONTH','STATE', 'LATITUDE','LONGITUDE','STAT_CAUSE_DESCR']]
    y = firedata['FIRE_SIZE_CLASS']
    print("Running ......")
    print("Fire Class Prediction")
    SVMModel.run_SVM(X, y, 0.20)
    BaggingModel.run_Bagging(X, y, 0.20)


    #Random Forests Classifier
    firedata = read_sql.read_data()
    firedata_copy = firedata.copy()
    firedata = process_df.preprocess(firedata)
    firedata['LABEL'] = firedata_copy['STAT_CAUSE_DESCR'].apply(lambda x: MergeLabel.set_label(x))
    firedata['DISCOVERY_DATE'] = firedata_copy['DISCOVERY_DATE'].copy()
    firedata = firedata.reindex(columns=['SOURCE_REPORTING_UNIT_NAME', 'STATE', 'LATITUDE', 'LONGITUDE', 'FIRE_SIZE_CLASS', 'FIRE_YEAR', 'DISCOVERY_DATE','STAT_CAUSE_DESCR','CONT_DATE','FIRE_SIZE','LABEL','DAY_OF_WEEK','MONTH'])

    X = firedata[['FIRE_YEAR','DAY_OF_WEEK','DISCOVERY_DATE','MONTH','STATE', 'LATITUDE','LONGITUDE','FIRE_SIZE']]
    y = firedata['LABEL']
    print("Running ......")
    print("Fire Cause Model and Prediction")
    RandomFModel.run_RF(X, y, 0.20)

    X = firedata[['FIRE_YEAR','DAY_OF_WEEK','DISCOVERY_DATE','MONTH','STATE', 'LATITUDE','LONGITUDE','STAT_CAUSE_DESCR']]
    y = firedata['FIRE_SIZE_CLASS']
    print("Running ......")
    print("Fire Class Prediction")
    RandomFModel.run_RF(X, y, 0.20)

    
    
    return



if __name__ == '__main__':
    main_p()    
