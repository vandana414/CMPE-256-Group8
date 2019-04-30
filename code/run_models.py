import sqlite3
import pandas as pd
from pandas_ml import ConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
import sys
sys.path.append("../")
from preprocessing_scripts.readdata import read_sql
from preprocessing_scripts.process_data import process_df
from NaiveBayes import NBModel 
from DecisionTree import DecTreeModel
import seaborn as sns

def main_p():
    print("Running ......")
    
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
    return



if __name__ == '__main__':
    main_p()    
