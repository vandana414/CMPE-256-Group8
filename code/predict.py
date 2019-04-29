import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from preprocessing_scripts.labels import labels
from NeuralNetwork import NNModel
from LogisticRegression import LRModel
from collections import Counter
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score


    
if __name__ == '__main__':
    test_df = pd.read_csv('../data/test.csv',usecols = ['STAT_CAUSE_DESCR','LATITUDE','LONGITUDE','DISCOVERY_DATE','FIRE_SIZE'])

    #Separate the labels
    y_test = pd.DataFrame()
    y_test['STAT_CAUSE_DESCR']=test_df['STAT_CAUSE_DESCR']
    test_df = test_df.drop(columns=['STAT_CAUSE_DESCR'])

    #create new classes
    y_test=labels.createLabel(y_test)
    y_test=y_test['STAT_CAUSE_DESCR'].astype(int)
    
    #Predict using Logistic Regression trained model
    model = LRModel()
    model.load_model("../models/lrmodel")
    y_pred = model.predict(test_df)
    

    #Evaluate#
    print("Logistic Regression Model")
    print("------------------------------------------------------------------------------------------")
    print("F1-score: %f"%(f1_score(y_test,y_pred,average='macro')))
    print(confusion_matrix(y_test, y_pred))
    
    #Predict using Neural Network trained model
    model = NNModel()
    model.load_model("../models/nnmodel")
    y_pred = model.predict(test_df)
    
    #Evaluate#
    print("Nueral Network Model")
    print("------------------------------------------------------------------------------------------")
    print("F1-score: %f"%(f1_score(y_test,y_pred,average='macro')))
    print(confusion_matrix(y_test, y_pred))