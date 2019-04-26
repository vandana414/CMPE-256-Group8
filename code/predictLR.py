import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from preprocessing_scripts.labels import labels
from LogisticRegression import LRModel

test_df = pd.read_csv('../data/test_new.csv',usecols = ['STAT_CAUSE_DESCR','LATITUDE','LONGITUDE','STATE','DISCOVERY_DATE','FIRE_SIZE','avg_temp'])

#Separate the labels
y_test = pd.DataFrame()
y_test['STAT_CAUSE_DESCR']=test_df['STAT_CAUSE_DESCR']
test_df = test_df.drop(columns=['STAT_CAUSE_DESCR'])
test_df = test_df.drop(columns=['STATE'])

#create new classes
y_test=labels.createLabel(y_test)
y_test=y_test['STAT_CAUSE_DESCR'].astype(int)

#Predict using trained model
model = LRModel()
model.load_model("../models/lrmodel")
y_pred = model.predict(test_df)

#Evaluate
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
print("F1-score: %f"%(f1_score(y_test,y_pred,average='macro')))
confusion_matrix(y_test, y_pred)