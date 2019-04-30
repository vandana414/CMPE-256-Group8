import sqlite3
import pandas as pd
from pandas_ml import ConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
import seaborn as sns

# %matplotlib inline
# import plotly.graph_objs as go 
# from plotly.offline import init_notebook_mode,iplot,plot
# init_notebook_mode(connected=True)
def run_tree(X,y,split):
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split)
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import classification_report,accuracy_score  
#     print("Classification Report for Decision Tree")
#     print(classification_report(y_test,y_pred))  
    print("Accuracy score for Decision Tree:",accuracy_score(y_test,y_pred))
#     confusion_matrix = ConfusionMatrix(y_test, y_pred)
#     print("Confusion matrix:\n%s" % confusion_matrix)
    return

def run_NB(X, y, split):
# quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
# X_q = quantile_transformer.fit_transform(X)
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split)
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)


    from sklearn.metrics import classification_report,accuracy_score  
#     print("Score for Naive Baye's:",model.score(X_train, y_train))
#     print("Classification Report")
#     print(classification_report(y_test,y_pred))  
    print("Accuracy score for Naive  baye's:",accuracy_score(y_test,y_pred))
#     confusion_matrix = ConfusionMatrix(y_test, y_pred)
#     print("Confusion matrix:\n%s" % confusion_matrix)
    return

def preprocess(firedata):
    
    firedata['DISCOVERY_DATE'] = pd.to_datetime(firedata['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D' )
    firedata['DAY_OF_WEEK'] = pd.to_datetime(firedata['DISCOVERY_DATE']).dt.weekday_name
    firedata['MONTH'] = pd.DatetimeIndex(firedata['DISCOVERY_DATE']).month
    firedata['CONT_DATE'] = pd.to_datetime(firedata['CONT_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')                        
    enc = preprocessing.LabelEncoder()
    firedata['STAT_CAUSE_DESCR'] = enc.fit_transform(firedata['STAT_CAUSE_DESCR'])
    firedata['DAY_OF_WEEK'] = enc.fit_transform(firedata['DAY_OF_WEEK'])
    firedata['MONTH'] = enc.fit_transform(firedata['MONTH'])
    firedata['STATE'] = enc.fit_transform(firedata['STATE'])
    firedata['FIRE_SIZE_CLASS'] = enc.fit_transform(firedata['FIRE_SIZE_CLASS'])
    fire = firedata[['FIRE_YEAR','DAY_OF_WEEK','MONTH','STATE', 'LATITUDE','LONGITUDE','FIRE_SIZE_CLASS','FIRE_SIZE','STAT_CAUSE_DESCR']]
    return firedata


def read_data():

    conn = sqlite3.connect('../data/FPA_FOD_20170508.sqlite')
    df = pd.read_sql("SELECT * from Fires LIMIT 1000",con=conn)
    firedata = df.filter(['SOURCE_REPORTING_UNIT_NAME','STATE','LATITUDE','LONGITUDE','FIRE_SIZE','FIRE_SIZE_CLASS','FIRE_YEAR','DISCOVERY_DATE','STAT_CAUSE_DESCR','CONT_DATE','CONT_DOY','CONT_TIME','OWNER_CODE','COUNTY'],axis=1)
    return firedata

def main_p():
    print("Running ......")
    firedata =read_data()
    fire = preprocess(firedata)

    X = fire[['FIRE_YEAR','DAY_OF_WEEK','MONTH','STATE', 'LATITUDE','LONGITUDE','FIRE_SIZE']]
    y = fire['STAT_CAUSE_DESCR']
    print("Fire Cause Model and Prediction")
    run_NB(X, y, 0.2)
    run_tree(X,y,0.2)

    X = firedata[['FIRE_YEAR','DAY_OF_WEEK','MONTH','STATE', 'LATITUDE','LONGITUDE','STAT_CAUSE_DESCR']]
    y = firedata['FIRE_SIZE_CLASS']
    print("Running ......")
    print("Fire Class Model and Prediction")
    run_NB(X, y, 0.2)
    run_tree(X,y,0.2)
    return

main_p()