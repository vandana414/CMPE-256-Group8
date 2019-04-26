from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from joblib import dump,load
import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from preprocessing_scripts.handlecategorical import categories
from preprocessing_scripts.labels import labels

class LRModel:
    normalizer = Normalizer(norm ='l2')
    model = LogisticRegression(C=1000,penalty='l2',random_state=10,class_weight='balanced' ,solver='liblinear',multi_class='ovr')
   
       
    def fitNormalizer(self,train):
        self.normalizer.fit(train)
    
    def normalize(self,test):
        return self.normalizer.transform(test)
        
    
    def trainModel(self,train,y):
        train = categories.createCategorical(train)
        self.fitNormalizer(train)
        train = self.normalize(train)
        self.model.fit(train, y)
    
    def predict(self,test):
        test = categories.createCategorical(test)
        test1 = self.normalize(test)
        return self.model.predict(test1).tolist()
    
    def save(self,f_name):
        dump(self.normalizer, f_name+'_normalizer.joblib')
        dump(self.model,f_name+'_model.joblib')
        
    def load_model(self,f_name):
        self.normalizer = load(f_name+'_normalizer.joblib')
        self.model = load(f_name+'_model.joblib')
        
if __name__ == '__main__':
    #read train data from train.csv
    train_df = pd.read_csv('../data/train_new.csv',usecols = ['STAT_CAUSE_DESCR','LATITUDE','LONGITUDE','DISCOVERY_DATE','FIRE_SIZE','avg_temp'])
    y = pd.DataFrame()
    y['STAT_CAUSE_DESCR']=train_df['STAT_CAUSE_DESCR']
    y=labels.createLabel(y)
    y=y['STAT_CAUSE_DESCR'].astype(int)
    train_df = train_df.drop(columns=['STAT_CAUSE_DESCR'])
    model = LRModel()
    model.trainModel(train_df,y)
    model.save("../models/lrmodel")
	