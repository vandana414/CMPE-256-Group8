from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from joblib import dump,load
import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from preprocessing_scripts.handlecategorical import categories
from preprocessing_scripts.labels import labels
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Activation
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model

class NNModel:
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=20, activation='relu',use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(20,activation='relu',use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(4, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    model = baseline_model()    
    #estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=1000, verbose=1)
        
    
    def trainModel(self,train,y):
        train = categories.createCategorical(train)
        #estimator = KerasClassifier(build_fn=self.model, epochs=10, batch_size=1000, verbose=1)
        self.model.fit(train,np.eye(4)[y.ravel().tolist()],epochs=100, batch_size=1000, verbose=1)
    
    def predict(self,test):
        test = categories.createCategorical(test)
        results = self.model.predict(test)
        final = []
        for result in results:
            final.append(np.argmax(result))
        return final
    
    def save(self,f_name):
        self.model.save(f_name+'_model.h5')
        
    def load_model(self,f_name):
        self.model = load_model(f_name+'_model.h5')

seed = 7
np.random.seed(seed)

if __name__ == '__main__':
    #read train data from train.csv
    train_df = pd.read_csv('../data/train.csv',usecols = ['STAT_CAUSE_DESCR','LATITUDE','LONGITUDE','DISCOVERY_DATE','FIRE_SIZE'])
    y = pd.DataFrame()
    y['STAT_CAUSE_DESCR']=train_df['STAT_CAUSE_DESCR']
    
    #create labels by grouping the causes
    y=labels.createLabel(y)
    y=y['STAT_CAUSE_DESCR'].astype(int)
    train_df = train_df.drop(columns=['STAT_CAUSE_DESCR'])
  
    #train and save the model
    model = LRModel()
    model.trainModel(train_df,y)
    model.save("../models/nnmodel")
    
