import pandas as pd
import numpy as np
import pickle

class categories:
    def createCategorical(fire_df):
        fire_df['DATE'] = pd.to_datetime(fire_df['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')
        fire_df['DAY_OF_WEEK'] = pd.to_datetime(fire_df['DATE']).dt.dayofweek
        fire_df['MONTH'] = pd.DatetimeIndex(fire_df['DATE']).month

        #create a column for each day_of_week
        features = ['mon', 'tue', 'wed', 'thur','fri', 'sat']
        day_features = pd.DataFrame(0, index=np.arange(len(fire_df)), columns=features)

        for f in range(len(features)):
            day_features[features[f]] = (fire_df.DAY_OF_WEEK == (f))*1

        #create a column for each month
        features = ['jan', 'feb', 'mar', 'apr', 'may','jun', 'jul', 'aug', 'sep', 'oct', 'nov']
        month_features = pd.DataFrame(0, index=np.arange(len(fire_df)), columns=features)
        for f in range(len(features)):
            month_features[features[f]] = (fire_df.MONTH == (f+1))*1
     
        
        fire_df = fire_df.drop(columns=['MONTH','DAY_OF_WEEK','DISCOVERY_DATE','DATE'])
        return pd.concat([fire_df,day_features,month_features],axis=1,sort=False)
        
