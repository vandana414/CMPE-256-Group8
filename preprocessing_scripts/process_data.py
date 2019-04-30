import pandas as pd
from sklearn import preprocessing
class process_df:
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