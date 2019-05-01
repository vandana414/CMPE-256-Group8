import numpy as np
import pandas as pd
import sqlite3

conn = sqlite3.connect('../data/FPA_FOD_20170508.sqlite')
df = pd.read_sql("SELECT * from Fires LIMIT 700000 ",con=conn)
firedata = df.filter(['SOURCE_REPORTING_UNIT_NAME','STATE','LATITUDE','LONGITUDE','FIRE_SIZE','FIRE_SIZE_CLASS','FIRE_YEAR','DISCOVERY_DATE','STAT_CAUSE_DESCR','CONT_DATE','CONT_DOY','CONT_TIME','OWNER_CODE','COUNTY'],axis=1)
firedata['DISCOVERY_DATE'] = pd.to_datetime(firedata['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D' )
firedata['CONT_DATE'] = pd.to_datetime(firedata['CONT_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')

firedata.to_csv('../data/fires.csv',sep=',')
       