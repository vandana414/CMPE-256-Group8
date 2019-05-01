import pandas as pd
import numpy as np
import sqlite3

class read_sql:
    def read_data():

        df = pd.read_csv('../data/fires.csv' ,sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
        firedata = df[['SOURCE_REPORTING_UNIT_NAME','STATE','LATITUDE','LONGITUDE','FIRE_SIZE','FIRE_SIZE_CLASS','FIRE_YEAR','DISCOVERY_DATE','STAT_CAUSE_DESCR','CONT_DATE','CONT_DOY','CONT_TIME','OWNER_CODE','COUNTY']]
        return firedata