import numpy as np
import pandas as pd
import sqlite3
import math as math

conn = sqlite3.connect('../data/FPA_FOD_20170508.sqlite')

fire_df = pd.read_sql_query("SELECT STAT_CAUSE_DESCR,LATITUDE,LONGITUDE,STATE,DISCOVERY_DATE,FIRE_SIZE FROM 'Fires'", conn)

from sklearn.utils import shuffle
fire_df = shuffle(fire_df).reset_index(drop = True)

train_df = fire_df[:math.ceil(len(fire_df)*0.7)]
train_df.to_csv('../data/train.csv',sep=',')

test_df = fire_df[math.ceil(len(fire_df)*0.7):]
test_df.to_csv('../data/test.csv',sep=',')