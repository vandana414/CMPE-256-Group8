import numpy as np
import pandas as pd

avg_temp = pd.read_csv('../data/avgtemp_new.csv')
avg_temp['MONTH'] = pd.to_datetime(avg_temp['dt']).dt.month
avg_temp['YEAR'] = pd.to_datetime(avg_temp['dt']).dt.year
class temperature:
    def appendAverageTemperature(self,df):
        df['DATE'] = pd.to_datetime(df['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')
        df['MONTH'] = pd.DatetimeIndex(df['DATE']).month
        df['YEAR'] = pd.DatetimeIndex(df['DATE']).year
        state_dict = {}
        for state in df.STATE.unique():
            state_dict[state] = avg_temp[['AverageTemperature','MONTH','YEAR']][avg_temp.State == state]
        per_state={}
        for state in df.STATE.unique():
            df1 = state_dict[state]
            per_state[state] = {}
            per_year = per_state[state]
            for year in range(1992,2016):
                per_year[year]={}
                per_month=per_year[year]
                for month in range(1,13):
                    per_month[month] = {}
                    per_month[month] = df1[['AverageTemperature']][df1.MONTH == month]
        avg_temps = []
        for i in range(len(df)):
            if(df.iloc[i].STATE == 'PR'):
                avg_temps.append(0)
            else:
                avg_temps.append(per_state[df.iloc[i].STATE][df.iloc[i].YEAR][df.iloc[i].MONTH].AverageTemperature.values[0])
        df['avg_temp']=avg_temps
        df = df.drop(columns=['MONTH','DATE','YEAR'])
        return df
if __name__ == '__main__':
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    temp = temperature()
    train_df = temp.appendAverageTemperature(train_df)
    test_df = temp.appendAverageTemperature(test_df)
    test_df.to_csv('../data/test_new.csv',sep=',')
    train_df.to_csv('../data/train_new.csv',sep=',')