import pandas as pd
from pyspark import SparkConf, SparkContext
import sys
sys.path.append("../")
from preprocessing_scripts.handlecategorical import categories
from preprocessing_scripts.labels import labels 
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import MulticlassMetrics

conf = SparkConf()
sc = SparkContext(conf=conf)



train_df = pd.read_csv('../data/train_new.csv',usecols = ['STAT_CAUSE_DESCR','LATITUDE','LONGITUDE','DISCOVERY_DATE','FIRE_SIZE','avg_temp'])
y = pd.DataFrame()
y['STAT_CAUSE_DESCR']=train_df['STAT_CAUSE_DESCR']
y=labels.createLabel(y)
y['STAT_CAUSE_DESCR']=y['STAT_CAUSE_DESCR'].astype(int)
train_df = train_df.drop(columns=['STAT_CAUSE_DESCR'])
train_df=categories.createCategorical(train_df)
pd.concat([y,train_df],axis=1,sort=False).reset_index(drop=True).to_csv('../data/traintemp.csv',header=False,index=False)



train = sc.textFile("../data/traintemp.csv").map(lambda line: line.split(","))


def parsePoint(line):
    return LabeledPoint(line[0], line[1:])

parsedData = train.map(lambda line:parsePoint(line))


model =RandomForest.trainClassifier(parsedData,seed=42,numClasses=4,numTrees=200,maxDepth=10,\
categoricalFeaturesInfo={5:2,6:2,7:2,8:2,9:2,10:2,11:2,12:2,13:2,14:2,15:2,16:2,17:2,18:2,19:2,20:2,21:2})

test_df = pd.read_csv('../data/test_new.csv',usecols = ['STAT_CAUSE_DESCR','LATITUDE','LONGITUDE','STATE','DISCOVERY_DATE','FIRE_SIZE','avg_temp'])


y_test = pd.DataFrame()
y_test['STAT_CAUSE_DESCR']=test_df['STAT_CAUSE_DESCR']
test_df = test_df.drop(columns=['STAT_CAUSE_DESCR'])
test_df = test_df.drop(columns=['STATE'])

#create new classes
y_test=labels.createLabel(y_test)
y_test=y_test['STAT_CAUSE_DESCR'].astype(int)

test_df = categories.createCategorical(test_df)
pd.concat([y_test,test_df],axis=1,sort=False).reset_index(drop=True).to_csv('../data/testtemp.csv',header=False,index=False)

test = sc.textFile("../data/testtemp.csv").map(lambda line: line.split(","))
test = test.map(lambda x:((x[1:]),x[0]))
y_pred = model.predict(test.map(lambda x:x[0]))
labelsandpreds = test.map(lambda x:float(x[1])).zip(y_pred)
print(labelsandpreds.first())
#Evaluate
metrics = MulticlassMetrics(labelsandpreds)
print(metrics.confusionMatrix().toArray())
print("F1 score: %f"%(metrics.fMeasure()))
