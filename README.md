# CMPE-256-Group8
Wildfire Cause Prediction

Dataset : https://www.kaggle.com/rtatman/188-million-us-wildfires
A separate python sql_to_csv.py to convert the SQLite data to Fires.csv in data folder. If fires file is not present in data folder run this script.
Dependencies
``` pandas, numpy, scikit-learn, matplotlib, joblib, tensorflow, h5py```

-----------------------------------------------------------------------------------------------------------------
To get the predictions for SVM, Bagging Classifier, RandomForest, Decision Tree and Naive Bayes:
```python run_models.py```
The output file goes as output.txt in data folder


-----------------------------------------------------------------------------------------------------------------
The models for logistic regression and newural network are trained and stored in models folder, but if it gives some error the models needs to be trained again. 
To train the models execute below commands(execute this if python predict.py fails)

```python LogisticRegression.py```

```python NeuralNetwork.py```

This will train and save models in models folder

-----------------------------------------------------------------------------------------------------------------
To get the predictions for Logistic Regression and Neural Network model:

```python predict.py```

-----------------------------------------------------------------------------------------------------------------

The above commands need to be run from the code folder of the project
