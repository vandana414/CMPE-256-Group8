# CMPE-256-Group8
Wildfire Cause Prediction

Dataset : https://www.kaggle.com/rtatman/188-million-us-wildfires

Dependencies
``` pandas, numpy, scikit-learn, matplotlib, joblib, tensorflow, h5py```

The models for logistic regression and newural network are trained and stored in models folder, but if it gives some error the models needs to be trained again. 
To train the models execute classes.py (execute this if python predict.py fails)

```python LogisticRegression.py```

```python NeuralNetwork.py```

This will train and save models in models folder

-----------------------------------------------------------------------------------------------------------------
To get the predictions for Logistic Regression and Neural Network model:

```python predict.py```

-----------------------------------------------------------------------------------------------------------------
To get the predictions for SVM, Bagging Classifier, RandomForest, Decision Tree and Naive Bayes:

```python run_models.py```

-----------------------------------------------------------------------------------------------------------------
The above commands need to be run from the code folder of the project
