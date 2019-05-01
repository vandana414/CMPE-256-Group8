from pandas_ml import ConfusionMatrix
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report,accuracy_score 
from sklearn.model_selection import train_test_split
class BaggingModel:
    def run_Bagging(X, y, split):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split)
        clf = BaggingClassifier(n_estimators=5, warm_start=True,random_state=3141)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(y_pred) 
        
        print("Accuracy score for Bagging classifier is:",accuracy_score(y_test,y_pred))
        confusion_matrix = ConfusionMatrix(y_test, y_pred)
        print("Confusion matrix:\n%s" % confusion_matrix)
        report = classification_report(y_test,y_pred)
        print("Classification report:\n%s" % report)
 
        return