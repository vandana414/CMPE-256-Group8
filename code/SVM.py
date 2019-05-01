from pandas_ml import ConfusionMatrix
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score 
from sklearn.model_selection import train_test_split

class SVMModel:
    def run_SVM(X, y, split):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split)
        model = svm.SVC(gamma=0.1)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print(y_pred) 
 
        print("Accuracy score for SVM is:",accuracy_score(y_test,y_pred))
        confusion_matrix = ConfusionMatrix(y_test, y_pred)
        print("Confusion matrix:\n%s" % confusion_matrix)
        report = classification_report(y_test,y_pred)
        print("Classification report:\n%s" % report)
        return