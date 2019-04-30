from pandas_ml import ConfusionMatrix
class DecTreeModel:
    def run_tree(X,y,split):
        from sklearn.model_selection import train_test_split 
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split)
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        from sklearn.metrics import classification_report,accuracy_score  
    #     print("Classification Report for Decision Tree")
    #     print(classification_report(y_test,y_pred))  
        print("Accuracy score for Decision Tree:",accuracy_score(y_test,y_pred))
        confusion_matrix = ConfusionMatrix(y_test, y_pred)
        print("Confusion matrix:\n%s" % confusion_matrix)
        return