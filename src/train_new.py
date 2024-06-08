import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

#set parameter with softmax

def train_mod(X_train, y_train, X_test, y_test):

    # define model with softmax
    model = XGBClassifier(class_weight='balanced')

    # fit model on training data
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)

    # evaluate predictions

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    #print classification report

    print(classification_report(y_test, y_pred))

    return model
    