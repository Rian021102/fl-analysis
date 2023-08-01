import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest


    
def cleandata(X_train, y_train, X_test,y_test):
    X_train.drop(['Wells', 'LITHO','VCALCITE','VQUARTZ','VSH','VCOAL','VORGSH','VGAS','VOIL','VWATER','PHIT','PHIE','SWE','PERM'], axis=1, inplace=True)
    X_test.drop(['Wells', 'LITHO','VCALCITE','VQUARTZ','VSH','VCOAL','VORGSH','VGAS','VOIL','VWATER','PHIT','PHIE','SWE','PERM'], axis=1, inplace=True)

    # Drop rows with NaN values
    X_train.dropna(inplace=True)
    #align y_train with X_train
    y_train = y_train.loc[X_train.index]
    # Drop rows with NaN values
    X_test.dropna(inplace=True)
    #align y_test with X_test
    y_test = y_test.loc[X_test.index]

    # Remove outliers
    iso = IsolationForest(contamination=0.1)
    yhat = iso.fit_predict(X_train)
    mask = yhat != -1
    X_train = X_train.loc[mask, :]
    y_train = y_train.loc[mask]


    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # Print the shape of the training sets
    print(X_train_sm.shape)
    print(y_train_sm.shape)

    return X_train_sm, y_train_sm, X_test, y_test
