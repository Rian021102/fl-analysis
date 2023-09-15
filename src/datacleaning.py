import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


    
def cleandata(X_train, y_train, X_test,y_test):
    X_train.drop(['LITHO','DT'], axis=1, inplace=True)
    X_test.drop(['LITHO','DT'], axis=1, inplace=True)

    #filter RHOB and NPHI where is not negative
    X_train = X_train[(X_train['RHOB'] > 0) & (X_train['NPHI'] > 0)]

    #allign y_train with X_train
    y_train = y_train.loc[X_train.index]

    



    # Drop rows with NaN values
    X_train.dropna(inplace=True)
    #align y_train with X_train
    y_train = y_train.loc[X_train.index]
    # Drop rows with NaN values
    X_test.dropna(inplace=True)
    #align y_test with X_test
    y_test = y_test.loc[X_test.index]



   
    # Remove outliers
    iso = IsolationForest(contamination=0.05)
    yhat = iso.fit_predict(X_train)
    mask = yhat != -1
    X_train = X_train.loc[mask, :]
    y_train = y_train.loc[mask]



    # Print the shape of the training sets
    print(X_train.shape)
    print(y_train.shape)

    return X_train, y_train, X_test, y_test
