from loaddatafl import loaddata
from edafl import eda
from datacleaning import cleandata
from trainingmodel import train
from modelling import model_based
from train_new import train_mod
from catboosttrain import train_my_cat
<<<<<<< HEAD
# import sys
=======
#importing confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
>>>>>>> 40e742ab7021690a36b4f4aebe620a336acb81d5
def main():
    # Load data
    path = r"P:/project/pythonpro/myvenv/fl-analysis/data/raw/combined_new.csv"
    X_train,y_train,X_test,y_test = loaddata(path)
    print('data loaded')

    # Exploratory Data Analysis
<<<<<<< HEAD
    # eda(X_train, y_train)
=======
    eda(X_train, y_train)
>>>>>>> 40e742ab7021690a36b4f4aebe620a336acb81d5

    # Data Cleaning
    X_train, y_train, X_test,y_test = cleandata(X_train, y_train, X_test, y_test)
    print('data cleaned')

<<<<<<< HEAD
    # train(X_train, y_train, X_test, y_test)
    # # Model
    model=train_my_cat(X_train, y_train, X_test, y_test)
    # #model_based(X_train, y_train, X_test, y_test)
    # train_mod(X_train, y_train, X_test, y_test)
=======
    #train(X_train, y_train, X_test, y_test)
    # Model
    # model=train_my_cat(X_train, y_train, X_test, y_test)
    #model_based(X_train, y_train, X_test, y_test)
    train_mod(X_train, y_train, X_test, y_test)
>>>>>>> 40e742ab7021690a36b4f4aebe620a336acb81d5
    

if __name__ == '__main__':
    main()

