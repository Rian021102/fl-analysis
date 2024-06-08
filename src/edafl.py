import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def eda(X, y):
    #split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    #print the shape of the training sets
    print('X_train Shape: ',X_train.shape)
    print('y_train Shape: ',y_train.shape)    
    
    #print missing values in training set

    print("Number of Missing Data: ",X_train.isnull().sum())

    #print descriptive statistics for training set
    print("Data Statistic Describe: ",X_train.describe())

    #plot histogram of X_train
    X_train.hist(figsize=(20,15))
    plt.show()
    
    #plot boxplot of X_train
    X_train.boxplot(figsize=(20,15))
    plt.show()

    #plot correlation matrix
    corr = X_train.corr()
    sns.heatmap(corr, annot=True)
    plt.show()

    #print values counts for y_train
    print(y_train.value_counts())

    
    return X_train, X_test, y_train, y_test



