import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eda(X_train, y_train):
    
    
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




