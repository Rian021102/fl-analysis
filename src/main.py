from loaddatafl import loaddata
from edafl import eda
from datacleaning import cleandata
from trainingmodel import train
from modelling import model_based
from train_new import train_mod
from catboosttrain import train_my_cat
def main():
    # Load data
    path = '/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/combined.csv'
    X_train,y_train,X_test,y_test = loaddata(path)
    print('data loaded')

    # Exploratory Data Analysis
    #eda(X_train, y_train)

    # Data Cleaning
    X_train, y_train, X_test,y_test = cleandata(X_train, y_train, X_test, y_test)
    print('data cleaned')

    #train(X_train, y_train, X_test, y_test)
    # Model
    model=train_my_cat(X_train, y_train, X_test, y_test)
    #model_based(X_train, y_train, X_test, y_test)
   #train_mod(X_train, y_train, X_test, y_test)
    

if __name__ == '__main__':
    main()

