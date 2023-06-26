from loaddatafl import loaddata
from edafl import eda
from datacleaning import cleandata
from trainingmodel import train

def main():
    # Load data
    path = '/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/dftest.csv'
    X,y = loaddata(path)
    print('data loaded')

    # Exploratory Data Analysis
    X_train, X_test, y_train, y_test = eda(X, y)

    # Data Cleaning
    X_train_sm, y_train_sm, X_test,y_test = cleandata(X_train, y_train, X_test, y_test)
    print('data cleaned')

    # Training Model
    train(X_train_sm, y_train_sm, X_test, y_test)
    print('model trained')
   
        

if __name__ == '__main__':
    main()

