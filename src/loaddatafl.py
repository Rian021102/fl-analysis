import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# initialize logging format and level
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def loaddata(pathfile):
    # a log message indicating that data loading is starting
    logging.info('Loading data')

    # load the data using pandas read_csv()
    df = pd.read_csv(pathfile)
    # a log message indicating that data loading is complete
    logging.info('Data loaded')

   #fillna fluid with 0
    df['FLUID'].fillna(0, inplace=True)

    # Rename values in FLUID column
    df['FLUID'] = df['FLUID'].map({0: 0, 7: 1, 8: 2, 9:3, 10: 4, 11: 5, 12: 6 })

    # reset index
    df.reset_index(drop=True, inplace=True)

    #exclude where RT is -999.000000, RHOB is -999.000000, NPHI is -9.990000
    df = df[df['RT'] != -999.000000]
    df = df[df['RHOB'] != -999.000000]
    df = df[df['NPHI'] != -9.990000]
    

    # print the first 5 rows
    print(df.head())

    # print the last 5 rows
    print(df.tail())

    # print the shape of the data
    print(df.shape)

    # print the data types
    print(df.dtypes)

    # set X and y
    X = df.drop('FLUID', axis=1)
    y = df['FLUID']

    # print X and y shape
    print(X.shape)
    print(y.shape)

    return X, y
