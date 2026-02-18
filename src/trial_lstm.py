import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data(path):
    df = pd.read_csv(path)
    #fillna fluid with 0
    df['FLUID'].fillna(0, inplace=True)

    # Ensure numeric data and coerce invalid values to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Rename values in FLUID column
    df['FLUID'] = df['FLUID'].map({0: 0, 7: 1, 8: 2, 9:3, 10: 4, 11: 5, 12: 6})
    df = df.dropna(subset=['FLUID'])

    # reset index
    df.reset_index(drop=True, inplace=True)

    #exclude where RT is -999.000000, RHOB is -999.000000, NPHI is -9.990000
    df = df[df['RT'] != -999.000000]
    df = df[df['RHOB'] != -999.000000]
    df = df[df['NPHI'] != -9.990000]
    #remove outliers using IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    df = df.dropna()
    return df

def main():
    path = 'P:/project/pythonpro/myvenv/fl-analysis/data/raw/combined_new.csv'
    df = load_data(path)

    # Build sequences by depth order
    df = df.reset_index(drop=True)
    X = df.drop(columns=['FLUID']).values
    y = df['FLUID'].values

    seq_len = 10
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(seq_len, X_seq.shape[2])),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('\nTest loss:', test_loss)
    print('Test accuracy:', test_acc)

if __name__ == "__main__":
    main()
 
