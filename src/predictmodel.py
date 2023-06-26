import pandas as pd
import numpy as np
import pickle
dftest = pd.read_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/dftest.csv')
print(dftest.head())

# Fill NaN values in FLUID with corresponding LITHO value
dftest['FLUID'] = dftest['FLUID'].fillna(0) 

# Rename values in FLUID column
dftest['FLUID'] = dftest['FLUID'].map({0: 0, 7: 1, 8: 2, 9:3, 10: 4, 11: 5, 12: 6 })

# Drop missing values
dftest.dropna(inplace=True)

# Reset index
dftest.reset_index(drop=True, inplace=True)

# Create dfpred DataFrame without 'LITHO', 'Wells', and 'FLUID' columns
dfpred = dftest.drop(['LITHO', 'Wells', 'FLUID'], axis=1)

# Load model
model = pickle.load(open('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/models/model.pkl', 'rb'))

# Predict
prediction = model.predict(dfpred)

# Prediction labels
prediction_labels = [
    'Non-SST', 'GAS', 'Poss Gas', 'Oil', 'Poss Oil', 'Wtr', 'Wtrrise'
]

# Put prediction and prediction labels into dftest
dftest['PREDICTION'] = prediction
dftest['PREDICTION_LABELS'] = [prediction_labels[pred] for pred in prediction]
print(dftest.head())

# Save prediction
dftest.to_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/processed/dftestnew.csv', index=False)
