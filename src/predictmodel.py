import pandas as pd
import numpy as np
import pickle
import lasio

# Load LAS file
las = lasio.read('/Users/rianrachmanto/pypro/project/data/S-15BP1_Petrophysical Log.las')

# Convert to DataFrame
dftest = las.df()

# Reset index and move 'DEPTH' to a regular column
dftest.reset_index(inplace=True)

# Create dfpred DataFrame without 'LITHO', 'Wells', and 'FLUID' columns
dfpred = dftest.copy()

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
