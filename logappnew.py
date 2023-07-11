import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler
import os

@lru_cache(maxsize=1)
def load_model():
    with open('models/model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

def preprocess_data(df):
    df.dropna(inplace=True)
    df.drop(['Wells', 'LITHO', 'FLUID'], axis=1, inplace=True)  # Remove 'FLUID'
    return df

def predict(df):
    predictions = model.predict(df)
    labels = ['Non-SST', 'Gas', 'PosGas', 'Oil', 'PosOil', 'WTR', 'WtrRise']
    df['PREDICTION'] = predictions
    df['LABEL'] = [labels[prediction] for prediction in predictions]
    return df

def make_facies_log_plot(logs, facies_colors, facies_labels):
    # Make sure logs are sorted by depth
    logs = logs.sort_values(by='DEPTH')
    cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')

    ztop = logs.DEPTH.min()
    zbot = logs.DEPTH.max()

    cluster_prediction = np.repeat(np.expand_dims(logs['PREDICTION'].values, 1), 100, 1)

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 15))
    ax[0].plot(logs.GR, logs.DEPTH, '-g')
    ax[1].plot(logs.RT, logs.DEPTH, '-')
    ax[2].plot(logs.NPHI, logs.DEPTH, '-', color='0.40')
    ax[3].plot(logs.RHOB, logs.DEPTH, '-', color='r')
    im_prediction = ax[4].imshow(cluster_prediction, interpolation='none', aspect='auto',
                                cmap=cmap_facies, vmin=0, vmax=10)

    divider = make_axes_locatable(ax[4])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im_prediction, cax=cax)
    cbar.set_ticks(range(len(facies_labels)))
    cbar.set_ticklabels(facies_labels)

    # Adjust tick label properties
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xticks(horizontalalignment='center')
    plt.yticks(horizontalalignment='center')
    plt.tick_params(axis='both', which='both', pad=20)

    # Set y-axis limits for RT plot
    ax[1].set_ylim(logs.DEPTH.min(), logs.DEPTH.max())

    for i in range(len(ax) - 1):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)

    ax[0].set_xlabel("GR")
    ax[1].set_xlabel("RT")
    ax[2].set_xlabel("NPHI")
    ax[3].set_xlabel("RHOB")

    fig.tight_layout()  # Adjust the layout to prevent overlapping

    st.pyplot(fig)  # Display the plot in Streamlit

def main():
    st.title("Fluid Analysis Prediction Using Machine Learning")
    
    facies_labels = ['Non-SST', 'Gas', 'PosGas', 'Oil', 'PosOil', 'WTR', 'WtrRise']

    #setting facies colors as follows:[dark grey, maroon, red, dark green, light green, dark blue, light blue]
    facies_colors = ['#000000', '#990000', '#CC3333', '#006600', '#00CC00', '#000099', '#0000CC']
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Original Data")
        st.write(df)
        
        df_scaled = preprocess_data(df)  # Added preprocessing step

        if st.button('Predict'):
            predictions_df = predict(df)
            st.subheader("Predictions")
            st.write(predictions_df)
            
            make_facies_log_plot(predictions_df, facies_colors, facies_labels)

        # Add save button to save the prediction to a CSV file
        if st.button('Save Prediction'):
            predictions_df = predict(df)
            
            # Get the path to the Downloads directory
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            
            # Save the predictions to the Downloads directory
            predictions_df.to_csv(os.path.join(downloads_path, 'predictions.csv'), index=False)
            st.write('Saved to predictions.csv')
            
if __name__ == '__main__':
    main()
