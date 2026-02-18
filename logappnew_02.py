import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import lru_cache
import base64
import os
import lasio
import tempfile
import requests


def predict(df, model): 
    #drop columns
    df2=df.copy()
    df2=df2[['DEPTH','GR','RT','RHOB','NPHI']]
    predictions = model.predict(df2).astype(int)  # Ensure predictions are integers
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
    # Ensure RT values are positive for semilogx plot
    valid_rt = logs[logs.RT > 0]
    # ax[1].semilogx(valid_rt.RT, valid_rt.DEPTH)
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
    
    # Setting facies colors as follows: [dark grey, maroon, red, dark green, light green, dark blue, light blue]
    facies_colors = ['#000000', '#990000', '#CC3333', '#006600', '#00CC00', '#000099', '#0000CC']
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()
        
        # Use the temporary file path with lasio
        las = lasio.read(temp_file.name)
        df = las.df()
        #dropna
        df.dropna(inplace=True)
        #reset index
        df.reset_index(inplace=True)
        df=df[['DEPTH','GR','RT','RHOB','NPHI']]
        
        st.subheader("Original Data")
        st.write(df)
                
        prediction_done = False
        if st.button('Predict'):
            # Define the API endpoint
            data = {
            "data": df.to_dict(orient='records')  # This converts the DataFrame to a list of dictionaries
            }

            url = 'http://10.18.0.60/sdaz10/predict'

            # Make the prediction by calling the API
            response = requests.post(url, json=data)
            if response.status_code == 200:
                predictions = response.json()
                # make the prediction result as a new column of the dataframe
                df['PREDICTION'] = predictions['predictions']
                # Optionally, add LABEL column if needed for plotting
                labels = ['Non-SST', 'Gas', 'PosGas', 'Oil', 'PosOil', 'WTR', 'WtrRise']
                df['LABEL'] = [labels[pred] if 0 <= pred < len(labels) else 'Unknown' for pred in df['PREDICTION']]
                print(df)
                prediction_done = True
            else:
                st.error(f"Failed to get predictions: {response.status_code} - {response.text}")
            
            if prediction_done:
                df_plot=df.copy()
                #select only positive values
                df_plot = df_plot[df_plot['PREDICTION'] >= 0]
                
                make_facies_log_plot(df_plot, facies_colors, facies_labels)  
                st.write(df)
                
           
            
if __name__ == '__main__':
    main()
