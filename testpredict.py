import pandas as pd
import requests  # Import the requests library
import lasio

def load_data(path):
    data=lasio.read(path)
    df= data.df()
    df.reset_index(inplace=True)
    #drop nan
    df.dropna(inplace=True)
    df=df[['DEPTH','GR','RT','RHOB','NPHI']]
    return df


def main():
    path = "P:/project/pythonpro/myvenv/fl-analysis/data/raw/STA-38_8.5 in_Petrophysical_Provisional_RM.las"
    df = load_data(path)
    data = {
        "data": df.to_dict(orient='records')  # This converts the DataFrame to a list of dictionaries
    }

    # Define the API endpoint
    url = 'http://10.18.0.60/sdaz10/predict'

    # Make the prediction by calling the API
    response = requests.post(url, json=data)
    if response.status_code == 200:
        predictions = response.json()
      
    #make the prediction result as a new column of the dataframe
        df['PRED_LIQUID'] = predictions['predictions']
        print(df)
    else:
        print("Failed to get predictions", response.status_code, response.text)

if __name__ == "__main__":
    main()

