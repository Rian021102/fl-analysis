import lasio
import pandas as pd
import joblib

model=joblib.load('P:/project/pythonpro/myvenv/fl-analysis/models/model_cat.pkl')

las= lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/STA-38_8.5 in_Petrophysical_Provisional_RM.las')
df= las.df()
df_pred=df.copy()
df_pred=df_pred.reset_index()
df_pred=df_pred[['DEPTH','GR','RHOB','NPHI','DT']]