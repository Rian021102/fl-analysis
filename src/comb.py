import pandas as pd
import lasio


sta_2_logs = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/OH Log/sta-2_Logs.las')
sta_2_fl = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/Litho-Fluid/sta-2_Litho-Fluid.las') # Assuming a different file for sta_2_fl
sta_2_logs_df = sta_2_logs.df().reset_index()
sta_2_fl_df=sta_2_fl.df().reset_index()
sta_2_df=pd.merge(sta_2_logs_df,sta_2_fl_df, on='DEPTH', how='inner')

sta_3_logs = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/OH Log/sta-3_Logs.las')
sta_3_fl = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/Litho-Fluid/sta-3_Litho-Fluid.las') # Assuming a different file for sta_3_fl
sta_3_logs_df = sta_3_logs.df().reset_index()
sta_3_fl_df=sta_3_fl.df().reset_index()
sta_3_df=pd.merge(sta_3_logs_df,sta_3_fl_df, on='DEPTH', how='inner')

sta_4_logs = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/OH Log/sta-4_Logs.las')
sta_4_fl = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/Litho-Fluid/sta-4_Litho-Fluid.las') # Assuming a different file for sta_3_fl
sta_4_logs_df = sta_4_logs.df().reset_index()
sta_4_fl_df=sta_4_fl.df().reset_index()
sta_4_df=pd.merge(sta_4_logs_df,sta_4_fl_df, on='DEPTH', how='inner')

sta_6_logs = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/OH Log/sta-6_Logs.las')
sta_6_fl = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/Litho-Fluid/sta-6_Litho-Fluid.las') # Assuming a different file for sta_3_fl
sta_6_logs_df = sta_6_logs.df().reset_index()
sta_6_fl_df=sta_6_fl.df().reset_index()
sta_6_df=pd.merge(sta_6_logs_df,sta_6_fl_df, on='DEPTH', how='inner')

sta_18_logs = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/OH Log/sta-18_Logs.las')
sta_18_fl = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/Litho-Fluid/sta-18_Litho-Fluid.las') # Assuming a different file for sta_3_fl
sta_18_logs_df = sta_18_logs.df().reset_index()
sta_18_fl_df=sta_18_fl.df().reset_index()
sta_18_df=pd.merge(sta_18_logs_df,sta_18_fl_df, on='DEPTH', how='inner')

sta_20_logs = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/OH Log/sta-20_Logs.las')
sta_20_fl = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/Litho-Fluid/sta-20_Litho-Fluid.las') # Assuming a different file for sta_3_fl
sta_20_logs_df = sta_20_logs.df().reset_index()
sta_20_fl_df=sta_20_fl.df().reset_index()
sta_20_df=pd.merge(sta_20_logs_df,sta_20_fl_df, on='DEPTH', how='inner')

sta_21_logs = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/OH Log/sta-21_Logs.las')
sta_21_fl = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/Litho-Fluid/sta-21_Litho-Fluid.las') # Assuming a different file for sta_3_fl
sta_21_logs_df = sta_21_logs.df().reset_index()
sta_21_fl_df=sta_21_fl.df().reset_index()
sta_21_df=pd.merge(sta_21_logs_df,sta_21_fl_df, on='DEPTH', how='inner')

sta_22_logs = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/OH Log/sta-22_Logs.las')
sta_22_fl = lasio.read('P:/project/pythonpro/myvenv/fl-analysis/data/raw/Litho-Fluid/sta-22_Litho-Fluid.las') # Assuming a different file for sta_3_fl
sta_22_logs_df = sta_22_logs.df().reset_index()
sta_22_fl_df=sta_22_fl.df().reset_index()
sta_22_df=pd.merge(sta_22_logs_df,sta_22_fl_df, on='DEPTH', how='inner')

df_santan=pd.concat([sta_2_df,sta_3_df,sta_4_df,sta_6_df,
                    sta_18_df,sta_20_df,sta_21_df,sta_22_df],
                    ignore_index=True)

print(df_santan)
df_santan.to_csv('P:/project/pythonpro/myvenv/fl-analysis/data/raw/new_test_suntan.csv', index=False)