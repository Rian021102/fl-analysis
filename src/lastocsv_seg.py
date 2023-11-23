import pandas as pd
import lasio

#petrophyics
seg2pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/OH Log/SGA-2_Logs.las')
seg3pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/OH Log/SGA-3_Logs.las')
seg3rd1pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/OH Log/SGA-3RD1_Logs.las')
seg4pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/OH Log/SGA-4_Logs.las')
segb2pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/OH Log/SGB-2_Logs.las')
segb3pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/OH Log/SGB-3_Logs.las')
segb4pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/OH Log/SGB-4_Logs.las')

#lithofluid
seg2lf=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/Litho-Fluid/seguni_sga-2_Litho-Fluid.las')
seg3lf=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/Litho-Fluid/seguni_sga-3_Litho-Fluid.las')
seg3rd1lf=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/Litho-Fluid/seguni_sga-3rd1_Litho-Fluid.las')
seg4lf=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/Litho-Fluid/seguni_sga-4_Litho-Fluid.las')
segb2lf=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/Litho-Fluid/seguni_sgb-2_Litho-Fluid.las')
segb3lf=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/Litho-Fluid/seguni_sgb-3_Litho-Fluid.las')
segb4lf=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/SEGUNI Logs for AutoFluid_20231011/Litho-Fluid/seguni_sgb-4_Litho-Fluid.las')

#transform to dataframe
seg2petdf=seg2pet.df()
seg3petdf=seg3pet.df()
seg3rd1petdf=seg3rd1pet.df()
seg4petdf=seg4pet.df()
segb2petdf=segb2pet.df()
segb3petdf=segb3pet.df()
segb4petdf=segb4pet.df()
seg2lfdf=seg2lf.df()
seg3lfdf=seg3lf.df()
seg3rd1lfdf=seg3rd1lf.df()
seg4lfdf=seg4lf.df()
segb2lfdf=segb2lf.df()
segb3lfdf=segb3lf.df()
segb4lfdf=segb4lf.df()

#check sample from dataframe
print(seg2petdf.head())
print(seg2lfdf.head())

#merge petrophysics and lithofluid
seg2df=pd.merge(seg2petdf,seg2lfdf,on='DEPTH')
seg3df=pd.merge(seg3petdf,seg3lfdf,on='DEPTH')
seg3rd1df=pd.merge(seg3rd1petdf,seg3rd1lfdf,on='DEPTH')
seg4df=pd.merge(seg4petdf,seg4lfdf,on='DEPTH')
segb2df=pd.merge(segb2petdf,segb2lfdf,on='DEPTH')
segb3df=pd.merge(segb3petdf,segb3lfdf,on='DEPTH')
segb4df=pd.merge(segb4petdf,segb4lfdf,on='DEPTH')

#concatenate all data
well=pd.concat([seg2df,seg3df,seg3rd1df,seg4df,segb2df,segb3df,segb4df])
well.reset_index(inplace=True)

#export to csv
well.to_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/new_test_seg.csv',index=False)