import pandas as pd
import numpy as np
import lasio

#petrophysics
jempangpet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/OH Log/JEMPANG-1_Logs.las')
sdd_2rdpet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/OH Log/SDD-2RD1_Logs.las')
sdd_5pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/OH Log/SDD-5_Logs.las')
sdd_5rdpet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/OH Log/SDD-5RD1_Logs.las')
sdd_6pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/OH Log/SDD-6_Logs.las')
sdd_6rdpet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/OH Log/SDD-6RD1_Logs.las')
sdd_7pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/OH Log/SDD-7_Logs.las')

#lithofluid
jempanglit=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/Litho-Fluid/jempang-1_Litho-Fluid.las')
sdd_2rdlit=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/Litho-Fluid/sdd-2rd1_Litho-Fluid.las')
sdd_5lit=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/Litho-Fluid/sdd-5_Litho-Fluid.las')
sdd_5rdlit=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/Litho-Fluid/sdd-5rd1_Litho-Fluid.las')
sdd_6lit=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/Litho-Fluid/sdd-6_Litho-Fluid.las')
sdd_6rdlit=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/Litho-Fluid/sdd-6rd1_Litho-Fluid.las')
sdd_7lit=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/SEDANDANG Logs for AutoFluid_20231011/Litho-Fluid/sdd-7_Litho-Fluid.las')

#transform to dataframe
jempangpetdf=jempangpet.df()
sdd_2rdpetdf=sdd_2rdpet.df()
sdd_5petdf=sdd_5pet.df()
sdd_5rdpetdf=sdd_5rdpet.df()
sdd_6petdf=sdd_6pet.df()
sdd_6rdpetdf=sdd_6rdpet.df()
sdd_7petdf=sdd_7pet.df()

jempanglitdf=jempanglit.df()
sdd_2rdlitdf=sdd_2rdlit.df()
sdd_5litdf=sdd_5lit.df()
sdd_5rdlitdf=sdd_5rdlit.df()
sdd_6litdf=sdd_6lit.df()
sdd_6rdlitdf=sdd_6rdlit.df()
sdd_7litdf=sdd_7lit.df()

#check sample from dataframe
print(jempangpetdf.head())

#check sample from dataframe
print(jempanglitdf.head())

#merge petrophysics and lithofluid
jempang=pd.merge(jempangpetdf,jempanglitdf,how='inner',on='DEPTH')
sdd_2rd=pd.merge(sdd_2rdpetdf,sdd_2rdlitdf,how='inner',on='DEPTH')
sdd_5=pd.merge(sdd_5petdf,sdd_5litdf,how='inner',on='DEPTH')
sdd_5rd=pd.merge(sdd_5rdpetdf,sdd_5rdlitdf,how='inner',on='DEPTH')
sdd_6=pd.merge(sdd_6petdf,sdd_6litdf,how='inner',on='DEPTH')
sdd_6rd=pd.merge(sdd_6rdpetdf,sdd_6rdlitdf,how='inner',on='DEPTH')
sdd_7=pd.merge(sdd_7petdf,sdd_7litdf,how='inner',on='DEPTH')

#concat all well
well=pd.concat([jempang,sdd_2rd,sdd_5,sdd_5rd,sdd_6,sdd_6rd,sdd_7])
well.reset_index(inplace=True)
print(well.head())

#save to csv
well.to_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/sdd_train.csv',index=False)

