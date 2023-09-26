import pandas as pd
import numpy as np
import lasio

#petrophysincs
sa5pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-5RD1HZ_Logs.las')
sa10pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-10_Logs.las')
sa15pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-15RD1ST_Logs.las')
sa15rd3pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-15RD3_Logs.las')
sa21pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-21_Logs.las')
sa24pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-24HZ_Logs.las')
sa25pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-25RD1_Logs.las')
sa25bppet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-25RD1BP1_Logs.las')
sa32pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-32_Logs.las')
sa32rd1pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-32RD1ST1_Logs.las')
sa35pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-35ST1.las')
sa41pet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-41_Litho-Fluid.las')
sa44hzpet=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-44hz_Litho-Fluid.las')
#lithofluid
sa5fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-5rd1hz_Litho-Fluid.las')
sa10fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-10_Litho-Fluid.las')
sa15fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-15rd1st_Litho-Fluid.las')
sa15rd3fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-15rd3_Litho-Fluid.las')
sa21fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-21_Litho-Fluid.las')
sa24fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-24hz_Litho-Fluid.las')
sa25fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-25rd1_Litho-Fluid.las')
sa25bpfl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-25rd1bp1_Litho-Fluid.las')
sa32fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-32_Litho-Fluid.las')
sa32rd1fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-32rd1st1_Litho-Fluid.las')
sa35fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/Litho-Fluid/sa-35st1_Litho-Fluid.las')
sa41fl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-41_Logs.las')
sa44hzfl=lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/Surrounding Wells Data SA-5RD2 & SA-29RD4/OH Log/SA-44HZ_Logs.las')

#transform to dataframe 
sa5petdf=sa5pet.df()
sa10petdf=sa10pet.df()
sa15petdf=sa15pet.df()
sa15rd3petdf=sa15rd3pet.df()
sa21petdf=sa21pet.df()
sa24petdf=sa24pet.df()
sa25petdf=sa25pet.df()
sa25bppetdf=sa25bppet.df()
sa32petdf=sa32pet.df()
sa32rd1petdf=sa32rd1pet.df()
sa35petdf=sa35pet.df()
sa41petdf=sa41pet.df()
sa44hzpetdf=sa44hzpet.df()

#check sample from dataframe
print(sa5petdf.head())

sa5fldf=sa5fl.df()
sa10fldf=sa10fl.df()
sa15fldf=sa15fl.df()
sa15rd3fldf=sa15rd3fl.df()
sa21fldf=sa21fl.df()
sa24fldf=sa24fl.df()
sa25fldf=sa25fl.df()
sa25bpfldf=sa25bpfl.df()
sa32fldf=sa32fl.df()
sa32rd1fldf=sa32rd1fl.df()
sa35fldf=sa35fl.df()
sa41fldf=sa41fl.df()
sa44hzfldf=sa44hzfl.df()

#check sample from dataframe
print(sa5fldf.head())

#merge petrophysics and lithofluid
sa5df=pd.merge(sa5petdf,sa5fldf,on='DEPTH')
sa10df=pd.merge(sa10petdf,sa10fldf,on='DEPTH')
sa15df=pd.merge(sa15petdf,sa15fldf,on='DEPTH')
sa15rd3df=pd.merge(sa15rd3petdf,sa15rd3fldf,on='DEPTH')
sa21df=pd.merge(sa21petdf,sa21fldf,on='DEPTH')
sa24df=pd.merge(sa24petdf,sa24fldf,on='DEPTH')
sa25df=pd.merge(sa25petdf,sa25fldf,on='DEPTH')
sa25bpdf=pd.merge(sa25bppetdf,sa25bpfldf,on='DEPTH')
sa32df=pd.merge(sa32petdf,sa32fldf,on='DEPTH')
sa32rd1df=pd.merge(sa32rd1petdf,sa32rd1fldf,on='DEPTH')
sa35df=pd.merge(sa35petdf,sa35fldf,on='DEPTH')
sa41df=pd.merge(sa41petdf,sa41fldf,on='DEPTH')
sa44hzdf=pd.merge(sa44hzpetdf,sa44hzfldf,on='DEPTH')

#concat all well
well=pd.concat([sa5df,sa10df,sa15df,sa15rd3df,sa21df,sa24df,sa25df,sa25bpdf,sa32df,sa32rd1df,sa35df,sa41df,sa44hzdf])
well.reset_index(inplace=True)
print(well.head())

#export to csv
well.to_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/new_train.csv',index=False)
