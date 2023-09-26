import pandas as pd
import lasio

#petrophyics
S15BP1pet=lasio.read("/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/fwdatas/S-15BP1_Perophysical Log.las")
S6RD4pet=lasio.read("/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/fwdatas/S-6RD4_Petrophysical Log.las")
S7RD2pet=lasio.read("/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/fwdatas/S-7RD2_Petrophysical Log.las")


#lithofluid
S15BP1fl=lasio.read("/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/fwdatas/S-15BP1_Litho-Fluid.las")
S6RD4fl=lasio.read("/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/fwdatas/S-6RD4_Litho-Fluid.las")
S7RD2fl=lasio.read("/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/fwdatas/S-7RD2_Litho-Fluid.las")


#transform to dataframe
S15BP1petdf=S15BP1pet.df()
S6RD4petdf=S6RD4pet.df()
S7RD2petdf=S7RD2pet.df()
S15BP1fldf=S15BP1fl.df()
S6RD4fldf=S6RD4fl.df()
S7RD2fldf=S7RD2fl.df()

#check sample from dataframe
print(S15BP1petdf.head())
print(S15BP1fldf.head())

#merge petrophysics and lithofluid
S15BP1df=pd.merge(S15BP1petdf,S15BP1fldf,on='DEPTH')
S6RD4df=pd.merge(S6RD4petdf,S6RD4fldf,on='DEPTH')
S7RD2df=pd.merge(S7RD2petdf,S7RD2fldf,on='DEPTH')

#concatenate all data
well=pd.concat([S15BP1df,S6RD4df,S7RD2df])
well.reset_index(inplace=True)

#export to csv
well.to_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/new_test_sie.csv',index=False)