import pandas as pd

df2=pd.read_csv('P:/project/pythonpro/myvenv/fl-analysis/data/raw/new_test_sie.csv')
print(df2.head())
df2_new=df2[['DEPTH','GR','RT','RHOB','NPHI','LITHO','FLUID']]
print(df2_new.head())

dfold=pd.read_csv('P:/project/pythonpro/myvenv/fl-analysis/data/raw/new_train.csv')
dfold.drop(['DT'],axis=1,inplace=True)
print(dfold.head())

sdd=pd.read_csv('P:/project/pythonpro/myvenv/fl-analysis/data/raw/sdd_train.csv')
sdd.drop(['DT'],axis=1,inplace=True)
print(sdd.head())

seg=pd.read_csv('P:/project/pythonpro/myvenv/fl-analysis/data/raw/new_test_seg.csv')
seg.drop(['DT'],axis=1,inplace=True)
print(seg.head())

moo=pd.read_csv('P:/project/pythonpro/myvenv/fl-analysis/data/raw/new_test_moo.csv')
moo=moo[['DEPTH','GR','RT','RHOB','NPHI','LITHO','FLUID']]
print(moo.head())

sun=pd.read_csv('P:/project/pythonpro/myvenv/fl-analysis/data/raw/new_test_suntan.csv')
sun=sun[['DEPTH','GR','RT','RHOB','NPHI','LITHO','FLUID']]
print(sun.head())

#merge df2_new and dfold
df=pd.concat([df2_new,dfold,sdd,seg,moo,sun])
print(df.head())

#export to csv
df.to_csv('P:/project/pythonpro/myvenv/fl-analysis/data/raw/combined_new.csv',index=False)