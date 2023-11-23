import pandas as pd

df2=pd.read_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/new_test_sie.csv')
print(df2.head())
df2_new=df2[['DEPTH','GR','RT','RHOB','NPHI','LITHO','FLUID']]
print(df2_new.head())
dfold=pd.read_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/new_train.csv')
dfold.drop(['DT'],axis=1,inplace=True)
print(dfold.head())
sdd=pd.read_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/sdd_train.csv')
sdd.drop(['DT'],axis=1,inplace=True)
print(sdd.head())
seg=pd.read_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/new_test_seg.csv')
seg.drop(['DT'],axis=1,inplace=True)
print(seg.head())
#merge df2_new and dfold
df=pd.concat([df2_new,dfold,sdd,seg])
print(df.head())

#export to csv
df.to_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/combined.csv',index=False)