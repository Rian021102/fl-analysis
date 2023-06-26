import lasio

# Read the file
df = lasio.read('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/S-15BP1_Petrophysical Log.las')

# convert to pandas dataframe
df = df.df()
print(df.head())

#reset index
df.reset_index(inplace=True)

#list of columns
cols = df.columns.tolist()
print(cols)