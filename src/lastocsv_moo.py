import pandas as pd
import lasio
import os

# Function to read LAS files into dataframes and store in a dictionary
def load_las_to_df(directory, suffix_strip=None, rename_cols=False):
    df_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('.las'):
            file_path = os.path.join(directory, filename)
            las_file = lasio.read(file_path)
            # Remove the specific suffix if provided
            if suffix_strip and filename.endswith(suffix_strip + '.las'):
                df_name = filename.replace(suffix_strip + '.las', '')
            else:
                df_name = filename.replace('.las', '')
            df = las_file.df().reset_index()  # Ensure DEPTH is a column
            # Rename columns if the flag is True (only for Litho-Fluid files)
            if rename_cols:
                df.rename(columns={'LITHO_CODE': 'LITHO', 'FLUID_CODE': 'FLUID'}, inplace=True)
            df_dict[df_name] = df
            print(f"Loaded {df_name}")
    return df_dict

# Directories
path_petrophysical = '/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/Petrophysical'
path_lithofluid = '/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/Litho-Fluid'

# Load LAS files into dataframes
petro_df_dict = load_las_to_df(path_petrophysical)
litho_df_dict = load_las_to_df(path_lithofluid, '_Litho-Fluid', rename_cols=True)  # Pass rename_cols=True only for this directory

# List to hold all merged dataframes
all_merged_dfs = []

# Merge corresponding dataframes
for petro_key in petro_df_dict:
    if petro_key in litho_df_dict:
        # Ensuring DEPTH remains a column after merge
        merged_df = pd.merge(petro_df_dict[petro_key], litho_df_dict[petro_key], on='DEPTH', how='inner')
        all_merged_dfs.append(merged_df)
        print(f"Merged and added {petro_key}")
    else:
        print(f"No matching lithofluid file for {petro_key}")

# Concatenate all dataframes
final_df = pd.concat(all_merged_dfs, ignore_index=True)
print("All dataframes concatenated.")

# Save to CSV
final_df.to_csv('/Users/rianrachmanto/pypro/project/Litho-Fluid-Id/data/raw/new_test_moo.csv', index=False)
print("Data saved to new_test_moo.csv")
