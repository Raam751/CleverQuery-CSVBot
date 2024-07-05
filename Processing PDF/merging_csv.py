import os
import pandas as pd
import csv

# Directory containing the CSV files
input_dir = "output_tables"

# List to hold DataFrames
dfs = []

# Iterate through the CSV files in order
csv_files = sorted([f for f in os.listdir(input_dir) if f.startswith("table_page_") and f.endswith(".csv")],
                   key=lambda x: int(x.split('_')[-1].split('.')[0]))

for i, csv_file in enumerate(csv_files):
    csv_path = os.path.join(input_dir, csv_file)
    
    # Read the CSV file with all columns as strings
    df = pd.read_csv(csv_path, dtype=str)
    
    # Keep the first 5 rows only for the first CSV file
    if i == 0:
        dfs.append(df)
    else:
        dfs.append(df.iloc[3:])

# Concatenate all DataFrames
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame to a new CSV file without altering data format
merged_csv_path = "2.csv"
merged_df.to_csv(merged_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

print(f"Merged {len(csv_files)} CSV files into '{merged_csv_path}'.")
