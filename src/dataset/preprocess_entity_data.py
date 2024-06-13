import pandas as pd
from pathlib import Path


path_to_input_data = Path('../../data/raw/entities')
path_to_output_data = Path('../../data/processed/entities')

persons_df = pd.read_json(path_to_input_data / 'persons.json_', lines=True)
organisations_df = pd.read_json(path_to_input_data / 'organisations.json_', lines=True)

# Select only the names
persons_df = persons_df[['norm_name']]
organisations_df = organisations_df[['norm_name']]

# rename columns for convenience
persons_df.columns = ["name"]
organisations_df.columns = ["name"]

persons_df["label"] = "per"
organisations_df["label"] = "org"

# Preprocess text of the names since we only need firstnames and lastnames as their own rows

persons_df["name"] = persons_df["name"].str.lower()
persons_df['name'] = persons_df['name'].str.split(' ')
persons_df = persons_df.explode('name').reset_index(drop=True)
persons_df = persons_df.drop_duplicates().reset_index(drop=True)

organisations_df["name"] = organisations_df["name"].str.lower()
organisations_df['name'] = organisations_df['name'].str.split(' ')
organisations_df = organisations_df.explode('name').reset_index(drop=True)
organisations_df = organisations_df.drop_duplicates().reset_index(drop=True)

result = pd.concat([persons_df, organisations_df], ignore_index=True)

path_to_output_data.mkdir(parents=True, exist_ok=True)
result.to_csv(path_to_output_data / "entities.csv", index=False)

print("List of labeled entities has been created")
