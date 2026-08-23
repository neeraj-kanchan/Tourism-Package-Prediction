import os
import pandas as pd

# Load dataset locally from the repository folder
data_path = "tourism_project/data/tourism.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print("Dataset registered successfully!")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
else:
    raise FileNotFoundError(f"Dataset not found at {data_path}")