import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from huggingface_hub import login, HfApi

api = HfApi(token=os.getenv("GH_TOKEN"))
DATASET_LOC = "hf://datasets/neeraj-kanchan/Tourism-Package-Prediction/tourism.csv"
bank_dataset = pd.read_csv(DATASET_LOC)
print("Dataset loaded successfully.")

# Target variable
target = 'ProdTaken'

# Numeric columns
numeric_features = [
    'Age',
    'CityTier',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch'
]

# Categorical columns
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched',
    'Passport',
    'OwnCar',
]

# Predictor matrix (X)
X = bank_dataset[numeric_features + categorical_features]

# Target variable
y = bank_dataset[target]

# Spliting dataset into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Input (X) and target variable (y)
    test_size=0.2,     # 20% of the data for testing
    random_state=42    # Fixed random seed to be able to reproduce the results.
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="neeraj-kanchan/Tourism-Package-Prediction",
        repo_type="dataset",
    )
