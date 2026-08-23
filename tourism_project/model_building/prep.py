import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

DATASET_LOC = "tourism_project/data/tourism.csv"
input_dataset = pd.read_csv(DATASET_LOC)
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
X = input_dataset[numeric_features + categorical_features]

# Target variable
y = input_dataset[target]

# Spliting dataset into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Input (X) and target variable (y)
    test_size=0.2,     # 20% of the data for testing
    random_state=42    # Fixed random seed to be able to reproduce the results.
)

Xtrain.to_csv("tourism_project/data/Xtrain.csv",index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv",index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv",index=False)
ytest.to_csv("tourism_project/data/ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

