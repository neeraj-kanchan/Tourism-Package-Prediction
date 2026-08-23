import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="neeraj-kanchan/Tourism-Package-Prediction",
    filename="best_tourism-model_v1.joblib"
)

# Load the trained model
model = joblib.load(model_path)

# Streamlit UI
st.title("MLOPS – Tourism Package Buy Prediction Application")
st.write(
    "This application is prepared as part of MLOps project to predict the likelyhood of a customer to "
    "buy a travel package based on dataset provided."
)
st.write("Enter customer details:")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

contract_type = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

city_tier = st.selectbox("City Tier", [1, 2, 3])

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Freelancer", "Small Business", "Large Business"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

no_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1, max_value=10, value=2
)

preferred_property_rating = st.selectbox(
    "Preferred Property Ratings",
    [1, 2, 3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

no_of_trips = st.number_input(
    "Number of Trips (per year)",
    min_value=0, max_value=50, value=2
)

passport = st.selectbox("Has Passport?", ["Yes", "No"])
car_owner = st.selectbox("Owns a Car?", ["Yes", "No"])

no_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0, max_value=5, value=0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=5000, max_value=500000, value=50000
)

pitch_satisfaction_score = st.slider(
    "Pitch Satisfaction Score",
    min_value=1, max_value=5, value=3
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe"]
)

no_of_followups = st.number_input(
    "Number of Follow-ups",
    min_value=0, max_value=20, value=2
)

duration_of_pitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=1, max_value=120, value=15
)

# -----------------------------
# Prepare input data
# -----------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contract_type,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": no_of_person_visiting,
    "PreferredPropertyStar": preferred_property_rating,
    "MaritalStatus": marital_status,
    "NumberOfTrips": no_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if car_owner == "Yes" else 0,
    "NumberOfChildrenVisiting": no_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": no_of_followups,
    "DurationOfPitch": duration_of_pitch
}])

# Threshold
classification_threshold = 0.5

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)

    if prediction == 1:
        st.success("  The customer is likely to purchase the package.")
    else:
        st.error("The customer is unlikely to purchase the package.")