import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# Load your dataset
df = pd.read_csv("ai_sleep_quality_dataset_2000.csv")  # Make sure this file exists

# Example preprocessing – customize for your dataset
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Smoking'] = df['Smoking'].map({'Yes': 1, 'No': 0})
df['Sleep Disorder History'] = df['Sleep Disorder History'].map({'Yes': 1, 'No': 0})
df['Wake-up Consistency'] = df['Wake-up Consistency'].map({'Regular': 1, 'Irregular': 0})

X = df.drop(columns=['Sleep Quality'])  # features
y = df['Sleep Quality']                 # labels

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = XGBClassifier()
model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(model, "xgb_sleep_quality_model.pkl")
joblib.dump(scaler, "scaler_sleep_quality.pkl")
