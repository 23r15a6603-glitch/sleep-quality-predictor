import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import joblib

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ✅ Step 1: Check if file exists
filename = "ai_sleep_quality_dataset_2000.csv"
if not os.path.exists(filename):
    raise FileNotFoundError(f"❌ Dataset not found: {filename}")

# ✅ Step 2: Load dataset
df = pd.read_csv(filename)

# ✅ Step 3: Rename column if needed
if "Sleep Quality" not in df.columns:
    if "Quality of Sleep" in df.columns:
        df.rename(columns={"Quality of Sleep": "Sleep Quality"}, inplace=True)
    else:
        raise ValueError("❌ Dataset must have a 'Sleep Quality' or 'Quality of Sleep' column")

# ✅ Step 4: Split features and label
X = df.drop("Sleep Quality", axis=1)
y = df["Sleep Quality"]

# ✅ Step 5: Upsample to balance classes
df_combined = pd.concat([X, y], axis=1)
class_counts = df_combined['Sleep Quality'].value_counts()
max_count = class_counts.max()

frames = []
for cls in class_counts.index:
    cls_df = df_combined[df_combined['Sleep Quality'] == cls]
    upsampled = resample(cls_df, replace=True, n_samples=max_count, random_state=42)
    frames.append(upsampled)

balanced_df = pd.concat(frames)
X = balanced_df.drop("Sleep Quality", axis=1)
y = balanced_df["Sleep Quality"]

# ✅ Step 6: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_sleep_quality.pkl")

# ✅ Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Step 8: Train and save models
# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb.predict(X_test)))
joblib.dump(xgb, "xgb_sleep_quality_model.pkl")

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

# SVM
svm = SVC()
svm.fit(X_train, y_train)
print("SVM Accuracy:", accuracy_score(y_test, svm.predict(X_test)))
