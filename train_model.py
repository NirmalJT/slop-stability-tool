import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
data = pd.read_csv("dataset.csv")
data.columns = data.columns.str.strip().str.upper()

# Clean column names
data.columns = data.columns.str.strip()

# Rename for easy use
data = data.rename(columns={
    "COHESION": "c",
    "PHI": "phi",
    "HEIGHT": "H",
    "WATER LEVEL": "water",
    "SLOPE": "slope",
    "SOIL": "soil",
    "FOS": "fos"
})

# Fill missing water levels
data["water"] = data["water"].fillna(0)

# Convert slope "1.5:1" → 1.5
def convert_slope(s):
    try:
        return float(s.split(":")[0])
    except:
        return 1

data["slope"] = data["slope"].astype(str).apply(convert_slope)

# Encode soil (CL=0, CI=1)
data["soil"] = data["soil"].map({"CL": 0, "CI": 1})

# Features
X = data[["c", "phi", "H", "water", "slope", "soil"]]
y = data["fos"]

# Train model
model = RandomForestRegressor(n_estimators=200)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained successfully!")