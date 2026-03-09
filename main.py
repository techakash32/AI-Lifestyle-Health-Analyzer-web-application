import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from sqlalchemy import create_engine

# ── Connect to MySQL ───────────────────────────────────────────────────────────
engine = create_engine('mysql+mysqlconnector://root:your_password@localhost/health_db')

# ── Load dataset ───────────────────────────────────────────────────────────────
query = "SELECT * FROM health_data"
df    = pd.read_sql(query, engine)


# ── Drop ID column ─────────────────────────────────────────────────────────────
df.drop(columns=['User_ID'], inplace=True)
X = df.drop(columns=['Work_Productivity_Score'])
y = df['Work_Productivity_Score']

gender_encoder     = LabelEncoder()
occupation_encoder = LabelEncoder()
device_encoder     = LabelEncoder()

X = X.copy()
X['Gender']      = gender_encoder.fit_transform(X['Gender'])
X['Occupation']  = occupation_encoder.fit_transform(X['Occupation'])
X['Device_Type'] = device_encoder.fit_transform(X['Device_Type'])


# ── Save feature order BEFORE split ───────────────────────────────────────────
feature_order = X.columns.tolist()
print("Feature order:", feature_order)

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train model ───────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)
print("Model Accuracy:", r2)


# ── Save all artifacts ───────
joblib.dump(model,             'productivity_model.pkl')
joblib.dump(gender_encoder,    'gender_encoder.pkl')
joblib.dump(occupation_encoder,'occupation_encoder.pkl')
joblib.dump(device_encoder,    'device_encoder.pkl')
joblib.dump(feature_order,     'feature_order.pkl')

print("\n✅ Saved:")
print("   productivity_model.pkl")
print("   gender_encoder.pkl")
print("   occupation_encoder.pkl")
print("   device_encoder.pkl")
print("   feature_order.pkl")