import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("credit_risk_dataset.csv")

# Handle categorical columns
le_edu = LabelEncoder()
le_house = LabelEncoder()

df['Education_Level'] = le_edu.fit_transform(df['Education_Level'])
df['Housing_Status'] = le_house.fit_transform(df['Housing_Status'])

# Features & Target
X = df.drop("Default", axis=1)
y = df["Default"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model + encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_edu, open("le_edu.pkl", "wb"))
pickle.dump(le_house, open("le_house.pkl", "wb"))

print("✅ Model trained successfully")