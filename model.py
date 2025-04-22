import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('rba-dataset.csv')

# Define independent variables (features)
features = ['Country', 'Region', 'City', 'ASN', 'User Agent String',
            'Browser Name and Version', 'OS Name and Version', 'Device Type']
target = 'Is Account Takeover'  # Target column

# Ensure all necessary columns are present
df = df[features + [target]].copy()

# Handle missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Convert categorical features to numeric using LabelEncoder
label_encoders = {}
for col in features:
    le = LabelEncoder()
    df.loc[:, col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Store encoders in case needed later

# Define X (features) and y (target)
X = df[features]
y = df[target].astype(int)

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Train the model
classifier = RandomForestClassifier(n_estimators=100, random_state=50)
classifier.fit(X_train, y_train)

# Save the trained model
pickle.dump(classifier, open('model.pkl', 'wb'))

print("✅ Model training complete. Pickle file saved.")
