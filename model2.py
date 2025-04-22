import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import ipaddress
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

data = pd.read_csv('rba-dataset.csv', nrows=1000000)
chunk_size = 1000000  # Adjust based on your memory limits

# Initialize an empty list to store the data
chunks = []

# Read the CSV file in chunks
for chunk in pd.read_csv('/kaggle/input/rba-dataset/rba-dataset.csv', chunksize=chunk_size):
    chunks.append(chunk)
    print("SSs")

# Concatenate all chunks into a single DataFrame
data = pd.concat(chunks, axis=0)
data.head()
len(data)
data.dtypes
data.info()
data.describe
data.isna().sum()
data['Login Hour'] = pd.to_datetime(data['Login Timestamp']).dt.hour
data.head()
data['Is Account Takeover'] = data['Is Account Takeover'].astype(np.uint8)
data['Is Attack IP'] = data['Is Attack IP'].astype(np.uint8)
data['Login Successful'] = data['Login Successful'].astype(np.uint8)
data = data.drop(columns=["Round-Trip Time [ms]", 'Region', 'City', 'Login Timestamp', 'index'])
data.head()
data['User Agent String'], _ = pd.factorize(data['User Agent String'])
data['Browser Name and Version'], _ = pd.factorize(data['Browser Name and Version'])
data['OS Name and Version'], _ = pd.factorize(data['OS Name and Version'])
def ip_to_int(ip):
    return int(ipaddress.ip_address(ip))

data['IP Address'] = data['IP Address'].apply(ip_to_int)
data.head(20)
account_takeover_rows = data[data['Is Account Takeover'] == 1]

# Display or further process the filtered rows
account_takeover_rows
categorical_cols = ['Country', 'Device Type']
numeric_cols = ['ASN', 'Login Hour', 'IP Address', 'User Agent String', 'Browser Name and Version', 'OS Name and Version']
"""percentage wise calculations
"""
percentage1=data['Is Attack IP'].value_counts(normalize=True)*100
percentage2=data['Is Account Takeover'].value_counts(normalize=True)*100
percentage3=data['Login Successful'].value_counts(normalize=True)*100
classlabels=["No Attack","Attacked"]
plt.figure(figsize=(12,7))
plt.pie(percentage1,labels=classlabels,autopct='%1.3f%%')
plt.title("Attack IP")
plt.show()
classlabels=["Not Takeover", "Takeover"]
plt.figure(figsize=(12,7))
plt.pie(percentage2,labels=classlabels,autopct='%1.5f%%')
plt.title("Is Account Takeover")
plt.show()
classlabels=["Success","Unsuccess"]
plt.figure(figsize=(12,7))
plt.pie(percentage3,labels=classlabels,autopct='%1.3f%%')
plt.title("Login Success")
plt.show()
# Splitting the dataset
features = data.drop(['Is Attack IP', 'Is Account Takeover'], axis=1)
labels = data['Is Account Takeover']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42,stratify=labels)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Classifiers
classifiers = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'decision_tree': DecisionTreeClassifier(),
    'svm': SVC(probability=True),
    'random_forest': RandomForestClassifier(),
    'Adaboost' : AdaBoostClassifier(),
    'Extra' :  ExtraTreesClassifier(),
    'lgbm' : LGBMClassifier(),
    'XGB': XGBClassifier()
}

# A function to choose classifiers
def make_pipeline(classifier_key):
    if classifier_key in classifiers:
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifiers[classifier_key])
        ])
        return clf
    else:
        raise ValueError(f"Classifier {classifier_key} is not defined")
    classifier_key = 'logistic_regression'
    pipeline = make_pipeline(classifier_key)
    pipeline.fit(X_train, y_train)

    # Evaluation
    lrpredictions = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probs)

    print(f"AUC Score: {auc_score}")
    Score = accuracy_score(y_test, lrpredictions)
    Classification_Report = classification_report(y_test, lrpredictions)

    print("Logistic Regression")
    print("Accuracy Score value: {:.4f}".format(Score))
    print(Classification_Report)

Logistic_Regression_Confusion_Matrix = ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
Logistic_Regression_Confusion_Matrix
plt.show()
classifier_key = 'decision_tree'
pipeline = make_pipeline(classifier_key)
pipeline.fit(X_train, y_train)

# Evaluation
dtpredictions = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, probs)

print(f"AUC Score: {auc_score}")

Score = accuracy_score(y_test,dtpredictions)
Classification_Report = classification_report(y_test,dtpredictions)

print("Decision Tree")
print ("Accuracy Score value: {:.8f}".format(Score))
print (Classification_Report)

Logistic_Regression_Confusion_Matrix = ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
Logistic_Regression_Confusion_Matrix
plt.show()
classifier_key = 'random_forest'
pipeline = make_pipeline(classifier_key)
pipeline.fit(X_train, y_train)

# Evaluation
predictions = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, probs)

print(f"AUC Score: {auc_score}")
