import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
from imblearn.over_sampling import SMOTE

# === 1. LOAD & EDA ===
df = pd.read_csv('ObesityDataSet.csv')
print(df.info())

# Cleaning
df.drop_duplicates(inplace=True)
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Konversi numerik
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in numerical_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Hapus outlier
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df = remove_outliers_iqr(df, numerical_features)

# Label Encoding
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns.tolist()
if 'NObeyesdad' in cat_cols:
    cat_cols.remove('NObeyesdad')
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])

# === 2. SMOTE & SCALING ===
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# === 3. TUNING ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    scoring='f1_weighted',
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train, y_train)

# === 4. EVALUASI ===
model = grid.best_estimator_
y_pred = model.predict(X_test)

print("Best Estimator:", model)
print("Best Params:", grid.best_params_)
print("Best F1:", grid.best_score_)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === 5. VISUALISASI ===
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: Tuned Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
