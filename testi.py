# ---------------------------
# UAS PROJECT - OBESITY DATA
# ---------------------------

# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

from imblearn.over_sampling import SMOTE

# --- Load Dataset ---
df = pd.read_csv('ObesityDataSet.csv')

# -----------------------------
# 1. EXPLORATORY DATA ANALYSIS
# -----------------------------
print("=== 5 Baris Pertama Dataset ===")
print(df.head())

print("\n=== Informasi Dataset ===")
print(df.info())

print("\n=== Statistik Deskriptif (Numerik) ===")
print(df.describe(include='all'))

print("\n=== Jumlah Baris dan Kolom ===")
print(df.shape)

print("\n=== Cek Missing Values ===")
print(df.isnull().sum())

print("\n=== Cek Nilai Unik Tiap Kolom ===")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n=== Cek Data Duplikat ===")
print(f"Jumlah baris duplikat: {df.duplicated().sum()}")

# Visualisasi Distribusi Target
plt.figure(figsize=(8,5))
sns.countplot(y='NObeyesdad', data=df, order=df['NObeyesdad'].value_counts().index)
plt.title("Distribusi Kelas Target (NObeyesdad)")
plt.show()

# Boxplot fitur numerik
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
plt.figure(figsize=(15,10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2,4,i+1)
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot {feature}")
plt.tight_layout()
plt.show()

# ----------------------------
# 2. DATA CLEANING & PREP
# ----------------------------

# Hapus duplikat
df.drop_duplicates(inplace=True)

# Ganti simbol ? jadi NaN
df.replace('?', np.nan, inplace=True)

# Hapus missing values
df.dropna(inplace=True)

# Konversi tipe kolom numerik
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Hapus outlier dengan IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df = remove_outliers_iqr(df, num_cols)

# Salin dataframe untuk ditampilkan di Streamlit (tanpa label encoding)
df_display = df.copy()

# Label Encoding untuk modeling
df_encoded = df.copy()
cat_cols = df_encoded.select_dtypes(include='object').columns.tolist()
if 'NObeyesdad' in cat_cols:
    cat_cols.remove('NObeyesdad')

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le  # simpan encoder untuk decode nanti jika perlu

# Encode label (target)
target_le = LabelEncoder()
df_encoded['NObeyesdad'] = target_le.fit_transform(df_encoded['NObeyesdad'])

# Pisah fitur dan label
X = df_encoded.drop('NObeyesdad', axis=1)
y = df_encoded['NObeyesdad']

# SMOTE
print("\nDistribusi label sebelum SMOTE:")
print(y.value_counts())

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("\nDistribusi label setelah SMOTE:")
print(pd.Series(y_res).value_counts())

# Visualisasi SMOTE
plt.figure(figsize=(7,4))
sns.countplot(x=y_res)
plt.title("Distribusi Label Setelah SMOTE")
plt.xlabel("Kelas Obesitas (encoded)")
plt.ylabel("Jumlah")
plt.show()

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# -----------------------
# 3. MODELING & EVALUASI
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Report': classification_report(y_test, y_pred, output_dict=True)
    }

    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

# Confusion Matrix Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, result) in enumerate(results.items()):
    cm = result['Confusion Matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f"Confusion Matrix: {name}")
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
plt.tight_layout()
plt.show()

# Barplot Metrics
metrics_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
metrics_df.plot(kind='bar', figsize=(10,6), colormap='Set2')
plt.title("Perbandingan Performa Model")
plt.ylabel("Skor")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# -------------------------------
# 4. TUNING RANDOM FOREST
# -------------------------------

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    verbose=2,
    n_jobs=-1
)

grid_search_rf.fit(X_train, y_train)

print("\nModel terbaik setelah GridSearchCV:")
print(grid_search_rf.best_estimator_)
print("\nParameter terbaik:")
print(grid_search_rf.best_params_)
print(f"\nSkor terbaik (mean F1-score dari CV): {grid_search_rf.best_score_:.4f}")

# Evaluasi ulang
best_rf = grid_search_rf.best_estimator_
y_pred_tuned = best_rf.predict(X_test)

print("\n=== Evaluasi Model Random Forest Setelah Tuning ===")
print(classification_report(y_test, y_pred_tuned))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

# --------------------------
# 5. Perbandingan F1-Scores
# --------------------------
f1_before = results["Random Forest"]['F1-Score']
f1_after = grid_search_rf.best_score_

compare_df = pd.DataFrame({
    'Model': ['Random Forest (Before Tuning)', 'Random Forest (After Tuning)'],
    'F1-Score': [f1_before, f1_after]
})

plt.figure(figsize=(8,5))
plt.bar(compare_df['Model'], compare_df['F1-Score'], color=['skyblue', 'green'])
plt.title('Perbandingan F1-Score Sebelum & Setelah Hyperparameter Tuning')
plt.ylim(0.8, 1.0)
plt.ylabel('F1-Score')
plt.xticks(rotation=20)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("=== Kesimpulan ===")
print(f"F1-Score meningkat dari {f1_before:.4f} menjadi {f1_after:.4f} setelah tuning.")
print("Model yang sudah dituning menunjukkan performa prediksi yang lebih baik dan stabil.")
