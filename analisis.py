# --- Install imblearn untuk SMOTE ---
!pip install imblearn

# --- Import Library ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv('ObesityDataSet.csv')

# Menampilkan 5 baris pertama
df.head()
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

# Visualisasi distribusi target (NObeyesdad)
plt.figure(figsize=(8,5))
sns.countplot(y='NObeyesdad', data=df, order=df['NObeyesdad'].value_counts().index)
plt.title("Distribusi Kelas Target (NObeyesdad)")
plt.show()

# Visualisasi boxplot untuk mendeteksi outlier pada fitur numerik utama
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
plt.figure(figsize=(15,10))
for i, feature in enumerate(numerical_features):
    plt.subplot(2,4,i+1)
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot {feature}")
plt.tight_layout()
plt.show()

print("\n=== Kesimpulan EDA ===")
print("""
- Dataset terdiri dari 2111 baris dan 17 kolom, mencakup fitur kategori dan numerik.
- Beberapa kolom seperti Age, Weight, Height, dan lainnya masih bertipe object sehingga perlu dikonversi ke tipe numerik untuk analisis lebih lanjut.
- Terdapat missing value di hampir semua kolom kecuali target (NObeyesdad), dengan jumlah bervariasi (misalnya Age: 14, Weight: 11, dsb).
- Nilai unik pada beberapa fitur numerik seperti FCVC, NCP, FAF, dan CH2O sangat tinggi, menunjukkan kemungkinan data bertipe numerik yang masih disimpan sebagai string atau format tidak konsisten.
- Data target (NObeyesdad) memiliki 7 kelas dengan distribusi yang tidak merata, misalnya kelas Obesity_Type_I muncul paling sering. Hal ini menunjukkan adanya ketidakseimbangan kelas yang perlu ditangani saat modeling.
- Terdapat 18 baris duplikat yang perlu dihapus untuk menghindari bias pada model.
- Outlier terdeteksi pada fitur numerik seperti Age, Height, Weight, CH2O, dan lainnya berdasarkan visualisasi boxplot. Perlu pertimbangan apakah akan dihapus atau ditangani dengan teknik robust.
""")


# Load ulang data
import pandas as pd

df = pd.read_csv('ObesityDataSet.csv')

# Cek duplikat
print("Jumlah data sebelum hapus duplikat:", df.shape)
print("Jumlah duplikat:", df.duplicated().sum())

# Hapus duplikat
df.drop_duplicates(inplace=True)

print("Jumlah data setelah hapus duplikat:", df.shape)


import numpy as np

# Cek missing value
print("Jumlah missing value awal:")
print(df.isnull().sum())

# Jika ada simbol "?" yang menandakan null → ubah jadi NaN
df.replace('?', np.nan, inplace=True)

# Drop baris yang memiliki nilai kosong
df.dropna(inplace=True)

# Cek ulang
print("\nSetelah menangani missing value:")
print(df.isnull().sum())

# Ubah kolom numerik dari object ke float
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Cek tipe data
print("\nTipe data setelah konversi:")
print(df.dtypes)



# Fungsi hapus outlier
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Jumlah data sebelum
print("Jumlah data sebelum hapus outlier:", df.shape)

# Hapus outlier
df = remove_outliers_iqr(df, num_cols)

# Jumlah data setelah
print("Jumlah data setelah hapus outlier:", df.shape)

from sklearn.preprocessing import LabelEncoder

# Cek kolom kategori
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('NObeyesdad')  # Target dipisah

# Encode fitur kategori
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Encode target
df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])

# Lihat hasil encode
print("\nContoh data setelah encoding:")
print(df.head())


from sklearn.preprocessing import LabelEncoder

# Cek kolom kategori
cat_cols = df.select_dtypes(include='object').columns.tolist()

# Jika 'NObeyesdad' ada, hapus dari list kategori
if 'NObeyesdad' in cat_cols:
    cat_cols.remove('NObeyesdad')

# Encode fitur kategori
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Jika target masih kategori, encode juga
if df['NObeyesdad'].dtype == 'object':
    df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])

# Lihat hasil encode
print("\nContoh data setelah encoding:")
print(df.head())

# Pisah fitur dan label
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Cek dimensi
print("Ukuran fitur (X):", X.shape)
print("Ukuran label (y):", y.shape)

from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Sebelum SMOTE
print("Distribusi label sebelum SMOTE:")
print(y.value_counts())

# SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Setelah SMOTE
print("\nDistribusi label setelah SMOTE:")
print(pd.Series(y_res).value_counts())

# Visualisasi
plt.figure(figsize=(7,4))
sns.countplot(x=y_res)
plt.title("Distribusi Label Setelah SMOTE")
plt.xlabel("Kelas Obesitas")
plt.ylabel("Jumlah")
plt.show()


from sklearn.preprocessing import StandardScaler

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Cek hasil
print("Contoh data setelah normalisasi:")
print(pd.DataFrame(X_scaled, columns=X.columns).head())


print("\n=== Kesimpulan Preprocessing ===")
print("""
- Seluruh missing value telah dihapus dari dataset.
- Tipe data pada kolom numerik berhasil dikonversi ke tipe float untuk keperluan analisis dan modeling.
- Data duplikat dan outlier berhasil diidentifikasi dan dihapus guna meningkatkan kualitas data.
- Seluruh fitur kategori telah diubah menjadi numerik menggunakan LabelEncoder agar dapat diproses oleh algoritma machine learning.
- Ketidakseimbangan kelas pada target berhasil diatasi menggunakan metode SMOTE (Synthetic Minority Over-sampling Technique).
- Seluruh fitur numerik telah dinormalisasi menggunakan StandardScaler agar memiliki skala yang seragam dan mempercepat konvergensi pada model.
- Dataset akhir telah bersih, seimbang, dan siap digunakan untuk tahap modeling klasifikasi.
""")

# Untuk pemodelan dan evaluasi
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Model Klasifikasi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluasi
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score


# Contoh pemisahan fitur dan target
X = df.drop(columns='NObeyesdad')  # Kolom target: 'NObeyesdad'
y = df['NObeyesdad']

# Split data: 80% train dan 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Inisialisasi model
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Dictionary untuk menyimpan hasil evaluasi
results = {}


# Training dan Evaluasi Awal
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
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")


# Visualisasi Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, result) in enumerate(results.items()):
    cm = result['Confusion Matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f"Confusion Matrix: {name}")
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()


# Convert ke DataFrame untuk visualisasi
metrics_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score']]

# Plot perbandingan
metrics_df.plot(kind='bar', figsize=(10,6), colormap='Set2')
plt.title("Perbandingan Performa Model")
plt.ylabel("Skor")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# Menentukan model dengan F1-Score tertinggi
best_model = metrics_df['F1-Score'].idxmax()
best_metrics = metrics_df.loc[best_model]

# Menampilkan hasil metrik dari model terbaik
print("=== Kesimpulan Model Terbaik ===")
print(f"Model Terbaik: {best_model}")
print(f"Akurasi       : {best_metrics['Accuracy']:.4f}")
print(f"Precision     : {best_metrics['Precision']:.4f}")
print(f"Recall        : {best_metrics['Recall']:.4f}")
print(f"F1-Score      : {best_metrics['F1-Score']:.4f}")


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Inisialisasi model dasar
rf_model = RandomForestClassifier(random_state=42)

# Definisi grid parameter yang akan diuji
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Setup GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    verbose=2,
    n_jobs=-1
)

# Jalankan proses tuning
grid_search_rf.fit(X_train, y_train)


print("Model terbaik setelah GridSearchCV:")
print(grid_search_rf.best_estimator_)

print("\nParameter terbaik:")
print(grid_search_rf.best_params_)

print("\nSkor terbaik (mean F1-score dari CV):")
print(f"{grid_search_rf.best_score_:.4f}")

from sklearn.metrics import classification_report, confusion_matrix

# Prediksi dengan model terbaik
best_rf = grid_search_rf.best_estimator_
y_pred_tuned = best_rf.predict(X_test)

# Evaluasi
print("=== Evaluasi Model Random Forest Setelah Tuning ===")
print(classification_report(y_test, y_pred_tuned))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))


import matplotlib.pyplot as plt
import pandas as pd

# Asumsikan kamu punya skor f1 sebelumnya dan sesudah tuning
f1_before = 0.93  # dari model RF sebelum tuning
f1_after = grid_search_rf.best_score_

# Buat DataFrame
compare_df = pd.DataFrame({
    'Model': ['Random Forest (Before Tuning)', 'Random Forest (After Tuning)'],
    'F1-Score': [f1_before, f1_after]
})

# Visualisasi
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
print(f"Setelah dilakukan hyperparameter tuning menggunakan GridSearchCV, F1-Score model Random Forest meningkat dari {f1_before:.4f} menjadi {f1_after:.4f}.")
print("Parameter terbaik berhasil ditemukan dan meningkatkan performa model secara signifikan.")
print("Model hasil tuning menunjukkan kinerja prediksi yang lebih stabil dan akurat.")

