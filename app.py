import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Obesity ML App", layout="wide")
st.title("🏥 Obesity Classification Web App")

menu = st.sidebar.selectbox("Navigasi", ["Beranda", "EDA", "Preprocessing", "Modeling", "Tuning", "Prediksi Manual"])

@st.cache_data
def load_data():
    return pd.read_csv("ObesityDataSet.csv")

raw_df = load_data()
df = raw_df.copy()

# --- Preprocessing ---
df.drop_duplicates(inplace=True)
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

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

cat_cols = df.select_dtypes(include='object').columns.tolist()
label_maps = {}
if 'NObeyesdad' in cat_cols:
    cat_cols.remove('NObeyesdad')
for col in cat_cols:
    le = LabelEncoder()
    le.fit(raw_df[col])
    df[col] = le.transform(df[col])
    label_maps[col] = dict(enumerate(le.classes_))
    label_maps[col + "_inverse"] = {v: k for k, v in label_maps[col].items()}

le_target = LabelEncoder()
df['NObeyesdad'] = le_target.fit_transform(df['NObeyesdad'])

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Validasi sebelum SMOTE
if X is not None and y is not None and not X.empty and not y.empty:
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        st.write("✅ SMOTE berhasil dilakukan.")
    except Exception as e:
        st.error(f"❌ Gagal melakukan SMOTE: {str(e)}")
        X_res, y_res = X, y
else:
    st.warning("⚠️ Data tidak valid untuk SMOTE. Menggunakan data asli.")
    X_res, y_res = X, y

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)

# Modeling
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'model': model,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

# Tuning
rf_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)

# --- Menu ---
if menu == "Beranda":
    st.header("🩺 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dikembangkan untuk memprediksi status obesitas seseorang berdasarkan atribut gaya hidup dan kebiasaan makan.
    """)

elif menu == "EDA":
    st.header("🔍 Exploratory Data Analysis (EDA)")
    st.dataframe(raw_df.head())
    st.write(raw_df.describe(include='all'))
    st.text(raw_df.info())
    st.write(raw_df.isnull().sum())
    st.write("Jumlah duplikat:", raw_df.duplicated().sum())
    st.subheader("Distribusi Kelas Target")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=raw_df, y='NObeyesdad', order=raw_df['NObeyesdad'].value_counts().index, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Boxplot Outlier")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=raw_df[num_cols], orient='h', ax=ax2)
    st.pyplot(fig2)

elif menu == "Preprocessing":
    st.header("⚙️ Preprocessing Data")
    st.write("Jumlah data setelah preprocessing:", df.shape)
    st.write("Kolom kategori yang diencode:", cat_cols)
    st.write("Distribusi kelas setelah SMOTE:")
    st.bar_chart(pd.Series(y_res).value_counts())
    st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())

elif menu == "Modeling":
    st.header("📊 Modeling dan Evaluasi")
    metrics_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
    st.dataframe(metrics_df)

    st.subheader("Perbandingan Visual")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    metrics_df.plot(kind='bar', ax=ax3)
    plt.xticks(rotation=0)
    st.pyplot(fig3)

    st.subheader("Confusion Matrix")
    for name, result in results.items():
        st.text(name)
        st.write(result['Confusion Matrix'])

elif menu == "Tuning":
    st.header("🧪 Hyperparameter Tuning")
    st.write("Parameter terbaik:", grid_search.best_params_)
    st.write("F1-Score terbaik:", f"{grid_search.best_score_:.4f}")

    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    st.subheader("Confusion Matrix After Tuning")
    st.write(cm_tuned)

    f1_before = results['Random Forest']['F1-Score']
    f1_after = grid_search.best_score_
    compare_df = pd.DataFrame({
        'Model': ['Random Forest (Before)', 'Random Forest (Tuned)'],
        'F1-Score': [f1_before, f1_after]
    })
    fig4, ax4 = plt.subplots()
    sns.barplot(data=compare_df, x='Model', y='F1-Score', palette='Blues', ax=ax4)
    st.pyplot(fig4)

elif menu == "Prediksi Manual":
    st.header("🧮 Prediksi Obesitas Manual")
    manual_input = {}
    for col in X.columns:
        if col in cat_cols:
            label_dict = label_maps[col]
            selected = st.selectbox(col, list(label_dict.values()))
            manual_input[col] = label_maps[col + "_inverse"][selected]
        else:
            manual_input[col] = st.slider(
                col, float(df[col].min()), float(df[col].max()), float(df[col].mean())
            )

    if st.button("Prediksi"):
        input_array = np.array([list(manual_input.values())])
        input_scaled = scaler.transform(input_array)
        pred_class = best_rf.predict(input_scaled)[0]
        pred_label = le_target.inverse_transform([pred_class])[0]
        st.success(f"✅ Hasil Prediksi: **{pred_label}**")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    # Jalankan Streamlit lewat CLI dari dalam Python
    os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")

