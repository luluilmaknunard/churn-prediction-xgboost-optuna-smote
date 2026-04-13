# Customer Churn Prediction using XGBoost, SMOTE, and Optuna

## 📌 Project Overview
Project ini merupakan studi kasus **Customer Churn Prediction** menggunakan dataset Telco Customer Churn dari Kaggle.  
Tujuan utama dari project ini adalah memprediksi pelanggan yang berpotensi berhenti berlangganan (churn) serta memahami faktor-faktor yang mempengaruhi keputusan tersebut.

Metode yang digunakan dalam project ini adalah:
- SMOTE untuk mengatasi data imbalance
- XGBoost untuk klasifikasi
- Optuna untuk hyperparameter tuning
- SHAP untuk model interpretability

---

## 📊 Dataset
Dataset yang digunakan:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset berisi informasi pelanggan seperti:
- Tenure (lama berlangganan)
- Monthly Charges
- Total Charges
- Jenis layanan (Internet, Phone, dll)
- Contract type
- Churn (target variable)

---

## ⚙️ Workflow
Tahapan yang dilakukan dalam project ini:

### 1. **Data Loading & Understanding**
- Membaca dataset menggunakan pandas
- Mengecek struktur data, tipe data, dan missing values
- Mengidentifikasi distribusi churn (imbalanced dataset)

---

### 2. **Exploratory Data Analysis (EDA)**
- Analisis distribusi churn
- Membandingkan karakteristik customer churn vs non-churn
- Visualisasi hubungan antar fitur

---

### 3. **Data Preprocessing**
- Menghapus kolom tidak relevan (`customerID`)
- Konversi `TotalCharges` ke numerik
- Menangani missing values

---

### 4. **Feature Engineering**
- Membuat fitur baru:
  - `ChargePerTenure` → rata-rata biaya per lama berlangganan
- Menghitung jumlah layanan yang digunakan pelanggan

---

### 5. **Encoding**
- Mengubah fitur kategorikal menjadi numerik menggunakan Label Encoding

---

### 6. **Handling Imbalanced Data (SMOTE)**
- Dataset churn tidak seimbang
- Menggunakan SMOTE untuk menyeimbangkan kelas churn dan non-churn

---

### 7. **Model Training (XGBoost)**
- Membagi data:
  - 80% training
  - 20% testing
- Melatih model XGBoost untuk klasifikasi churn

---

### 8. **Hyperparameter Tuning (Optuna)**
- Mencari parameter terbaik secara otomatis
- Meningkatkan performa model

---

### 9. **Model Evaluation**
Model berhasil memprediksi churn dengan performa yang cukup baik setelah dilakukan penanganan imbalance menggunakan SMOTE dan tuning dengan Optuna.

### 🔹 Model Performance
+----------------+---------+
| Metric | Score |
+----------------+---------+
| Accuracy | 0.7622 |
| Precision | 0.5409 |
| Recall | 0.6898 |
| F1-Score | 0.6063 |
| AUC-ROC | 0.8216 |
+----------------+---------+

### 🔹 Classification Report
+--------------+-----------+--------+----------+---------+
| Class | Precision | Recall | F1-Score | Support |
+--------------+-----------+--------+----------+---------+
| No Churn | 0.88 | 0.79 | 0.83 | 1035 |
| Churn | 0.54 | 0.69 | 0.61 | 374 |
+--------------+-----------+--------+----------+---------+

| Accuracy | | 0.76 | 1409 |
| Macro Avg | 0.71 | 0.74 | 0.72 | 1409 |
| Weighted Avg | 0.79 | 0.76 | 0.77 | 1409 |
+--------------+-----------+--------+----------+---------+
---

### 10. **Model Explainability (SHAP)**
- Mengetahui fitur paling berpengaruh
- Memahami bagaimana model mengambil keputusan

---

## 📈 Results
- Model berhasil memprediksi churn dengan performa yang baik
- Setelah menggunakan SMOTE, model menjadi lebih seimbang dalam mendeteksi churn
- Optuna membantu meningkatkan performa model secara signifikan

Evaluasi model:
