# 💉 Diabetic Retinopathy Prognosis Prediction

This repository contains the data and analysis for a machine learning project focused on **predicting the prognosis of diabetic retinopathy** using patient physiological data.

The core analysis is performed in a Jupyter Notebook (`Diabetes_Prediction.ipynb`), which implements a classification workflow from data preprocessing through to optimized model evaluation, prioritizing **Recall** as the key performance metric for identifying high-risk patients.

---

## 📁 Repository Contents

| File Name | Description |
| :--- | :--- |
| `P600_pronostico_dataset.csv` | The raw dataset containing patient health indicators and the prognosis outcome. |
| `Diabetes_Prediction.ipynb` | The Jupyter Notebook with the complete machine learning pipeline, including model training, tuning, and evaluation. |

---

## 💾 Dataset: `P600_pronostico_dataset.csv`

The dataset is a semi-colon (`;`) separated CSV file. It contains 6,000 entries and the following six features:

| Column Name | Description | Data Type | Example Values |
| :--- | :--- | :--- | :--- |
| `ID` | Unique identifier for each patient. | `int64` | `0`, `1`, `2`... |
| `age` | Patient's age (in years). | `float64` | `77.19`, `63.52`... |
| `systolic_bp` | Patient's **Systolic Blood Pressure**. | `float64` | `85.28`, `99.37`... |
| `diastolic_bp` | Patient's **Diastolic Blood Pressure**. | `float64` | `80.02`, `84.85`... |
| `cholesterol` | Patient's **Cholesterol** level. | `float64` | `79.95`, `110.38`... |
| `prognosis` | **Target Variable**: The predicted outcome. | `object` (String) | `retinopathy`, `no_retinopathy` |

### Sample Data (`df.head()`)

| ID | age | systolic\_bp | diastolic\_bp | cholesterol | prognosis |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 0 | 77.196 | 85.289 | 80.022 | 79.957 | retinopathy |
| 1 | 63.530 | 99.380 | 84.852 | 110.382 | retinopathy |
| 2 | 69.004 | 111.349 | 109.851 | 100.828 | retinopathy |
| 3 | 82.638 | 95.056 | 79.667 | 87.066 | retinopathy |
| 4 | 78.346 | 109.155 | 90.713 | 92.512 | retinopathy |

---

## 💻 Analysis: `Diabetes_Prediction.ipynb`

This notebook details the machine learning approach:

### 1. Preprocessing and EDA
* Data cleaning, checking for missing values (none found).
* **Label Encoding** of the target variable (`prognosis`).
* **Standard Scaling** of the numerical features (`age`, `systolic_bp`, `diastolic_bp`, `cholesterol`).
* Splitting the data into training and testing sets.

### 2. Model Training and Selection
The notebook trains and compares the performance of several popular classification algorithms:
* **Logistic Regression**
* **Random Forest Classifier**
* **Support Vector Classifier (SVC)**
* **XGBoost Classifier**

### 3. Hyperparameter Optimization
* The best-performing model (inferred to be **SVC/SVM**) is selected for optimization.
* **`GridSearchCV`** and **`StratifiedKFold`** are used to tune the model's hyperparameters (e.g., `C` and `gamma` for SVM).
* The optimization metric is explicitly set to **`recall_score`** to ensure the model is highly effective at correctly identifying positive cases (`retinopathy`).

### 4. Results and Evaluation
The final evaluation includes:
* **Confusion Matrix** and **Classification Report**.
* Metrics such as **Accuracy, Precision, Recall, and F1-Score**.
* **ROC AUC Score** and **ROC Curve** visualization.

The output snippet confirms the optimization process:

> *...Performing hyperparameter tuning for best model: **SVM**...*
> *...Best parameters: **{'C': 0.1, 'class_weight': 'balanced'}**...*
> *...Optimized Model Performance: Recall (optimized): **0.8719**...*
