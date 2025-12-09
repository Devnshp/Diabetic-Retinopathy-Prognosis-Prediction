# 👁️ Diabetic Retinopathy Prognosis Prediction

This repository provides the data and analysis for a machine learning project focused on predicting the **prognosis of diabetic retinopathy** using patient health indicators.

The main objective of the project is to build an accurate classification model, with a specific focus on maximizing **Recall** to ensure high sensitivity in detecting the presence of retinopathy (reducing false negatives).

---

## 📁 Repository Contents

| File Name | Purpose |
| :--- | :--- |
| `P600_pronostico_dataset.csv` | **Input Data**: Contains 6,000 patient records with physiological measurements and the final diagnosis (`prognosis`). |
| `Diabetes_Prediction.ipynb` | **Analysis Workflow**: The Jupyter Notebook detailing the entire machine learning pipeline, from data preparation to model optimization. |

---

## 📊 Data Overview: `P600_pronostico_dataset.csv`

This dataset contains crucial health metrics used to predict the risk of retinopathy.

| Feature | Description | Data Type | Role in Model |
| :--- | :--- | :--- | :--- |
| `ID` | Unique patient identifier. | Integer | Index (not used for training) |
| **`age`** | Patient's age in years. | Float | Feature |
| **`systolic_bp`** | Systolic Blood Pressure. | Float | Feature |
| **`diastolic_bp`** | Diastolic Blood Pressure. | Float | Feature |
| **`cholesterol`** | Cholesterol level. | Float | Feature |
| **`prognosis`** | **Target Variable**: The diagnostic outcome, which is either `retinopathy` or `no_retinopathy`. | String | Target |

---

## 🧠 Machine Learning Workflow: `Diabetes_Prediction.ipynb`

The Jupyter Notebook executes a standard yet rigorous machine learning pipeline:

### 1. Data Preparation
* **Preprocessing**: Features are prepared for modeling, including **Standard Scaling** of numerical inputs and **Label Encoding** of the binary target variable (`prognosis`).
* **Splitting**: The data is partitioned into training and testing sets to validate model performance on unseen data.

### 2. Model Training and Comparison
The workflow compares multiple classification algorithms to find the best baseline predictor:
* **Logistic Regression**
* **Random Forest Classifier**
* **Support Vector Classifier (SVC)**
* **XGBoost Classifier**

### 3. Hyperparameter Optimization (Focused Tuning)
* The best initial model (**Support Vector Machine - SVM**) is chosen for fine-tuning.
* **`GridSearchCV`** with **`StratifiedKFold`** cross-validation is used to systematically search for the optimal hyperparameters (e.g., `C` and `class_weight`).
* **Crucially, the scoring metric is set to `recall`**. This emphasizes the model's ability to minimize false negatives (patients with retinopathy being missed), which is critical for a medical diagnosis task.

### 4. Results and Evaluation
The final model performance is rigorously assessed using:
* **Recall**: The primary metric, showing the proportion of actual positive cases (`retinopathy`) that were correctly identified.
* **Accuracy, Precision, F1-Score**.
* **Confusion Matrix** and **Classification Report**.
* **ROC AUC Score** and curve analysis.

The tuning process achieved significant improvement in the critical metric:
> *Best parameters for SVM: `{'C': 0.1, 'class_weight': 'balanced'}`*
> *Optimized **Recall**: **0.8719***
