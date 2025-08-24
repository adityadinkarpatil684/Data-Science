# Loan-Approval-prediction System using Data-Science/ML
---

## 🔍 Overview
This project demonstrates a **complete end-to-end Machine Learning pipeline** to identify the best-performing model for classification.  
We experimented with multiple algorithms, applied **Optuna hyperparameter tuning**, and finalized **XGBoost** as the best-performing model.  

The project covers:
- Data cleaning, EDA, preprocessing  
- Model building & evaluation (multiple ML models)  
- Hyperparameter optimization with **Optuna**  
- Saving the final trained model  

---

## 📂 Dataset
The dataset used in this project is taken from **Kaggle**:  
👉 [Dataset Link](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)   

The dataset consists of multiple numerical and categorical features.  
Each column was analyzed for its distribution, correlations, and impact on the target variable.  

---

## 🛠 Methodology

### 1. Data Cleaning
- Checked for **missing values** and handled them appropriately.  
- Removed **duplicate rows** to ensure data consistency.  

---

### 2. Exploratory Data Analysis (EDA)
- Studied statistical parameters like **mean, median, mode**.  
- Conducted **CFA (Confirmatory Factor Analysis)**.  
- Described each column in detail.  
- Analyzed **correlation and covariance** between features.  
- Visualized **correlation heatmap** for better feature understanding.  

---

### 3. Data Preprocessing
- Detected and removed **outliers**.  
- Encoded **categorical variables**.  
- Applied **feature scaling** where necessary.  
- Performed **feature importance analysis**.  
- Checked and handled **class imbalance** (oversampling/undersampling if required).  

---

### 4. Model Evaluation
#### 📌 Phase 1 → `Model_1.ipynb`
- Trained and evaluated the following models:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - K-Nearest Neighbors (KNN)  
  - XGBoost  

- Evaluated models using:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  

- Applied **K-Fold Cross Validation** to overcome overfitting.  
- Compared **training vs testing performance** to check for overfitting.  

---

### 5. Hyperparameter Tuning
#### 📌 Phase 2 → `Model_2.ipynb`
- Selected **top-performing models** (Logistic Regression, Random Forest, XGBoost).  
- Applied **Optuna Hyperparameter Optimization**.  
- Tuned key parameters:  
  - `n_estimators`  
  - `max_depth`  
  - `learning_rate`  

---

### 6. Final Model Selection
#### 📌 Phase 3 → `Model_final.ipynb`
- Selected **XGBoost** as the **best model** with optimal hyperparameters.  
- Trained and validated the model.  
- Exported the model using **Joblib** for future deployment.  

---

## 📓 Project Notebooks
- `Model_1.ipynb` → Initial model evaluation with multiple ML models.  
- `Model_2.ipynb` → Hyperparameter tuning with Optuna.  
- `Model_final.ipynb` → Final model selection & export.  

---

## 🏆 Results
- Best-performing model: **XGBoost**  
