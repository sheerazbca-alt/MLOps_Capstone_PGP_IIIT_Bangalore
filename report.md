# 📊 MLOps Capstone Project – Predictive Maintenance System

## 🔍 Overview
This project implements an end-to-end MLOps pipeline to predict machine failures using sensor data from industrial equipment. The objective is to classify whether a machine is operating normally or likely to fail, and identify the specific failure type.

The pipeline is designed with a strong focus on **reliability, reproducibility, and monitoring**, following industry-standard MLOps practices.

---

## 🧱 Pipeline Architecture
The system integrates multiple components into a unified workflow:

- Data validation using Pandera  
- Feature engineering based on physical relationships  
- Model training and comparison with MLflow  
- Hyperparameter tuning using Optuna  
- Drift detection using statistical methods  
- Model explainability using SHAP  

---

## 📂 Data Description
Three datasets were used:

- **train.csv** → Historical labelled data for training  
- **current.csv** → Stable production-like data  
- **stress.csv** → Drifted data simulating heavy-load conditions  

---

## 🧪 Data Validation & EDA
- A schema was defined using Pandera to ensure consistency across datasets.
- All datasets were validated successfully before model usage.
- Exploratory Data Analysis revealed:
  - Significant **class imbalance** in the target variable  
  - Some failure types have very limited samples  
- Distribution plots highlighted variations in key features such as rotational speed, torque, and temperature.

---

## ⚙️ Feature Engineering
Two domain-specific features were created:

- **Power_W** = Torque × (Rotational Speed × 2π / 60)  
- **Temp_diff** = Process Temperature − Air Temperature  

These features improve interpretability and capture underlying physical relationships in machine operations.

---

## 🤖 Model Training & Selection
- Data was split using stratified sampling (80/20)
- SMOTE (k_neighbors=3) was applied only on training data to handle class imbalance
- Four models were trained and tracked:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost  

### 📌 Evaluation Strategy
- **Macro F1-score** was used as the primary metric  
- Accuracy was not used as the main metric due to class imbalance  

### 🏆 Best Model
- **XGBoost** achieved the highest macro F1-score  
- Selected as the final model  

---

## 🔧 Hyperparameter Tuning
- Optuna was used for hyperparameter optimization  
- Key parameters tuned:
  - Number of estimators  
  - Maximum depth  
  - Learning rate  

The tuned model improved overall performance and generalization.

---

## 📉 Drift Detection
- Drift analysis compared incoming datasets with training data

### Findings:
- **current.csv** → No significant drift (stable data)  
- **stress.csv** → Significant distribution shift observed  

### Key Drifted Features:
- Rotational speed  
- Torque  

### Decision:
- Drift indicates potential performance degradation  
- **Model retraining is recommended under stress conditions**

---

## 🔍 Model Explainability (SHAP)
- SHAP was used to interpret model predictions
- Key insights:
  - Different failure types are influenced by different features  
  - Engineered features (Power_W, Temp_diff) are highly impactful  
- Explainability helps validate model behavior and supports monitoring decisions  

---

## 🧠 Key Insights
- Class imbalance significantly impacts model performance  
- Rare failure classes are difficult to predict due to limited data  
- Macro F1-score provides a balanced evaluation across all classes  
- Drift detection is essential for maintaining model reliability in production  

---

## ✅ Conclusion
- **XGBoost** is the best-performing model based on macro F1-score  
- Accuracy is misleading due to imbalanced class distribution  
- Poor performance on rare classes is due to **data scarcity**, not model limitations  
- Drift in stress conditions highlights the need for continuous monitoring  

---

## 🚀 Recommendation
Implement an automated **retraining pipeline triggered by drift detection**, especially when significant shifts occur in critical features. This ensures the model remains reliable in real-world scenarios.

---

## 📁 Generated Outputs
- `eda_distributions.png`  
- `class_distribution.png`  
- `drift_current.html`  
- `drift_stress.html`  
- `shap_per_class.png`  
- `best_model.pkl`  
- `label_encoder.pkl`  

---
