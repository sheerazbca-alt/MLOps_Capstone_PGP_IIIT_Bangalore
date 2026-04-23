# 🚀 MLOps Predictive Maintenance Pipeline

## 📌 Overview
This project builds an end-to-end MLOps pipeline to predict machine failures using sensor data. It focuses on reliability, reproducibility, and monitoring in a production-like setup.

---

## ⚙️ Key Features
- ✅ Data validation using Pandera  
- ✅ Feature engineering with domain logic  
- ✅ Model comparison using MLflow  
- ✅ Class imbalance handled with SMOTE  
- ✅ Hyperparameter tuning using Optuna  
- ✅ Drift detection on new data  
- ✅ Model explainability using SHAP  

---

## 🏆 Best Model
- **XGBoost** selected based on **Macro F1-score**  
- Macro F1 used due to highly imbalanced dataset  

---

## 📊 Key Insights
- Dataset is heavily imbalanced → accuracy is misleading  
- Rare failure types are hard to predict due to limited data  
- Drift detected in stress conditions → model performance risk  

---

## 📉 Drift Monitoring
- `current.csv` → Stable  
- `stress.csv` → Significant drift detected  

👉 Retraining is recommended when drift is observed.

---

## 📁 Outputs
- `eda_distributions.png`  
- `class_distribution.png`  
- `drift_current.html`  
- `drift_stress.html`  
- `shap_per_class.png`  
- `best_model.pkl`  
- `label_encoder.pkl`  

---

## 🛠️ Tech Stack
Python, Pandas, Scikit-learn, XGBoost, MLflow, Optuna, SHAP, Pandera

---

