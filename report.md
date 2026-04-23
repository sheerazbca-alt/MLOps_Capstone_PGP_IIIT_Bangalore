📊 MLOps Capstone Project – Predictive Maintenance System
🔍 Overview

This project implements an end-to-end MLOps pipeline to predict machine failures using sensor data from industrial equipment. The goal is to identify whether a machine is operating normally or likely to fail, and classify the specific failure type.

The pipeline is designed with a strong focus on reliability, reproducibility, and monitoring, following industry-standard MLOps practices.

🧱 Pipeline Architecture

The system integrates multiple stages into a unified workflow:

Data validation using Pandera
Feature engineering based on physical relationships
Model training and comparison with MLflow
Hyperparameter tuning using Optuna
Drift detection using statistical methods
Model explainability using SHAP
📂 Data Description

Three datasets were used:

train.csv → Historical labelled data for training
current.csv → Stable production-like data
stress.csv → Drifted data simulating heavy-load conditions
🧪 Data Validation & EDA
A strict schema was defined using Pandera to ensure data consistency across all datasets.
Validation confirmed that all datasets are structurally correct.
Exploratory Data Analysis revealed:
Significant class imbalance in the target variable
Some failure types have very limited samples
Distribution plots highlighted variations in key features such as rotational speed and torque.
⚙️ Feature Engineering

Two domain-driven features were introduced:

Power_W = Torque × (Rotational Speed × 2π / 60)
Temp_diff = Process Temperature − Air Temperature

These features improved model interpretability and captured physical machine behavior.

🤖 Model Training & Selection
Data was split using stratified sampling (80/20)
SMOTE (k_neighbors=3) was applied only on training data to handle imbalance
Four models were trained and tracked:
Logistic Regression
Decision Tree
Random Forest
XGBoost
📌 Evaluation Strategy
Macro F1-score was used as the primary metric
Accuracy was avoided as it is misleading for imbalanced datasets
🏆 Best Model
XGBoost achieved the highest macro F1-score
Selected as the final model
🔧 Hyperparameter Tuning
Optuna was used to optimize:
Number of estimators
Tree depth
Learning rate
The tuned model showed improved generalization performance
📉 Drift Detection
Drift analysis compared incoming data with training data
Findings:
current.csv → No significant drift (stable)
stress.csv → Clear distribution shift
📌 Key Insight
Drift was observed in operational features such as:
Rotational speed
Torque
⚠️ Decision
Drift indicates potential degradation in model performance
Retraining is recommended under stress conditions
🔍 Model Explainability (SHAP)
SHAP analysis provided feature importance for each class
Key observations:
Different failure types are influenced by different feature patterns
Engineered features (Power_W, Temp_diff) were highly impactful
Explainability helped validate model behavior and support monitoring decisions
🧠 Key Insights
Class imbalance significantly affects model performance
Rare failure types are difficult to predict due to limited data
Macro F1-score provides a balanced evaluation across all classes
Drift detection is essential for maintaining production reliability
✅ Conclusion
XGBoost was selected as the best-performing model based on macro F1-score
Accuracy alone is insufficient due to class imbalance
The weakest class performance is due to data scarcity, not model limitations
Drift in stress conditions highlights the need for continuous monitoring
🚀 Recommendation

Implement a retraining pipeline triggered by drift detection, especially when significant shifts are observed in key operational features. This ensures the model remains reliable in changing real-world conditions.

📁 Generated Outputs
eda_distributions.png
class_distribution.png
drift_current.html
drift_stress.html
shap_per_class.png
best_model.pkl
label_encoder.pkl
