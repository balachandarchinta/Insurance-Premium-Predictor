# 🏥 Insurance Premium Predictor

## 📌 Project Overview

This project focuses on predicting insurance premium prices using customer demographic and medical condition data. The workflow covers Exploratory Data Analysis (EDA), hypothesis testing, feature engineering, and machine learning modeling, followed by deployment-ready artifacts.

---

## 🎯 Problem Statement

The objective is to **predict the insurance premium price** (PremiumPrice) for individuals based on:

* Age and demographic attributes
* Medical conditions (e.g., diabetes, chronic diseases, transplants)
* Physical attributes (height, weight, BMI)
* Medical history (surgeries, history of cancer in family)

---

## 🎯 Target Metric

The models are evaluated using:

* **RMSE (Root Mean Squared Error)**
* **R2 Score**
* **Adjusted R2 Score**

These metrics are used to measure prediction accuracy and generalization performance.

---

🔍 **Steps Followed**
**1. Data Loading & Cleaning**
* Dataset loaded using Pandas
* No missing values detected
* Data types validated

---

**2. 📊 Exploratory Data Analysis (EDA)**

**Key steps involved:**
* Created **Age Groups** for analysis
* Boxplots to study premium distribution across:
  * Age groups
  * Medical conditions (with vs without)
* Correlation heatmap to identify relationships

**Key insights**:
* Age is a strong driver of premium (correlation ≈ 0.69)
* Chronic diseases and transplants significantly increase premiums
* Premium pricing follows band/step-based structure, not continuous
* Mid-age groups (25–44) show highest variability

---

**3. Feature Engineering**

* Created age groups(binning)
  * Condition score (based on weighted impact of conditions)
  * RiskScore (0–100 normalized)
  * BMI from height & weight
* Count of medical conditions (NumConditions)

**Insights:**
* Premium increases with risk but not linearly
* Pricing saturates at high-risk levels (ceiling effect)

---

**4. Hypothesis Testing**

**T-Test (Condition vs Premium)**
* Most conditions (except allergies) are statistically significant (p < 0.05)
* Chronic diseases show highest differentiation

**Chi-Square Test (Age vs Conditions)**
* Strong association for:
  * Chronic diseases
  * Diabetes
  * Blood pressure
* Transplants are rare but high-impact

---

**5. Outlier Analysis**

* IQR method used to detect outliers
* Observed fixed premium values (~39K–40K) suggesting:
  * High-risk bucket or data anomalies

---

**6. Modelling Approach**

**Data Preparation:**
* Dropped leakage features:
  * RiskScore, ConditionScore, AgeScore were calculated.
* Removed non-significant feature:
  * KnownAllergies
* Standard scaling applied
* Train-test split: **80-20**

---

**🤖 Machine Learning Models**

**1. Linear Regression**
* RMSE: **3361.17**
* R²: **0.69**

**2. Random Forest (Tuned)**
* RMSE: **2674.34**
* R²: **0.81**

**Hyperparameters:**
* n_estimators = **300**
* max_depth = **7**
* min_samples_split = **5**
* min_samples_leaf = 5****

**3. XGBoost (Best Performing Model)**
* RMSE: **2635.32**
* R²: **0.81**
* Tuned parameters:
  * n_estimators = **500**
  * learning_rate = **0.01**
  * max_depth = **3**
  * subsample = **0.8**
  * colsample_bytree = **0.8**

**Key Observations:**
* Initial model showed overfitting → resolved via tuning
* Best balance between bias and variance achieved

**4. Neural Network**
* R²: **0.53**
* Underperformed compared to tree-based models

**📊 Model Validation**
* Cross-validation R²: **0.77–0.84**
* Standard deviation low → stable model
* Breusch-Pagan test confirmed **heteroscedasticity**
* Residual analysis shows:
  * Underprediction for high-risk individuals

**🔍 Feature Importance Insights**

Top drivers (from XGBoost & SHAP):
* Age
* Any Transplants
* Chronic Diseases
* Weight
* Number of Surgeries

**💡 Key Business Insights**
* Premium pricing is tier-based, not continuous
* High-risk individuals hit a pricing ceiling
* Chronic diseases and transplants are primary cost drivers
* Age acts as a major baseline risk factor
* BMI and weight are modifiable risk indicators

**📦 Deployment Artifacts**

The notebook generates and exports:
* Model file: insurance_model.pkl
* Scaler file: scaler.pkl
These files are used for deployment by using Streamlit app.

**🚀 Deployment Steps**
* Trained the model in notebook
* Saving artifacts using joblib
* Load the model & scaler in deployment script
* Accept user inputs
* Scaling inputs using saved scaler
* Predicting the premium using trained model

**📁 Project Contents**
* Python notebook with:
  * EDA
  * Hypothesis testing
  * Feature engineering
  * Model building & evaluation
* Deployment files:
  * Trained model (.pkl)
  * Scaler (.pkl)
---

**✅ Final Conclusion**
- Tree-based models (Random Forest & XGBoost) significantly outperform linear models
- XGBoost (tuned) provides the best performance
- The model captures non-linear, step-based pricing behavior effectively
- Dataset shows strong structure with low multicollinearity and stable validation performance

## 👤 Author

**Bala Chandar Kumar Chinta**
