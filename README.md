# Fall Risk Prediction in Long-Term Care Using Machine Learning and SHAP Explainability

This project develops a machine-learning model to predict the percentage of long-stay residents experiencing one or more falls with major injury in U.S. long-term care facilities. The workflow includes exploratory data analysis (EDA), multicollinearity assessment using Variance Inflation Factor (VIF), model comparison across multiple regressors, and SHAP-based explainability to identify the strongest predictors of fall risk.
The goal is to build an interpretable, data-driven tool that supports quality improvement, risk monitoring, and operational decision-making in long-term care settings.


---

##  **1. Exploratory Data Analysis (EDA)**

The EDA notebook (`eda.ipynb`) examines:

- Distribution of all variables  
- Missingness patterns (all < 1.2%)  
- Outliers in clinical and staffing metrics  
- Correlation structure and multicollinearity  
- Pairwise relationships among key predictors  

Key findings:

- The target variable (`falls_with_injury`) is right-skewed with meaningful outliers.  
- Rating variables (overall, health, staffing, quality) are highly collinear.  
- Clinical predictors (UTI %, catheter %, pressure ulcers, mobility decline) show realistic skewness.  
- Weak linear correlations with the target suggest a non-linear modeling approach.

The cleaned dataset was saved as **`df_final.csv`** for modeling.

---

##  **2. Feature Reduction Using VIF**

Variance Inflation Factor (VIF) was used to detect multicollinearity.

Variables removed due to extremely high VIF:

- `health_rating`  
- `staffing_rating`  
- `quality_rating`  
- `nurse_hours_total`  
- `adl_increase_pct`  

These removals improved model stability and interpretability.

---

##  **3. Model Training and Evaluation**

Eight regression models were trained and evaluated:

- Linear Regression  
- Lasso  
- Ridge  
- Random Forest  
- Gradient Boosting  
- Extra Trees  
- XGBoost  
- KNN Regressor  

**Performance Metrics:** RMSE and R² on the test set.

**Best Model:**  
### Extra Trees Regressor  
- RMSE: **1.16**  
- R²: **0.16**  

Tree-based models outperformed linear models, confirming the presence of non-linear relationships.

The best model and scaler were saved as:

- `best_model_extra_trees.pkl`  
- `scaler.pkl`

---

##  **4. SHAP Explainability**

SHAP was used to interpret the Extra Trees model.

### **Top Predictors (Global Importance)**

1. UTI rate (`uti_pct`)  
2. Incontinence rate (`incontinence_pct`)  
3. Catheter use (`catheter_pct`)  
4. Mobility decline (`mobility_decline_pct`)  
5. Staff turnover (`staff_turnover_pct`)  
6. Overall facility rating (`overall_rating`)  
7. Pressure ulcers (`pressure_ulcers`)  
8. Weight loss (`weight_loss_pct`)  

### **Key Insights**

- Higher UTI, incontinence, catheter use, and mobility decline strongly increase predicted fall risk.  
- Higher staff turnover is associated with elevated fall risk.  
- Higher overall facility rating is protective.  
- SHAP dependence plots reveal non-linear and interaction effects not captured by simple correlations.

Figures are stored in the `figures/` directory.

---

##  **5. Why This Matters**

Falls with major injury are a critical quality indicator in long-term care.  
This project demonstrates:

- A reproducible ML pipeline  
- Transparent model interpretation  
- Clinically meaningful insights  
- A foundation for real-world decision support tools  

Hence supporting:

- Quality improvement teams  
- State-level oversight  
- Facility benchmarking  
- Predictive risk monitoring   

---

## ▶ **6. How to Run the Project**

1. Clone the repository  
2. Install dependencies: pip install -r requirements.txt 

3. Open the notebooks in Jupyter or VS Code  
4. Run `eda.ipynb`  
5. Run `models.ipynb`  

---

##  Contact

For questions or collaboration, feel free to reach out.




