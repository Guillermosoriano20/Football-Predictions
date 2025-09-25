# ⚽ LaLiga Match Predictions with R

This repository contains a full pipeline in **R** for predicting match outcomes and key statistics in **LaLiga** using historical data, advanced feature engineering, and machine learning models.  

The project combines **statistical rigor** with **real-world sports analytics**, showing how mathematics and data science can be applied to football forecasting.

---

## 📂 Project Structure
- **`datos_LaLiga_estilos.csv`** → Input dataset with match statistics and team styles.  
- **`predicciones_futbol.R`** → Main R script containing the full pipeline.  
- **`Predicciones.csv`** → Example output with predicted match results.  
- **`mejor_modelo_por_variable.csv`** → Model evaluation summary (rolling validation).  

---

## 🛠️ Methods & Models
The pipeline follows these steps:

1. **Data preprocessing**  
   - Merge match-level features (team vs rival).  
   - Compute rolling averages (_L5) with NA-safe shrinkage.  
   - Engineer interactions (attack × defense).  
   - Add tactical styles (dummy-encoded).  

2. **Target variables**  
   - Goals (local & visitor).  
   - Shots.  
   - Yellow cards.  

3. **Regression models per variable**  
   - Generalized Linear Models (Poisson / Negative Binomial).  
   - Random Forest Regressors.  
   - XGBoost Regressors.  

4. **Classification models (match outcome)**  
   - Random Forest Classifier.  
   - XGBoost Multiclass Classifier.  

5. **Validation**  
   - Rolling validation (time-series split).  
   - Mean Absolute Error (MAE) for each model × variable.  
   - Automatic selection of the best-performing model.  

---
