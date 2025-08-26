# Model Selection for Heart Arrhythmia Prediction using AIC

This project provides a **Model Selection Framework** in Python for evaluating and comparing regression models (Linear, Ridge, and Lasso) using the **Akaike Information Criterion (AIC)**.  
It is specifically applied to **heart arrhythmia prediction datasets**, enabling selection of the most suitable model for identifying irregular heart rhythms from patient features.

---

## 📌 Features
- **AIC Calculation** for Linear, Ridge, Lasso, and Logistic models.  
- **k-Fold Cross-Validation** for robust evaluation across different data splits.  
- **Bootstrap Validation** to measure stability of models on resampled datasets.  
- **Statistical Summaries** (mean, variance, standard deviation of AIC scores).  
- **Visualizations**:
  - Horizontal bar plots of mean AIC with error bars.  
  - Boxplots showing distribution of AIC scores.  
- **Generic Process Function** for evaluating models on any dataset, including health datasets like arrhythmia.

---

## 📊 Akaike Information Criterion (AIC)
The **Akaike Information Criterion (AIC)** helps choose models that balance **fit** and **complexity**:

\[
AIC = 2k - 2\ln(L)
\]

Where:  
- \( k \) = number of model parameters  
- \( L \) = maximum likelihood of the model  

👉 Lower AIC indicates a better model for predicting heart arrhythmia events.

---

## 🏥 Heart Arrhythmia Prediction Use Case
Heart arrhythmia refers to irregular heartbeats that can indicate serious cardiac conditions.  
Using features such as **age, heart rate, blood pressure, ECG readings**, etc., this framework helps:

- Evaluate multiple regression models for predicting heart irregularities.  
- Select the model with the best trade-off between **accuracy and complexity**.  
- Quantify variability using **k-Fold cross-validation** and **bootstrap validation**.  

This approach ensures robust, reliable model selection for **clinical decision support systems**.

---

## 🛠️ Installation & Requirements
Ensure Python 3.7+ and install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
